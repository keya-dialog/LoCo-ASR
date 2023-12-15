from typing import List, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN


class FilterWeights(nn.Module):
    """Module for calculating adaptive weights for each channel of the input."""

    def __init__(self, input_dim):
        super(FilterWeights, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep: torch.FloatTensor) -> torch.FloatTensor:
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        return:
            filter_w : size (N, T, 1)
        """
        if torch.prod(torch.tensor(batch_rep.size()[2:])) != torch.prod(torch.tensor(self.W.weight.size()[1:])):
            batch_rep = nn.functional.pad(batch_rep, (0, self.W.in_features - batch_rep.size(-1)), mode="constant",
                                          value=0)

        filter_w = nn.functional.sigmoid(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        return filter_w


class MelFeatureExtractorAdaptive(nn.Module):
    """Module for extracting features from mel spectrogram using stack of conv layers with adaptive weights."""

    def __init__(self, config):
        super().__init__()
        self.conv = torch.nn.Sequential(
            *[nn.Sequential(nn.Conv2d(conv_in, out_channels=conv_out, kernel_size=(conv_kernel, conv_kernel),
                                      stride=(conv_stride, conv_stride)),
                            ACT2FN[config.feat_extract_activation]) for
              conv_in, conv_out, conv_kernel, conv_stride in
              zip([1, *config.conv_dim], config.conv_dim, config.conv_kernel,
                  config.conv_stride)],
        )

        # Check if chunk size is greater than product of kernel sizes
        self.min_conv_input_size = torch.prod(torch.tensor([conv_kernel for conv_kernel in zip(config.conv_kernel)]))
        if config.fe_chunk_size < self.min_conv_input_size:
            raise ValueError(f"Chunk size must be greater than product of kernel sizes: {self.min_conv_input_size}")

        self.config = config
        # Calculate input shapes for each conv layer and receptive fields
        self.input_shapes = self._get_input_shape(sizes=(config.fe_chunk_size, config.num_mel_bins),
                                                  strides=config.conv_stride,
                                                  kernel_sizes=config.conv_kernel, scale_x=True)
        self.proj_shape = self._get_input_shape((config.conv_dim[-1], config.num_mel_bins), strides=config.conv_stride,
                                                kernel_sizes=config.conv_kernel, scale_x=False)[-1]
        self.receptive_field = self.calculate_receptive_field_global(config.conv_kernel, config.conv_stride)
        self.global_stride = torch.prod(torch.tensor(config.conv_stride))

        # Initialized adaptive layers and output projection
        self.weight_fncs = nn.ModuleList([FilterWeights(shape[0] * shape[1]) for shape in self.input_shapes])
        self.out = nn.Linear(self.proj_shape[0] * self.proj_shape[1], config.hidden_size, bias=True)

    @staticmethod
    def _get_input_shape(sizes: Tuple[int, int], strides: List[int], kernel_sizes: List[int], scale_x: bool = False) \
            -> List[List[int]]:
        x_size, y_size = sizes
        x_sizes = [x_size]
        y_sizes = [y_size]
        for i in range(len(strides)):
            if scale_x:
                x_sizes.append((x_sizes[-1] - (kernel_sizes[i] - strides[i])) // strides[i])
            else:
                x_sizes.append(x_sizes[-1])
            y_sizes.append((y_sizes[-1] - (kernel_sizes[i] - strides[i])) // strides[i])
        return list(zip(x_sizes, y_sizes))

    @staticmethod
    def calculate_receptive_field_global(kernel_sizes: List[int], stride_sizes: List[int]) -> int:
        """Calculate global receptive field"""
        receptive_field = kernel_sizes[0]
        for kernel_size, stride in zip(kernel_sizes[1:], stride_sizes[1:]):
            receptive_field = receptive_field + (kernel_size - 1) + stride
        return receptive_field

    def find_unprocessed_segments_start(self, input_len: torch.IntTensor) -> torch.IntTensor:
        """Find start of unprocessed segments based on the size of the global perceptive field and global stride"""
        result = input_len - (((input_len - self.receptive_field) // self.global_stride) + 1) * self.global_stride
        return result

    def prepare_chunks(self, input_values: torch.FloatTensor) -> List[torch.FloatTensor]:
        """Split input values into chunks of size chunk_size"""
        x_chunked = list(input_values.unsqueeze(dim=1).split(self.config.fe_chunk_size, dim=2))
        if x_chunked[-1].size(1) < self.min_conv_input_size:
            last_chunk = x_chunked.pop()
            x_chunked[-1] = torch.cat([x_chunked[-1], last_chunk], dim=2)
        return x_chunked

    def forward(self, input_values: torch.FloatTensor) -> torch.FloatTensor:
        """Chunk input values and apply stack of conv layers with adaptive weights for each channel"""
        x_chunked = self.prepare_chunks(input_values)
        alphas = [torch.ones(conv.weight.size(0), device=input_values.device) for (conv, _) in self.conv]
        chunks_processed = []
        non_processed = 0
        for index, chunk in enumerate(x_chunked):
            # Prepend previously unprocessed part of the chunk
            if index != 0:
                chunk = torch.cat([x_chunked[index - 1][:, :, -non_processed:, ...], chunk], dim=2)
            # Find start of unprocessed part of the current chunk
            non_processed = self.find_unprocessed_segments_start(chunk.size(2))

            # Apply stack of conv layers
            for l_index, (conv_layer, activation) in enumerate(self.conv):
                # Apply conv layer and emphasize important channels by adaptive weights
                chunk = activation(conv_layer(chunk)) * alphas[l_index].unsqueeze(-1).unsqueeze(-1)
                # Alphas acts as residuals
                alphas[l_index] = 1 + self.weight_fncs[l_index](chunk.flatten(2, 3)).squeeze(-1).mean(dim=0)
            chunks_processed.append(self.out(chunk.transpose(1, 2).flatten(2, 3)))
        hidden_states = torch.concat(chunks_processed, dim=1).transpose(1, 2)
        return hidden_states


class MelFeatureExtractor(nn.Module):
    """Module for extracting features from mel spectrogram using stack of conv layers."""

    def __init__(self, config):
        super().__init__()
        self.conv = torch.nn.Sequential(
            *[nn.Sequential(nn.Conv2d(conv_in, out_channels=conv_out, kernel_size=(conv_kernel, conv_kernel),
                                      stride=(conv_stride, conv_stride)),
                            ACT2FN[config.feat_extract_activation]) for
              conv_in, conv_out, conv_kernel, conv_stride in
              zip([1, *config.conv_dim], config.conv_dim, config.conv_kernel,
                  config.conv_stride)],
        )

        linear_in_dim = config.conv_dim[-1] * (((config.num_mel_bins - 1) // 2 - 1) // 2)
        self.out = torch.nn.Linear(linear_in_dim, config.hidden_size, bias=True)

    def forward(self, input_values: torch.FloatTensor) -> torch.FloatTensor:
        """Apply stack of conv layers and linear layer to input values"""
        hidden_states = self.conv(input_values[:, None, ...])
        hidden_states = self.out(hidden_states.transpose(1, 2).flatten(2, 3))
        return hidden_states.transpose(1, 2)
