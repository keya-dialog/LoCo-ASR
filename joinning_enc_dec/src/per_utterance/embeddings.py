import math

import torch
from torch import Tensor, nn


def scale_hook(module, __, output):
    return output * math.sqrt(module.embedding_dim)


class PositionalEmbeddingWPELike(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.var_scaler = torch.nn.Parameter(torch.ones(1))
        self.mean_scaler = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_pos: Tensor) -> Tensor:
        """
        Arguments:
            x_pos: Tensor, shape ``[1, seq_len]``
        """
        return self.var_scaler * self.pe[:x_pos.size(1)] - self.mean_scaler
