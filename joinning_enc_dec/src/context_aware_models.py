from typing import Optional

import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.utils import logging

logger = logging.get_logger("transformers")


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        att_w = self.softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class Wav2Vec2WithContextV1(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.utterance_pool = SelfAttentionPooling(self.config.hidden_size)
        self.context_pool = SelfAttentionPooling(self.config.hidden_size)

    def forward(
            self,
            input_values: Optional[torch.Tensor],
            context_prev: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

    ) -> torch.Tensor:
        enc_hidden_states = self.encoder.forward(input_values, attention_mask, mask_time_indices, output_attentions,
                                                 output_hidden_states,
                                                 return_dict)
        enc_hidden_states_pooled = self.utterance_pool(enc_hidden_states)
        context_curr = self.context_pool(torch.vstack((enc_hidden_states_pooled, context_prev)))
        return torch.vstack((enc_hidden_states_pooled, context_curr))


class Wav2Vec2WithContextV2(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.utterance_pool = SelfAttentionPooling(self.config.hidden_size)
        self.context_pool = SelfAttentionPooling(self.config.hidden_size)

    def forward(
            self,
            input_values: Optional[torch.Tensor],
            context_prev: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

    ) -> torch.Tensor:
        enc_hidden_states = self.encoder.forward(input_values, attention_mask, mask_time_indices, output_attentions,
                                                 output_hidden_states,
                                                 return_dict)
        enc_hidden_states_pooled = self.utterance_pool(enc_hidden_states)
        context_curr = self.context_pool(torch.vstack((enc_hidden_states_pooled, context_prev)))
        return torch.vstack((enc_hidden_states_pooled, context_curr))
