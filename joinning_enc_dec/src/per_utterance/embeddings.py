import math

import torch
from torch import Tensor, nn


class PositionalEmbeddingWPELike(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x_pos: Tensor) -> Tensor:
        """
        Arguments:
            x_pos: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[:x_pos.size(1)]
