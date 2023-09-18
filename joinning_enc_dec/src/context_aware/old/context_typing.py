from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, Seq2SeqLMOutput, Wav2Vec2BaseModelOutput


@dataclass
class Seq2SeqLMOutputLosses(Seq2SeqLMOutput):
    enc_loss: Optional[torch.FloatTensor] = None
    dec_loss: Optional[torch.FloatTensor] = None
    new_context_vectors: Optional[Tuple[torch.FloatTensor]] = None
    new_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithContext(CausalLMOutput):
    new_context_vectors: Optional[Tuple[torch.FloatTensor]] = None
    new_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Wav2Vec2BaseModelOutputWithContext(Wav2Vec2BaseModelOutput):
    new_context_vectors: Optional[Tuple[torch.FloatTensor]] = None
    new_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithContext(BaseModelOutput):
    new_context_vectors: Optional[Tuple[torch.FloatTensor]] = None
    new_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
