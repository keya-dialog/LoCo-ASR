from typing import Optional

import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder, Wav2Vec2EncoderLayer, \
    Wav2Vec2EncoderLayerStableLayerNorm, Wav2Vec2EncoderStableLayerNorm

from context_aware.memory_cells import MemoryCell


class Wav2Vec2EncoderLayerStableLayerNormWithContext(Wav2Vec2EncoderLayerStableLayerNorm):
    def __init__(self, config):
        super().__init__(config)
        self.context_combiner = MemoryCell(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        """Insertion of context combiner (output & update attention) + Add & Norm"""
        hidden_states = self.context_combiner(self.final_layer_norm(hidden_states), attention_mask=attention_mask)

        hidden_states = hidden_states + self.feed_forward(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def activate_memory_params(self):
        for param in self.context_combiner.parameters():
            param.requires_grad = True

    def reset_memory(self):
        self.context_combiner.reset_memory()

    def connect_context_container(self, context_container):
        self.context_combiner.connect_context_container(context_container)


class Wav2Vec2EncoderLayerWithContext(Wav2Vec2EncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.context_combiner = MemoryCell(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)

        """Insertion of context combiner (output & update attention) + Add & Norm"""
        hidden_states = self.context_combiner(hidden_states, attention_mask=attention_mask)

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def activate_memory_params(self):
        for param in self.context_combiner.parameters():
            param.requires_grad = True

    def reset_memory(self):
        self.context_combiner.reset_memory()

    def connect_context_container(self, context_container):
        self.context_combiner.connect_context_container(context_container)


class Wav2Vec2EncoderStableLayerNormWithContext(Wav2Vec2EncoderStableLayerNorm):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNormWithContext(
                config) if index in config.memory_cells else Wav2Vec2EncoderLayerStableLayerNorm(config) for
             index in
             range(config.num_hidden_layers)]
        )

    def activate_memory_params(self):
        for layer in self.layers:
            if hasattr(layer, "activate_memory_params"):
                layer.activate_memory_params()

    def reset_memory(self):
        for layer in self.layers:
            if hasattr(layer, "reset_memory"):
                layer.reset_memory()

    def connect_context_container(self, context_container):
        for layer in self.layers:
            if hasattr(layer, "connect_context_container"):
                layer.connect_context_container(context_container)


class Wav2Vec2EncoderWithContext(Wav2Vec2Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerWithContext(
                config) if index in config.memory_cells else Wav2Vec2EncoderLayer(config) for
             index in
             range(config.num_hidden_layers)]
        )

    def activate_memory_params(self):
        for layer in self.layers:
            if hasattr(layer, "activate_memory_params"):
                layer.activate_memory_params()

    def reset_memory(self):
        for layer in self.layers:
            if hasattr(layer, "reset_memory"):
                layer.reset_memory()

    def connect_context_container(self, context_container):
        for layer in self.layers:
            if hasattr(layer, "connect_context_container"):
                layer.connect_context_container(context_container)


class Wav2Vec2ModelWithContext(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNormWithContext(config)
        else:
            self.encoder = Wav2Vec2EncoderWithContext(config)

        # Initialize weights and apply final processing
        self.post_init()

    def activate_memory_params(self):
        self.encoder.activate_memory_params()

    def reset_memory(self):
        self.encoder.reset_memory()

    def connect_context_container(self, context_container):
        self.encoder.connect_context_container(context_container)


class Wav2Vec2ConfigWithContext(Wav2Vec2Config):
    model_type = "wav2vec2-with-context"


class Wav2Vec2WithContextForCTC(Wav2Vec2ForCTC):
    config_class = Wav2Vec2ConfigWithContext

    def __init__(self, config: Wav2Vec2ConfigWithContext):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2ModelWithContext(config)

        # Initialize weights and apply final processing
        self.post_init()

    def activate_memory_params(self):
        self.wav2vec2.activate_memory_params()

    def reset_memory(self):
        self.wav2vec2.reset_memory()

    def connect_context_container(self, context_container):
        self.wav2vec2.connect_context_container(context_container)
