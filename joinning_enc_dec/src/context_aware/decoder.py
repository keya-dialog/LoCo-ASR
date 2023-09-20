from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config, GPT2LMHeadModel, GPT2Model

from context_aware.memory_cells import MemoryCell


class GPT2BlockWithContext(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.context_combiner = MemoryCell(config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        hidden_states = self.attention_adapters(attn_output, residual, None)

        """Insertion of context combiner (output & update attention) + Add & Norm"""
        hidden_states = self.context_combiner(hidden_states, attention_mask=attention_mask)

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = self.output_adapters(feed_forward_hidden_states, residual, None)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

    def activate_memory_params(self):
        for param in self.context_combiner.parameters():
            param.requires_grad = True

    def reset_memory(self):
        self.context_combiner.reset_memory()

    def connect_context_container(self, context_container):
        self.context_combiner.connect_context_container(context_container)

    def expand_context_states(self, expand_size):
        self.context_combiner.expand_context_states(expand_size)


class GPT2ModelWithContext(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList(
            [GPT2BlockWithContext(config, layer_idx=i) if i in config.memory_cells else GPT2Block(config, layer_idx=i)
             for i in range(config.num_hidden_layers)])

    def activate_memory_params(self):
        for layer in self.h:
            if hasattr(layer, "activate_memory_params"):
                layer.activate_memory_params()

    def reset_memory(self):
        for layer in self.h:
            if hasattr(layer, "reset_memory"):
                layer.reset_memory()

    def connect_context_container(self, context_container):
        for layer in self.h:
            if hasattr(layer, "connect_context_container"):
                layer.connect_context_container(context_container)

    def expand_context_states(self, expand_size):
        for layer in self.h:
            if hasattr(layer, "connect_context_container"):
                layer.expand_context_states(expand_size)



class GPT2ConfigWithContext(GPT2Config):
    model_type = "gpt2-with-context"


class GPT2WithContextLMHeadModel(GPT2LMHeadModel):
    config_class = GPT2ConfigWithContext

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelWithContext(config)

    def reset_memory(self):
        self.transformer.reset_memory()

    def activate_memory_params(self):
        self.transformer.activate_memory_params()

    def connect_context_container(self, context_container):
        self.transformer.connect_context_container(context_container)

    def expand_context_states(self, expand_size):
        self.transformer.expand_context_states(expand_size)
