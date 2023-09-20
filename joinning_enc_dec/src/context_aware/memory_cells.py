import torch
from torch import nn
from transformers.activations import ACT2FN

from context_aware.utils import ContextContainer


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, mem_length, hidden_size):
        super().__init__()

        self.emb = nn.Embedding(mem_length, hidden_size)
        self.register_buffer('position', torch.arange(mem_length, dtype=torch.long).unsqueeze(0))

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, mem):
        return self.layer_norm(self.emb(self.position) + mem)


class MemoryCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize (0th time stamp) memory cells as parameters + positional embeddings (MxH)
        self.memory_init = torch.randn(1, config.memory_dim, config.hidden_size)
        nn.init.normal_(self.memory_init)
        self.memory_init = nn.Parameter(self.memory_init)

        self.hidden_init = torch.randn(1, 0, config.hidden_size)
        self.hidden_init = nn.Parameter(self.hidden_init)

        self.memory_positional_embeddings = LearnablePositionalEmbedding(config.memory_dim, config.hidden_size)

        # Initialize update memory
        self.update_attention = nn.MultiheadAttention(config.hidden_size, num_heads=config.num_attention_heads,
                                                      dropout=config.attention_dropout, batch_first=True)
        self.update_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.update_ff = FeedForward(config)
        self.update_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Initialize output memory
        self.output_attention = nn.MultiheadAttention(config.hidden_size, num_heads=config.num_attention_heads,
                                                      dropout=config.attention_dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.context_holder = None

    def connect_context_container(self, context_container: ContextContainer):
        self.context_holder = context_container.get_holder()
        self.context_holder.bind_memory_initializer(self.memory_init)
        self.context_holder.bind_hidden_initializer(self.hidden_init)

    def forward(self, hidden_states, attention_mask):
        """Actualize memory state"""
        prev_memory_state, prev_hidden_states = self.context_holder.get_prev_state()

        """Update prev memory state with positional embeddings"""
        # Equation 4: Mt-1 = Mt-1 + PE(Mt-1)
        memory_state = self.memory_positional_embeddings(prev_memory_state)

        """Compute current memory state Mt"""
        # Equation 2: Mt-1_tilde = AddNorm(MHA(Mt-1, ht-1, ht-1))
        residual = memory_state
        temporal_memory_state, _ = self.update_attention(memory_state, prev_hidden_states, prev_hidden_states)
        temporal_memory_state = self.update_norm1(temporal_memory_state + residual)

        # Equation 3: Mt = AddNorm(FF(Mt-1_tilde))
        residual = temporal_memory_state
        current_memory_state = self.update_ff(temporal_memory_state)
        current_memory_state = self.update_norm2(current_memory_state + residual)

        """Compute modified hidden states ht_tilde"""
        residual = hidden_states
        # Equation 5: ht_tilde = MHA(ht, Mt, Mt)
        if attention_mask is not None:
            attention_mask_MHA = attention_mask.squeeze(dim=1)[..., :self.config.memory_dim].repeat_interleave(
                self.output_attention.num_heads, dim=0)
        else:
            attention_mask_MHA = None
        hidden_states, _ = self.output_attention(hidden_states, current_memory_state, current_memory_state,
                                                 attn_mask=attention_mask_MHA)
        hidden_states = hidden_states + residual
        hidden_states = self.output_norm(hidden_states)

        """Save current states for next step"""
        if attention_mask is not None:
            hidden_lens = attention_mask.size(-1) - (attention_mask < 0).sum(dim=-1).squeeze(dim=1)[:, 0]
        else:
            hidden_lens = torch.full((hidden_states.size(0),), hidden_states.size(1))
        self.context_holder.update_context_vectors(current_memory_state.detach(), hidden_states.detach(), hidden_lens)

        return hidden_states
