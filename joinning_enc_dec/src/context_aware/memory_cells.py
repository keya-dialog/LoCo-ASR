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


class MemoryCell(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Initialize (0th time stamp) memory cells as parameters + positional embeddings (MxH)
        self.memory_init = torch.randn(1, config.memory_dim, config.hidden_size)
        nn.init.normal_(self.memory_init)
        self.memory_init = nn.Parameter(self.memory_init)

        self.memory_positional_embeddings = nn.Embedding(config.memory_dim, config.hidden_size)

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
        self.context_holder.bind_vector_initializer(self.memory_init)

    def get_current_memory_state(self, prev_memory_state, prev_hidden_states):
        # Equation 4: Mt-1 = Mt-1 + PE(Mt-1)
        memory_state = self.memory_positional_embeddings(prev_memory_state)

        """Compute current memory state Mt"""
        # Equation 2: Mt-1_tilde = AddNorm(MHA(Mt-1, ht-1, ht-1))
        residual = memory_state
        temporal_memory_state, _ = self.update_memory(memory_state, prev_hidden_states, prev_hidden_states)
        temporal_memory_state = self.update_norm1(temporal_memory_state + residual)

        # Equation 3: Mt = AddNorm(FF(Mt-1_tilde))
        residual = temporal_memory_state
        current_memory_state = self.update_ff(temporal_memory_state)
        current_memory_state = self.update_norm2(current_memory_state + residual)
        return current_memory_state

    def forward(self, hidden_states, attention_mask):
        """Actualize memory state"""
        prev_memory_state, prev_hidden_states, prev_mask = self.context_holder.get_prev_state()

        if None not in prev_hidden_states:
            prev_hidden_states = torch.vstack(prev_hidden_states)
            """Update prev memory state with positional embeddings"""
            current_memory_state = self.get_current_memory_state(prev_memory_state, prev_hidden_states)
        else:
            current_memory_state = torch.stack([prev_memory_state[
                                                    index] if item is None else self.get_current_memory_state(
                prev_memory_state[index], item) for index, item in enumerate(prev_hidden_states)])

        """Compute modified hidden states ht_tilde"""
        residual = hidden_states
        # Equation 5: ht_tilde = MHA(ht, Mt, Mt)
        hidden_states, _ = self.output_attention(hidden_states, current_memory_state, current_memory_state,
                                                 attn_mask=attention_mask)
        hidden_states = hidden_states + residual
        hidden_states = self.output_norm(hidden_states)

        """Save current states for next step"""
        self.context_holder.update_context_vectors(current_memory_state, hidden_states)

        return hidden_states
