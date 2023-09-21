from collections import Counter
from typing import Iterator, List, Optional, Sized

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler


class RandomSamplerWithDependency(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, batch_size: int, conv_ids: Optional[List[str]] = None,
                 turn_idxs: Optional[List[str]] = None, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

        self.conversations = Counter(conv_ids)
        self.dependent_samples = {conv: [] for conv in self.conversations}
        for index, (conv_id, turn_index) in enumerate(zip(conv_ids, turn_idxs)):
            self.dependent_samples[conv_id].append((turn_index, index))
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        dependent_samples = [[tup[1] for tup in sorted(self.dependent_samples[conv], key=lambda x: x[0], reverse=True)]
                             for conv in
                             self.dependent_samples.keys()]
        self.initial_weights = torch.tensor(list(self.conversations.values()), dtype=torch.float)

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            raise Exception("Not implemented yet")
        else:
            weights = self.initial_weights
            for _ in range(self.num_samples // self.batch_size):
                selected_convs = torch.multinomial(weights, self.batch_size, generator=generator)
                weights_update = torch.zeros_like(weights)
                weights_update[selected_convs] = 1
                weights -= weights_update
                # return rand element of dataset in case conversation is already empty
                weights = torch.clip(weights, 0)
                yield from [dependent_samples[conv].pop() for conv
                            in selected_convs if len(dependent_samples[conv]) > 0]

    def __len__(self) -> int:
        return self.num_samples


class ContextHolder:
    def __init__(self):
        self.current_conversations = None
        self.current_context_vectors = []
        self.current_hidden_states = []
        self.hidden_states = {}
        self.context_vectors = {}
        self.memory_initializer = None
        self.hidden_initializer = None

    def bind_memory_initializer(self, memory_initializer):
        self.memory_initializer = memory_initializer

    def bind_hidden_initializer(self, hidden_initializer):
        self.hidden_initializer = hidden_initializer

    def reset_prediction_state(self, conversation_ids):
        self.current_conversations = conversation_ids
        for _ in conversation_ids:
            self.current_context_vectors.append(self.memory_initializer.squeeze(dim=0)[None, ...])
            self.current_hidden_states.append(self.hidden_initializer.squeeze(dim=0))
        self.current_context_vectors = torch.vstack(self.current_context_vectors)
        self.current_hidden_states = pad_sequence(self.current_hidden_states, batch_first=True)

    def prepare_current_vectors(self, conversation_ids):
        self.current_conversations = conversation_ids
        for conversation_id in self.current_conversations:
            self.current_context_vectors.append(
                self.context_vectors.get(conversation_id, self.memory_initializer.squeeze(dim=0))[None, ...])
            self.current_hidden_states.append(
                self.hidden_states.get(conversation_id, self.hidden_initializer.squeeze(dim=0)))
        self.current_context_vectors = torch.vstack(self.current_context_vectors)
        self.current_hidden_states = pad_sequence(self.current_hidden_states, batch_first=True)

    def expand_context_states(self, expand_size):
        self.current_context_vectors = self.current_context_vectors.repeat_interleave(expand_size, dim=0)
        self.current_hidden_states = self.current_hidden_states.repeat_interleave(expand_size, dim=0)

    def get_prev_state(self):
        # TODO: Fix this trick for beam decoding
        if isinstance(self.current_context_vectors, list):
            for conversation_id in self.current_conversations:
                self.current_context_vectors.append(
                    self.context_vectors.get(conversation_id, self.memory_initializer.squeeze(dim=0))[None, ...])
                self.current_hidden_states.append(
                    self.hidden_states.get(conversation_id, self.hidden_initializer.squeeze(dim=0)))
            self.current_context_vectors = torch.vstack(self.current_context_vectors)
            self.current_hidden_states = pad_sequence(self.current_hidden_states, batch_first=True)
        return self.current_context_vectors, self.current_hidden_states

    def update_context_vectors(self, memory_states, hidden_states, hidden_lens):
        for conversation_id, memory_state, hidden_state, hidden_len in zip(self.current_conversations, memory_states,
                                                                           hidden_states, hidden_lens):
            self.context_vectors[conversation_id] = memory_state.clone()
            self.hidden_states[conversation_id] = hidden_state[:hidden_len].clone()
        self.current_context_vectors = []
        self.current_hidden_states = []


class ContextContainer:
    def __init__(self, model_config):
        self.context_holders = [ContextHolder() for _ in
                                range(len(model_config.encoder.memory_cells) + len(model_config.decoder.memory_cells))]
        self.layer = 0

    def get_holder(self):
        holder = self.context_holders[self.layer]
        self.layer += 1
        return holder

    def prepare_current_vectors(self, conversation_ids):
        for holder in self.context_holders:
            holder.prepare_current_vectors(conversation_ids)

    def reset_prediction_state(self, conversation_ids):
        for holder in self.context_holders:
            holder.reset_prediction_state(conversation_ids)
