from typing import Any, Dict, Iterator, List, Optional, Sized, TYPE_CHECKING, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, DataLoader, Dataset
from transformers import Seq2SeqTrainer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    is_fairscale_available,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    nested_detach,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    has_length,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

# Integrations must be imported before ML frameworks:

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    pass

if is_torch_tpu_available(check_device=False):
    pass

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

# Integrations must be imported before ML frameworks:

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    pass

from torch.utils.data import Sampler

if is_torch_tpu_available(check_device=False):
    pass

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_nested_concat

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

logger = logging.get_logger("transformers")

from collections import Counter


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

        conversations = Counter(conv_ids)
        dependent_samples = {conv: [] for conv in conversations}
        for index, (conv_id, turn_index) in enumerate(zip(conv_ids, turn_idxs)):
            dependent_samples[conv_id].append((turn_index, index))

        self.dependent_samples = [[tup[1] for tup in sorted(dependent_samples[conv], key=lambda x: x[0], reverse=True)]
                                  for conv in
                                  dependent_samples.keys()]
        self.initial_weights = torch.tensor(list(conversations.values()), dtype=torch.float)
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
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
                yield from [self.dependent_samples[conv].pop() for conv
                            in selected_convs if len(self.dependent_samples[conv]) > 0]

    def __len__(self) -> int:
        return self.num_samples


class ContextAwareTrainer(Seq2SeqTrainer):

    def _get_context_container(self):
        context_container_index = self.callback_handler.callback_list.split().index('ContextContainerCallback')
        return self.callback_handler.callbacks[context_container_index]

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        conv_ids = self.train_dataset[self.args.conv_ids_column_name]
        turn_idxs = self.train_dataset[self.args.turn_index_column_name]

        # Build the sampler.
        if self.args.group_by_length:
            raise Exception("Not implemented yet")
        else:
            if self.args.world_size <= 1:
                return RandomSamplerWithDependency(self.train_dataset,
                                                   self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                                   conv_ids=conv_ids,
                                                   turn_idxs=turn_idxs, generator=generator)
            elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
            ):
                raise Exception("Not implemented yet")
            else:
                raise Exception("Not implemented yet")

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        context_container = self._get_context_container()
        conv_ids = inputs.pop("conv_ids")
        inputs = context_container.add_context_vectors(inputs, conv_ids)

        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        context_container.update_context_vectors(outputs.encoder_context_vectors, conv_ids=conv_ids)
        return loss.detach()

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        conv_ids = inputs.pop("conv_ids")
        context_container = self._get_context_container()
        inputs = context_container.add_context_vectors(inputs, conv_ids)

        if not self.args.predict_with_generate or prediction_loss_only:
            has_labels = False if len(self.label_names) == 0 else all(
                inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels or loss_without_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():
                if is_sagemaker_mp_enabled():
                    raw_outputs = smp_forward_only(model, inputs)
                    if has_labels or loss_without_labels:
                        if isinstance(raw_outputs, dict):
                            loss_mb = raw_outputs["loss"]
                            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                        else:
                            loss_mb = raw_outputs[0]
                            logits_mb = raw_outputs[1:]

                        loss = loss_mb.reduce_mean().detach().cpu()
                        logits = smp_nested_concat(logits_mb)
                    else:
                        loss = None
                        if isinstance(raw_outputs, dict):
                            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                        else:
                            logits_mb = raw_outputs
                        logits = smp_nested_concat(logits_mb)
                else:
                    if has_labels or loss_without_labels:
                        with self.compute_loss_context_manager():
                            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        loss = loss.mean().detach()

                        if isinstance(outputs, dict):
                            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                        else:
                            logits = outputs[1:]
                    else:
                        loss = None
                        with self.compute_loss_context_manager():
                            outputs = model(**inputs)
                        if isinstance(outputs, dict):
                            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                        else:
                            logits = outputs
                        # TODO: this needs to be fixed and made cleaner later.
                        if self.args.past_index >= 0:
                            self._past = outputs[self.args.past_index - 1]

            context_container.update_context_vectors(outputs.encoder_context_vectors, conv_ids=conv_ids)

            if prediction_loss_only:
                return (loss, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            return (loss, logits, labels)

        has_labels = "labels" in inputs

        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)
        if "context_vectors" in inputs:
            gen_kwargs["context_vectors"] = inputs.get("context_vectors", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        context_container.update_context_vectors(outputs.encoder_context_vectors, conv_ids=conv_ids)

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                    gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        context_container = self._get_context_container()
        context_container.on_evaluate_begin(self.model, dataloader.dataset, self.args.conv_ids_column_name)

        outputs = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        context_container.on_evaluate_end()
        return outputs
