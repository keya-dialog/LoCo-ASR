from dataclasses import dataclass, field, make_dataclass
from dataclasses import dataclass, field, make_dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from jiwer import cer, compute_measures
from transformers import BatchFeature, PreTrainedTokenizerFast, Seq2SeqTrainer, SpeechEncoderDecoderModel, \
    TrainerCallback, TrainerControl, TrainerState, TrainingArguments, Wav2Vec2FeatureExtractor
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import logging

logger = logging.get_logger("transformers")


def compute_metrics(tokenizer, pred):
    pred_ids = pred.predictions

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else '-' for label in
                 tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    metrics = compute_measures(label_str, pred_str)
    del metrics['ops']
    del metrics['truth']
    del metrics['hypothesis']

    return {"cer": cer(label_str, pred_str), **metrics}


class FrozenLayersManager(TrainerCallback):
    def __init__(self, enc_layers_to_freeze, dec_layers_to_freeze, steps_to_freeze_enc, steps_to_freeze_dec,
                 freeze_cross_attention=False, freeze_others=False, callbacks=None):
        super().__init__()
        self.enc_layers_to_freeze = enc_layers_to_freeze
        self.dec_layers_to_freeze = dec_layers_to_freeze
        self.steps_to_freeze_enc = steps_to_freeze_enc
        self.steps_to_freeze_dec = steps_to_freeze_dec
        self.freeze_cross_attention = freeze_cross_attention
        self.freeze_others = freeze_others
        self.callbacks = callbacks

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        curr_model: SpeechEncoderDecoderModel = kwargs['model']
        curr_model.train()
        curr_model.encoder.train()
        curr_model.decoder.train()
        if self.enc_layers_to_freeze > 0:
            for name, param in curr_model.encoder.named_parameters():
                if name.startswith("wav2vec2.encoder.layers"):
                    layer = int(name.split('.')[3])
                    if layer < self.enc_layers_to_freeze:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif self.enc_layers_to_freeze > 0 and name.startswith("wav2vec2.encoder"):
                    param.requires_grad = False
                elif 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = not self.freeze_others

        if self.dec_layers_to_freeze > 0:
            for name, param in curr_model.decoder.named_parameters():
                if name.startswith("transformer.h."):
                    if 'cross' in name and not self.freeze_cross_attention:
                        param.requires_grad = True
                    elif 'adapter' in name:
                        param.requires_grad = True
                    else:
                        layer = int(name.split('.')[2])
                        if layer < self.dec_layers_to_freeze:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                else:
                    param.requires_grad = not self.freeze_others

        if self.freeze_others:
            curr_model.freeze_feature_encoder()
        curr_model.decoder.lm_head.weight.requires_grad = not self.freeze_others

        if self.callbacks:
            for callback in self.callbacks:
                callback()

        logger.debug(str([n for n, p in curr_model.named_parameters() if p.requires_grad]))
        logger.info(
            f'Model info: total parameters - {curr_model.num_parameters()}, trainable parameters - '
            f'{sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, curr_model.parameters())])}')

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        curr_model: SpeechEncoderDecoderModel = kwargs['model']
        if state.global_step == self.steps_to_freeze_enc:
            logger.info(f'Step: {state.global_step} encoder unfrozen.')
            self.reactivate_params(curr_model, curr_model.encoder.parameters())

        if state.global_step == self.steps_to_freeze_dec:
            logger.info(f'Step: {state.global_step} decoder unfrozen.')
            self.reactivate_params(curr_model, curr_model.decoder.parameters())

    @staticmethod
    def reactivate_params(curr_model, params_to_activate):
        for param in params_to_activate:
            param.requires_grad = True
        logger.debug([n for n, p in curr_model.named_parameters() if p.requires_grad])
        logger.info(
            f'Model info: total parameters - {curr_model.num_parameters()}, trainable parameters - '
            f'{sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, curr_model.parameters())])}')


class AdditionalLossPrinterCallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        state.__class__ = make_dataclass('state_derived',
                                         [('additional_logs', List[List[float]], field(default_factory=list))],
                                         bases=(TrainerState,))
        state.additional_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(state, 'additional_logs') and len(state.additional_logs) > 0:
            enc_loss, dec_loss = torch.tensor(state.additional_logs).mean(axis=0)
            if state.is_local_process_zero:
                logs['enc_loss'] = float(enc_loss)
                logs['dec_loss'] = float(dec_loss)
            state.additional_logs = []


class AdditionalLossTrackerTrainer(Seq2SeqTrainer):
    """Custom trainer to log both losses"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if hasattr(self.state, 'additional_logs'):
            self.state.additional_logs.append([outputs.enc_loss.mean(), outputs.dec_loss.mean()])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def audio_object_stripper(audio, key="array"):
    return audio[key] if isinstance(audio, dict) and key in audio else audio


@dataclass
class Seq2SeqDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizerFast`)
            The processor used for processing the text data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    Based upon: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/
                /Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: PreTrainedTokenizerFast
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: str = None
    text_path: str = None

    def _encapsulate_utterance(self, utterance):
        utterance = utterance.lower()
        if self.tokenizer.bos_token_id != utterance[0]:
            return self.tokenizer.bos_token + utterance + self.tokenizer.eos_token
        else:
            return utterance

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = self.feature_extractor(
            [audio_object_stripper(feature[self.audio_path]) for feature in features],
            padding=True,
            sampling_rate=self.sampling_rate)
        labels = self.tokenizer.batch_encode_plus(
            [self._encapsulate_utterance(feature[self.text_path]) for feature in features],
            return_attention_mask=True,
            padding='longest',
            return_tensors='pt')

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if "input_features" in batch:
            batch["input_values"] = batch["input_features"]
            del batch["input_features"]
        return batch


@dataclass
class Seq2SeqDataCollatorWithPaddingAndConvId(Seq2SeqDataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = self.feature_extractor([feature["input_values"] for feature in features], padding=True,
                                                sampling_rate=self.sampling_rate)
        labels = self.tokenizer.batch_encode_plus(
            [self._encapsulate_utterance(feature['labels']) for feature in features], return_attention_mask=True,
            padding='longest', return_tensors='pt')

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch['conv_ids'] = [feature['recording'] for feature in features]

        if "input_features" in batch:
            batch["input_values"] = batch["input_features"]
            del batch["input_features"]
        return batch


def filter_sequences_in_range(batch: List[int], max_input_len: int, min_input_len: int):
    arr = np.array(batch)
    return (arr <= max_input_len) & (arr >= min_input_len)


def filter_out_sequence_from_dataset(df: Dataset, max_input_len: float = 5.0,
                                     min_input_len: float = 0.1, length_column="input_len") -> Dataset:
    """Filters out sequences form dataset which are longer than provided threshold"""
    lengths = np.array(df[length_column])
    indexes_ok = np.argwhere(np.logical_and(lengths <= max_input_len, lengths >= min_input_len))
    df = df.select(indexes_ok.flatten())
    return df


def group_params(model, weight_decay, learning_rate, cross_attention_scaling_factor):
    """Add different weight decay and lr rate for specific layers"""
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return [
        {
            "params": [
                p for n, p in model.named_parameters() if
                (n in decay_parameters and "cross" not in n)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if
                (n not in decay_parameters and "cross" not in n)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if
                (n in decay_parameters and "cross" in n)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate * cross_attention_scaling_factor
        },
        {
            "params": [
                p for n, p in model.named_parameters() if
                (n not in decay_parameters and "cross" in n)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate * cross_attention_scaling_factor
        },
    ]
