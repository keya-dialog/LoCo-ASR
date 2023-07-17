import os
import pickle
from dataclasses import dataclass, field, make_dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from jiwer import cer, compute_measures
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, BatchFeature, \
    EarlyStoppingCallback, HfArgumentParser, PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel, TrainerCallback, TrainerControl, \
    TrainerState, TrainingArguments, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import shift_tokens_right
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import logging

PAD_TOKEN = '|'
BOS_TOKEN = '<'
EOS_TOKEN = '>'
SAMPLING_RATE = 16000


@dataclass
class ModelArguments:
    base_encoder_model: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    base_decoder_model: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    enc_adapters: bool = field(
        default=False, metadata={"help": "Add adapters to the encoder."}
    )
    dec_adapters: bool = field(
        default=False, metadata={"help": "Add adapters to the decoder."}
    )


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    early_stopping_patience: int = field(
        default=1, metadata={"help": "Patience for early stopping."}
    )
    decoder_cold_start: bool = field(
        default=False, metadata={"help": "Whenever to reinitialize decoder weights"}
    )
    enc_layers_to_freeze: int = field(
        default=0, metadata={"help": "Encoder layers to freeze"}
    )
    dec_layers_to_freeze: int = field(
        default=0, metadata={"help": "Decoder layers to freeze"}
    )
    steps_to_freeze_enc: int = field(
        default=0, metadata={"help": "Steps to freeze encoder"}
    )
    steps_to_freeze_dec: int = field(
        default=0, metadata={"help": "Steps to freeze decoder"}
    )
    custom_optimizer: bool = field(
        default=False, metadata={"help": "Custom optimizer for decoder"}
    )
    cross_attention_scaling_factor: float = field(
        default=1, metadata={"help": "Custom scaling factor for cross attention weights"}
    )
    n_gpus: int = field(
        default=1, metadata={"help": "Number of gpus to be used"}
    )
    ctc_weight: float = field(
        default=0, metadata={"help": "Weight of CTC loss."}
    )


@dataclass
class GenerationArguments:
    num_beams: int = field(
        default=1, metadata={"help": "Num beams for decoding."}
    )
    max_len: int = field(
        default=200, metadata={"help": "Max number of generated tokens."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    train_split: str = field(
        default="train", metadata={"help": "Training split to be used."}
    )
    validation_split: str = field(
        default="validation", metadata={"help": "Validation split to be used."}
    )
    test_split: str = field(
        default="test", metadata={"help": "Test split to be used."}
    )
    val_indexes_to_use: str = field(
        default="", metadata={"help": "Part of the validation split to be used."}
    )


@dataclass
class Seq2SeqDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature extractor used for processing the data.
        decoder_tokenizer (:class:`~transformers.PreTrainedTokenizerFast`)
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
    decoder_tokenizer: PreTrainedTokenizerFast
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def _encapsulate_utterance(self, utterance):
        return self.decoder_tokenizer.bos_token + utterance + self.decoder_tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = self.feature_extractor([feature["input_values"] for feature in features], padding=True,
                                                sampling_rate=SAMPLING_RATE)
        labels = self.decoder_tokenizer.batch_encode_plus(
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

        return batch


class FrozenLayersManager(TrainerCallback):
    def __init__(self, enc_layers_to_freeze, dec_layers_to_freeze, steps_to_freeze_enc, steps_to_freeze_dec):
        super().__init__()
        self.enc_layers_to_freeze = enc_layers_to_freeze
        self.dec_layers_to_freeze = dec_layers_to_freeze
        self.steps_to_freeze_enc = steps_to_freeze_enc
        self.steps_to_freeze_dec = steps_to_freeze_dec

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
                    param.requires_grad = True

        if self.dec_layers_to_freeze > 0:
            for name, param in curr_model.decoder.named_parameters():
                if name.startswith("transformer.h."):
                    if 'cross' in name or 'adapter' in name:
                        param.requires_grad = True
                    else:
                        layer = int(name.split('.')[2])
                        if layer < self.dec_layers_to_freeze:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                else:
                    param.requires_grad = True

        curr_model.freeze_feature_encoder()
        curr_model.decoder.lm_head.weight.requires_grad = True
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


def compute_metrics(pred):
    pred_ids = pred.predictions

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = decoder_tokenizer.pad_token_id

    pred_str = decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else '-' for label in
                 decoder_tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    metrics = compute_measures(label_str, pred_str)
    del metrics['ops']
    del metrics['truth']
    del metrics['hypothesis']
    df = pd.DataFrame({'label': label_str, 'prediction': pred_str})
    df.to_csv(os.path.join(trainer.args.output_dir, f'predictions_step{trainer.state.global_step}.csv'))

    return {"cer": cer(label_str, pred_str), **metrics}


def filter_out_sequence_from_dataset(df: Dataset, max_input_len: float = 5.0,
                                     min_input_len: float = 0.1) -> Dataset:
    """Filters out sequences form dataset which are longer than provided threshold"""
    lengths = np.array(df['input_len'])
    indexes_ok = np.argwhere(np.logical_and(lengths <= max_input_len, lengths >= min_input_len))
    df = df.select(indexes_ok.flatten())
    return df


def group_params():
    """Add different weight decay and lr rate for specific layers"""
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return [
        {
            "params": [
                p for n, p in model.named_parameters() if
                (n in decay_parameters and "cross" not in n)
            ],
            "weight_decay": training_args.weight_decay,
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
            "weight_decay": training_args.weight_decay,
            "lr": training_args.learning_rate * model_args.cross_attention_scaling_factor
        },
        {
            "params": [
                p for n, p in model.named_parameters() if
                (n not in decay_parameters and "cross" in n)
            ],
            "weight_decay": 0.0,
            "lr": training_args.learning_rate * model_args.cross_attention_scaling_factor
        },
    ]


@dataclass
class Seq2SeqLMOutputLosses(Seq2SeqLMOutput):
    enc_loss: Optional[torch.FloatTensor] = None
    dec_loss: Optional[torch.FloatTensor] = None


class PrinterCallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        state.__class__ = make_dataclass('state_derived',
                                         [('additional_logs', List[List[float]], field(default_factory=list))],
                                         bases=(TrainerState,))
        state.additional_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(state, 'additional_logs'):
            enc_loss, dec_loss = torch.tensor(state.additional_logs).mean(axis=0)
            if state.is_local_process_zero:
                logs['enc_loss'] = float(enc_loss)
                logs['dec_loss'] = float(dec_loss)
            state.additional_logs = []


class CustomTrainer(Seq2SeqTrainer):
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


class JointCTCAttentionEncoderDecoder(SpeechEncoderDecoderModel):
    """Custom model for CTC+Attention loss based on the ESPNet architecture"""

    config_class = SpeechEncoderDecoderConfig
    base_model_prefix = "speech_encoder_decoder"
    main_input_name = "inputs"
    supports_gradient_checkpointing = True

    def __init__(self,
                 config: Optional[PretrainedConfig] = None,
                 encoder: Optional[PreTrainedModel] = None,
                 decoder: Optional[PreTrainedModel] = None):
        super().__init__(config, encoder, decoder)
        self.enc_loss_weight = config.ctc_weight
        self.dec_loss_weight = 1 - config.ctc_weight

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            spec_augment_cfg=None, reverb_cfg=None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = Wav2Vec2ForCTC.from_pretrained(encoder_pretrained_model_name_or_path,
                                                     *model_args,
                                                     **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
            self,
            inputs: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            input_values: Optional[torch.FloatTensor] = None,
            input_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputLosses]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                labels=labels,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs['hidden_states'][-1] if return_dict else encoder_outputs[1][-1]

        # optionally project encoder_hidden_states
        if (
                self.encoder_output_dim != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # compute correct encoder attention mask
        if attention_mask is not None:
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
        else:
            encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = enc_loss = dec_loss = None
        if labels is not None:
            dec_logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            dec_loss = loss_fct(dec_logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            enc_loss = encoder_outputs.loss if return_dict else encoder_outputs[0]
            loss = self.enc_loss_weight * enc_loss + self.dec_loss_weight * dec_loss

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutputLosses(
            loss=loss,
            enc_loss=enc_loss,
            dec_loss=dec_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


if __name__ == '__main__':
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Load dataset
    dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)
    for split in [data_args.train_split, data_args.validation_split, data_args.test_split]:
        dataset[split] = filter_out_sequence_from_dataset(dataset[split],
                                                          max_input_len=data_args.max_duration_in_seconds,
                                                          min_input_len=data_args.min_duration_in_seconds)

    if data_args.val_indexes_to_use:
        indexes = set(open(data_args.val_indexes_to_use).read().splitlines())
        indexes_to_select = [index for index, utt_id in enumerate(dataset[data_args.validation_split]['uttid']) if
                             utt_id in indexes]
        dataset[data_args.validation_split] = dataset[data_args.validation_split].select(indexes_to_select)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    decoder_tokenizer.bos_token_id = decoder_tokenizer.vocab[BOS_TOKEN]
    decoder_tokenizer.eos_token_id = decoder_tokenizer.vocab[EOS_TOKEN]
    decoder_tokenizer.pad_token_id = decoder_tokenizer.vocab[PAD_TOKEN]

    decoder_tokenizer.add_special_tokens({
        "additional_special_tokens": [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    })

    # 3. Initialize seq2seq model
    model = JointCTCAttentionEncoderDecoder.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=model_args.base_encoder_model,
        decoder_pretrained_model_name_or_path=model_args.base_decoder_model,
        bos_token_id=decoder_tokenizer.bos_token_id,
        eos_token_id=decoder_tokenizer.eos_token_id,
        pad_token_id=decoder_tokenizer.pad_token_id,
        encoder_feat_proj_dropout=0.0,
        encoder_layerdrop=0.0,
        min_length=0,
        no_repeat_ngram_size=0,
        early_stopping=True,
        length_penalty=1,
        max_length=gen_args.max_len,
        num_beams=gen_args.num_beams,
        encoder_add_adapter=model_args.enc_adapters,
        ctc_weight=training_args.ctc_weight,
        encoder_ctc_loss_reduction="mean",
        encoder_pad_token_id=decoder_tokenizer.pad_token_id,
        encoder_vocab_size=len(decoder_tokenizer),
    )

    if training_args.decoder_cold_start:
        logger.info('Reinitializing decoder weights')
        model.decoder.apply(model.decoder._init_weights)

    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id

    if model_args.dec_adapters:
        model.decoder.add_adapter("gpt2_fisher", set_active=True)
        model.decoder.train_adapter("gpt2_fisher")

    # 4. Init trainer
    layer_training_manager = FrozenLayersManager(training_args.enc_layers_to_freeze, training_args.dec_layers_to_freeze,
                                                 training_args.steps_to_freeze_enc, training_args.steps_to_freeze_dec)
    early_stopping = EarlyStoppingCallback(training_args.early_stopping_patience)
    printing_callback = PrinterCallback()
    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   decoder_tokenizer=decoder_tokenizer,
                                                   padding=True)
    optimizer = None
    if training_args.custom_optimizer:
        optimizer = AdamW(group_params(), lr=training_args.learning_rate)

    trainer = CustomTrainer(
        args=training_args,
        model=model,
        callbacks=[layer_training_manager, early_stopping, printing_callback],
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.validation_split],
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    # 5. Train
    trainer.train()

    decoder_tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'tokenizer'))
    feature_extractor.save_pretrained(os.path.join(training_args.output_dir, 'feature_extractor'))

    # 6. Eval on dev
    trainer.args.predict_with_generate = True
    model.config.output_hidden_states = True

    predictions = trainer.predict(dataset[data_args.validation_split])
    logger.info(compute_metrics(predictions))
    with open(os.path.join(training_args.output_dir, 'val_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)

    # # 6. Eval on test
    predictions = trainer.predict(dataset[data_args.test_split])
    logger.info(compute_metrics(predictions))
    with open(os.path.join(training_args.output_dir, 'test_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)
