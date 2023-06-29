import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, Dataset
from jiwer import cer, compute_measures
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    SpeechEncoderDecoderModel, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, BatchFeature, \
    logging as tfs_logging, Wav2Vec2FeatureExtractor, HfArgumentParser, AutoFeatureExtractor, EarlyStoppingCallback
from torch.optim import AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

PAD_TOKEN = '¬'
BOS_TOKEN = '@'
EOS_TOKEN = '<|endoftext|>'
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
        default=False, metadata={"help": "Patience for early stopping."}
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

    train_subset: float = field(
        default=1.0, metadata={"help": "Part of the training split to be used."}
    )
    train_split: str = field(
        default="train", metadata={"help": "Training split to be used."}
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

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        curr_model: SpeechEncoderDecoderModel = kwargs['model']
        curr_model.train()
        curr_model.encoder.train()
        curr_model.decoder.train()
        curr_model.freeze_feature_encoder()
        if self.enc_layers_to_freeze > 0:
            for name, param in curr_model.encoder.named_parameters():
                if name.startswith("encoder.layers"):
                    layer = int(name.split('.')[2])
                    if layer < self.enc_layers_to_freeze:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

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

        curr_model.decoder.lm_head.weight.requires_grad = True
        logging.debug([n for n, p in curr_model.named_parameters() if p.requires_grad])
        logging.info(
            f'Model info: total parameters - {curr_model.num_parameters()}, trainable parameters - '
            f'{sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, curr_model.parameters())])}')

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        curr_model: SpeechEncoderDecoderModel = kwargs['model']
        if state.global_step == self.steps_to_freeze_enc:
            logging.info(f'Step: {state.global_step} encoder unfrozen.')
            self.reactivate_params(curr_model, curr_model.encoder.parameters())

        if state.global_step == self.steps_to_freeze_dec:
            logging.info(f'Step: {state.global_step} decoder unfrozen.')
            self.reactivate_params(curr_model, curr_model.decoder.parameters())

    @staticmethod
    def reactivate_params(curr_model, params_to_activate):
        for param in params_to_activate:
            param.requires_grad = True
        logging.debug([n for n, p in curr_model.named_parameters() if p.requires_grad])
        logging.info(
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tfs_logging.set_verbosity_debug()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Load dataset
    dataset = load_from_disk(data_args.dataset_name)
    dataset[data_args.train_split] = filter_out_sequence_from_dataset(dataset[data_args.train_split],
                                                                      max_input_len=data_args.max_duration_in_seconds,
                                                                      min_input_len=data_args.min_duration_in_seconds)
    if data_args.train_subset:
        dataset[data_args.train_split] = dataset[data_args.train_split].select(
            range(int(data_args.train_subset * len(dataset[data_args.train_split]))))

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
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
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
    )

    if training_args.decoder_cold_start:
        logging.info('Reinitializing decoder weights')
        model.decoder.apply(model.decoder._init_weights)

    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id

    if model_args.dec_adapters:
        model.decoder.add_adapter("gpt2_fisher")

    # 4. Init trainer
    layer_training_manager = FrozenLayersManager(training_args.enc_layers_to_freeze, training_args.dec_layers_to_freeze,
                                                 training_args.steps_to_freeze_enc, training_args.steps_to_freeze_dec)
    early_stooping = EarlyStoppingCallback(training_args.early_stopping_patience)
    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   decoder_tokenizer=decoder_tokenizer,
                                                   padding=True)
    optimizer = None
    if training_args.custom_optimizer:
        optimizer = AdamW(group_params(), lr=training_args.learning_rate)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        callbacks=[layer_training_manager, early_stooping],
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    # 5. Train
    trainer.train()

    # 6. Eval on test
    metrics = trainer.evaluate(dataset['test'])
    logging.info(str(metrics))
