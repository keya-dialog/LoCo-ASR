from dataclasses import dataclass, field
from typing import List, Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    base_encoder_model: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    base_decoder_model: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    from_pretrained: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    pad_token: Optional[str] = field(
        default='|', metadata={"help": "PAD token"}
    )
    bos_token: Optional[str] = field(
        default='<', metadata={"help": "BOS token"}
    )
    eos_token: Optional[str] = field(
        default='>', metadata={"help": "EOS token"}
    )
    enc_adapters: Optional[bool] = field(
        default=False, metadata={"help": "Add adapters to the encoder."}
    )
    dec_adapters: Optional[bool] = field(
        default=False, metadata={"help": "Add adapters to the decoder."}
    )
    sampling_rate: Optional[int] = field(
        default=16_000, metadata={"help": "Sampling rate for the model"}
    )


@dataclass
class GeneralTrainingArguments(Seq2SeqTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=1, metadata={"help": "Patience for early stopping."}
    )
    decoder_cold_start: Optional[bool] = field(
        default=False, metadata={"help": "Whenever to reinitialize decoder weights"}
    )
    enc_layers_to_freeze: Optional[int] = field(
        default=0, metadata={"help": "Encoder layers to freeze"}
    )
    dec_layers_to_freeze: Optional[int] = field(
        default=0, metadata={"help": "Decoder layers to freeze"}
    )
    steps_to_freeze_enc: Optional[int] = field(
        default=0, metadata={"help": "Steps to freeze encoder"}
    )
    steps_to_freeze_dec: Optional[int] = field(
        default=0, metadata={"help": "Steps to freeze decoder"}
    )
    custom_optimizer: Optional[bool] = field(
        default=False, metadata={"help": "Custom optimizer for decoder"}
    )
    cross_attention_scaling_factor: Optional[float] = field(
        default=1, metadata={"help": "Custom scaling factor for cross attention weights"}
    )
    ctc_weight: Optional[float] = field(
        default=0, metadata={"help": "Weight of CTC loss."}
    )
    restart_from: Optional[str] = field(
        default="", metadata={"help": "Path to checkpoint used to restart the training."}
    )
    lsm_factor: Optional[float] = field(
        default=0, metadata={"help": "Label smoothing coefficient for CE loss."}
    )
    shared_lm_head: Optional[bool] = field(
        default=False, metadata={"help": "Whether to share LM head params."}
    )
    use_fbanks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use fbanks instead of raw audio signal."}
    )
    freeze_cross_attention: bool = field(
        default=False, metadata={"help": "Whether to freeze cross attentions"}
    )


@dataclass
class GenerationArguments:
    num_beams: Optional[int] = field(
        default=1, metadata={"help": "Num beams for decoding."}
    )
    max_len: Optional[int] = field(
        default=200, metadata={"help": "Max number of generated tokens."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "The config of the dataset to use (via the datasets library)."}
    )
    audio_column_name: Optional[str] = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: Optional[float] = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    train_split: Optional[str] = field(
        default="train", metadata={"help": "Training split to be used."}
    )
    validation_split: Optional[str] = field(
        default="validation", metadata={"help": "Validation split to be used."}
    )
    test_split: Optional[str] = field(
        default="test", metadata={"help": "Test split to be used."}
    )
    val_indexes_to_use: Optional[str] = field(
        default="", metadata={"help": "Part of the validation split to be used."}
    )
    apply_augmentations: Optional[bool] = field(
        default=False, metadata={"help": "Whether to apply on-the fly augmentations."}
    )


@dataclass
class ModelArgumentsContext(ModelArguments):
    enc_memory_cells_location: List[int] = field(
        default=None, metadata={"help": "Positions where to place memory cells in encoder"}
    )
    dec_memory_cells_location: List[int] = field(
        default=None, metadata={"help": "Positions where to place memory cells in decoder"}
    )
    enc_memory_dim: int = field(
        default=None, metadata={"help": "Size of memory on encoder size"}
    )
    dec_memory_dim: int = field(
        default=None, metadata={"help": "Size of memory on decoder size"}
    )


@dataclass
class GeneralTrainingArgumentsContext(GeneralTrainingArguments):
    freeze_others: bool = field(
        default=False, metadata={"help": "Whether to freeze rest of the model"}
    )
    conv_ids_column_name: str = field(
        default=None, metadata={"help": "Conv ids column."}
    )
    turn_index_column_name: str = field(
        default=None, metadata={"help": "Turn index column."}
    )


@dataclass
class TokenizerTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "The config of the dataset to use (via the datasets library)."}
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    train_split: Optional[str] = field(
        default="train", metadata={"help": "Training split to be used."}
    )
    vocab_size: Optional[int] = field(
        default=5_000, metadata={"help": "Vocab size."}
    )
