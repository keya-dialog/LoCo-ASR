import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_from_disk
from torch.optim import AdamW
from transformers import AutoFeatureExtractor, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, \
    Seq2SeqTrainingArguments
from transformers.utils import logging

from per_utterance.models import JointCTCAttentionEncoderDecoderSharedHeads
from utils import AdditionalLossPrinterCallback, AdditionalLossTrackerTrainer, FrozenLayersManager, \
    Seq2SeqDataCollatorWithPadding, compute_metrics, filter_out_sequence_from_dataset, group_params


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
    pad_token: Optional[str] = field(
        default='|', metadata={"help": "PAD token"}
    )
    bos_token: Optional[str] = field(
        default='<', metadata={"help": "BOS token"}
    )
    eos_token: Optional[str] = field(
        default='>', metadata={"help": "EOS token"}
    )
    blank_token: Optional[str] = field(
        default='[BLANK]', metadata={"help": "BLANK token"}
    )
    enc_adapters: bool = field(
        default=False, metadata={"help": "Add adapters to the encoder."}
    )
    dec_adapters: bool = field(
        default=False, metadata={"help": "Add adapters to the decoder."}
    )
    sampling_rate: int = field(
        default=16_000, metadata={"help": "Sampling rate for the model"}
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
    decoder_tokenizer.bos_token_id = decoder_tokenizer.vocab[model_args.bos_token]
    decoder_tokenizer.eos_token_id = decoder_tokenizer.vocab[model_args.eos_token]
    decoder_tokenizer.pad_token_id = decoder_tokenizer.vocab[model_args.pad_token]

    decoder_tokenizer.add_special_tokens({
        "additional_special_tokens": [model_args.pad_token, model_args.bos_token, model_args.eos_token]
    })

    # 3. Initialize seq2seq model
    model = JointCTCAttentionEncoderDecoderSharedHeads.from_encoder_decoder_pretrained(
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
    printing_callback = AdditionalLossPrinterCallback()
    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   tokenizer=decoder_tokenizer,
                                                   padding=True, sampling_rate=model_args.sampling_rate)
    optimizer = None
    if training_args.custom_optimizer:
        optimizer = AdamW(group_params(model, training_args.weight_decay, training_args.learning_rate,
                                       model_args.cross_attention_scaling_factor), lr=training_args.learning_rate)

    trainer = AdditionalLossTrackerTrainer(
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
    logger.info(compute_metrics(decoder_tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'val_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)

    # # 6. Eval on test
    predictions = trainer.predict(dataset[data_args.test_split])
    logger.info(compute_metrics(decoder_tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'test_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)
