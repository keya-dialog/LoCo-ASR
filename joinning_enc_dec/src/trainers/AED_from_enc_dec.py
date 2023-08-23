import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from audiomentations import AddGaussianNoise, Compose, PitchShift, Shift, TanhDistortion, TimeMask, TimeStretch
from datasets import load_from_disk
from torch.optim import AdamW
from transformers import AutoFeatureExtractor, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, \
    Seq2SeqTrainingArguments
from transformers.utils import logging

from per_utterance.models import JointCTCAttentionEncoderDecoder, MelFeatureExtractor
from utils import AdditionalLossPrinterCallback, AdditionalLossTrackerTrainer, FrozenLayersManager, \
    Seq2SeqDataCollatorWithPadding, compute_metrics, filter_out_sequence_from_dataset, group_params


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
class CustomTrainingArguments(Seq2SeqTrainingArguments):
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

    if data_args.apply_augmentations:
        augmenter = Compose([
            TimeMask(max_band_part=0.05, p=0.05),
            TimeMask(max_band_part=0.05, p=0.05),
            TimeMask(max_band_part=0.05, p=0.05),
            TimeMask(max_band_part=0.05, p=0.05),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.2),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
            TanhDistortion(min_distortion=0, max_distortion=0.2, p=0.2)
        ])
        dataset['train'].set_transform(
            lambda batch: {
                data_args.audio_column_name: [augmenter(np.array(audio), sample_rate=model_args.sampling_rate) for audio
                                              in
                                              batch[data_args.audio_column_name]]},
            columns=[data_args.audio_column_name], output_all_columns=True)

    if data_args.val_indexes_to_use:
        indexes = set(open(data_args.val_indexes_to_use).read().splitlines())
        indexes_to_select = [index for index, utt_id in enumerate(dataset[data_args.validation_split]['uttid']) if
                             utt_id in indexes]
        dataset[data_args.validation_split] = dataset[data_args.validation_split].select(indexes_to_select)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    tokenizer.bos_token_id = tokenizer.vocab[model_args.bos_token]
    tokenizer.eos_token_id = tokenizer.vocab[model_args.eos_token]
    tokenizer.pad_token_id = tokenizer.vocab[model_args.pad_token]

    tokenizer.add_special_tokens({
        "additional_special_tokens": [model_args.pad_token, model_args.bos_token, model_args.eos_token]
    })

    base_model_config = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "encoder_feat_proj_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "early_stopping": True,
        "length_penalty": 1,
        "max_length": gen_args.max_len,
        "num_beams": gen_args.num_beams,
        "encoder_add_adapter": model_args.enc_adapters,
        "ctc_weight": training_args.ctc_weight,
        "encoder_ctc_loss_reduction": "mean",
        "encoder_pad_token_id": tokenizer.pad_token_id,
        "encoder_vocab_size": len(tokenizer),
        "lsm_factor": training_args.lsm_factor,
        "shared_lm_head": training_args.shared_lm_head
    }

    # 3. Initialize seq2seq model
    if model_args.from_pretrained:
        model = JointCTCAttentionEncoderDecoder.from_pretrained(
            model_args.from_pretrained,
            **base_model_config)
    else:
        model = JointCTCAttentionEncoderDecoder.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=model_args.base_encoder_model,
            decoder_pretrained_model_name_or_path=model_args.base_decoder_model,
            **base_model_config
        )

    if training_args.use_fbanks:
        model.encoder.base_model.config.conv_kernel = [3, 3]
        model.encoder.config.conv_kernel = [3, 3]
        model.encoder.base_model.config.conv_stride = [2, 2]
        model.encoder.config.conv_stride = [2, 2]
        model.encoder.config.num_mel_bins = feature_extractor.num_mel_bins
        model.encoder.config.max_source_positions = 1024
        model.encoder.base_model.feature_extractor = MelFeatureExtractor(model.encoder.config)

    if training_args.decoder_cold_start:
        logger.info('Reinitializing decoder weights')
        model.decoder.apply(model.decoder._init_weights)

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if model_args.dec_adapters:
        model.decoder.add_adapter("dec_adapters", set_active=True)
        model.decoder.train_adapter("dec_adapters")

    # 4. Init trainer
    layer_training_manager = FrozenLayersManager(training_args.enc_layers_to_freeze, training_args.dec_layers_to_freeze,
                                                 training_args.steps_to_freeze_enc, training_args.steps_to_freeze_dec)
    early_stopping = EarlyStoppingCallback(training_args.early_stopping_patience)
    printing_callback = AdditionalLossPrinterCallback()
    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   tokenizer=tokenizer,
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
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred),
        optimizers=(optimizer, None)
    )

    # Ensure encoder return hidden states and predictions are generated
    trainer.args.predict_with_generate = True
    model.config.output_hidden_states = True

    # 5. Train
    trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'tokenizer'))
    feature_extractor.save_pretrained(os.path.join(training_args.output_dir, 'feature_extractor'))

    predictions = trainer.predict(dataset[data_args.validation_split])
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'val_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)

    # 6. Eval on test
    predictions = trainer.predict(dataset[data_args.test_split])
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'test_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)
