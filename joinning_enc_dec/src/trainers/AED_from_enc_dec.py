import os
import pickle

import numpy as np
from audiomentations import AddGaussianNoise, Compose, PitchShift, Shift, TanhDistortion, TimeMask, TimeStretch
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer, \
    EarlyStoppingCallback, HfArgumentParser
from transformers.utils import logging

from per_utterance.ctc_encoder_plus_autoregressive_decoder import JointCTCAttentionEncoderDecoder, \
    JointCTCAttentionEncoderDecoderConfig
from trainers.training_arguments import DataTrainingArguments, GeneralTrainingArguments, GenerationArguments, \
    ModelArguments
from utils import AdditionalLossPrinterCallback, AdditionalLossTrackerTrainer, FrozenLayersManager, \
    Seq2SeqDataCollatorWithPadding, audio_object_stripper, compute_metrics, filter_out_sequence_from_dataset, \
    group_params

AutoConfig.register("joint_aed_ctc_speech-encoder-decoder", JointCTCAttentionEncoderDecoderConfig)
AutoModelForSpeechSeq2Seq.register(JointCTCAttentionEncoderDecoderConfig, JointCTCAttentionEncoderDecoder)

if __name__ == '__main__':
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Load dataset
    if data_args.dataset_config is not None:
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config, keep_in_memory=False)
    else:
        dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)

    if training_args.length_column_name not in dataset[data_args.train_split].column_names:
        len_column = training_args.length_column_name
        audio_column = data_args.audio_column_name
        sampling_rate = model_args.sampling_rate


        def preprocess(example):
            example[len_column] = len(
                audio_object_stripper(example[audio_column])) / sampling_rate
            return example


        dataset = dataset.map(preprocess,
                              num_proc=data_args.preprocessing_num_workers,
                              writer_batch_size=data_args.writer_batch_size)

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
        dataset[data_args.train_split].set_transform(lambda batch: {data_args.audio_column_name: [
            augmenter(np.array(audio_object_stripper(audio), dtype=np.float32), sample_rate=model_args.sampling_rate)
            for audio in batch[data_args.audio_column_name]]}, columns=[data_args.audio_column_name],
                                                     output_all_columns=True)

    if data_args.val_indexes_to_use:
        indexes = set(open(data_args.val_indexes_to_use).read().splitlines())
        indexes_to_select = [index for index, utt_id in enumerate(dataset[data_args.validation_split]['uttid']) if
                             utt_id in indexes]
        dataset[data_args.validation_split] = dataset[data_args.validation_split].select(indexes_to_select)

    if training_args.preprocess_dataset_only:
        exit(0)

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
        "shared_lm_head": training_args.shared_lm_head,
        "use_fbanks": training_args.use_fbanks,
        "num_mel_bins": feature_extractor.num_mel_bins if hasattr(feature_extractor, "num_mel_bins") else None
    }

    # 3. Initialize seq2seq model
    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_args.from_pretrained,
            config=config)
    else:
        model = JointCTCAttentionEncoderDecoder.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=model_args.base_encoder_model,
            decoder_pretrained_model_name_or_path=model_args.base_decoder_model,
            **base_model_config
        )

    if training_args.decoder_cold_start:
        logger.info('Reinitializing decoder weights')
        model.decoder.apply(model.decoder._init_weights)

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Ensure encoder return hidden states and predictions are generated
    model.config.output_hidden_states = True

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
                                                   padding=True,
                                                   sampling_rate=model_args.sampling_rate,
                                                   audio_path=data_args.audio_column_name,
                                                   text_path=data_args.text_column_name
                                                   )
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
