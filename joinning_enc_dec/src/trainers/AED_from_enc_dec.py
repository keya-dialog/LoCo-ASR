import os
import pickle

from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer, \
    EarlyStoppingCallback, GenerationConfig, HfArgumentParser
from transformers.utils import logging

from per_utterance.ctc_encoder_plus_autoregressive_decoder import JointCTCAttentionEncoderDecoder, \
    JointCTCAttentionEncoderDecoderConfig
from trainers.training_arguments import DataTrainingArguments, GeneralTrainingArguments, GenerationArguments, \
    ModelArguments
from utils import AdditionalLossPrinterCallback, AdditionalLossTrackerTrainer, FrozenLayersManager, \
    Seq2SeqDataCollatorWithPadding, activate_joint_decoding, compute_metrics, fetch_AED_config, prepare_dataset

AutoConfig.register("joint_aed_ctc_speech-encoder-decoder", JointCTCAttentionEncoderDecoderConfig)
AutoModelForSpeechSeq2Seq.register(JointCTCAttentionEncoderDecoderConfig, JointCTCAttentionEncoderDecoder)

if __name__ == '__main__':
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # 2. Load dataset
    if data_args.dataset_config is not None:
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config, keep_in_memory=False,
                               num_proc=data_args.preprocessing_num_workers)
    else:
        dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)

    len_column = training_args.length_column_name
    text_column = data_args.text_column_name
    audio_column = data_args.audio_column_name
    sampling_rate = model_args.sampling_rate
    max_input_len = data_args.max_duration_in_seconds,
    min_input_len = data_args.min_duration_in_seconds

    # 3. Preprocess dataset
    logger.info("Preprocessing dataset...")
    dataset = prepare_dataset(
        dataset=dataset,
        dataset_name=data_args.dataset_name,
        length_column_name=len_column,
        text_column_name=text_column,
        audio_column_name=audio_column,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        writer_batch_size=data_args.writer_batch_size,
        train_split=data_args.train_split,
        validation_split=data_args.validation_split,
        fix_apostrophes=data_args.fix_apostrophes,
        remove_train_unks=data_args.remove_train_unks,
        apply_augmentations=data_args.apply_augmentations,
        unk_token=tokenizer.unk_token,
        sampling_rate=sampling_rate,
        max_input_len=max_input_len,
        min_input_len=min_input_len,
        validation_slice=data_args.validation_slice
    )

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        exit(0)

    base_model_config = {
        "bos_token_id": tokenizer.bos_token_id,
        "decoder_bos_token_id": tokenizer.bos_token_id,
        "encoder_bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "decoder_eos_token_id": tokenizer.eos_token_id,
        "encoder_eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "encoder_feat_proj_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "early_stopping": False,
        "length_penalty": gen_args.length_penalty,
        "max_length": gen_args.max_len,
        "num_beams": gen_args.num_beams,
        "encoder_add_adapter": model_args.enc_adapters,
        "ctc_weight": training_args.ctc_weight,
        "encoder_ctc_loss_reduction": "mean",
        "encoder_pad_token_id": tokenizer.pad_token_id,
        "encoder_vocab_size": len(tokenizer),
        "decoder_vocab_size": len(tokenizer),
        "lsm_factor": training_args.lsm_factor,
        "shared_lm_head": training_args.shared_lm_head,
        "use_fbanks": training_args.use_fbanks,
        "num_mel_bins": feature_extractor.num_mel_bins if hasattr(feature_extractor, "num_mel_bins") else None,
        "output_hidden_states": True,
        "decoder_start_token_id": tokenizer.bos_token_id,
    }

    # 4. Initialize seq2seq model
    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_args.from_pretrained,
            config=config)
    elif model_args.from_encoder_decoder_config:
        config = fetch_AED_config(model_args.base_encoder_model, model_args.base_decoder_model, base_model_config)
        model = JointCTCAttentionEncoderDecoder(config=config)
    else:
        model = JointCTCAttentionEncoderDecoder.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=model_args.base_encoder_model,
            decoder_pretrained_model_name_or_path=model_args.base_decoder_model,
            **base_model_config
        )

    if gen_args.decoding_ctc_weight is not None:
        activate_joint_decoding(model, gen_args.decoding_ctc_weight, gen_args.ctc_margin, len(tokenizer),
                                base_model_config['eos_token_id'])

    gen_config = GenerationConfig(bos_token_id=base_model_config['bos_token_id'],
                                  pad_token_id=base_model_config['pad_token_id'],
                                  decoder_start_token_id=base_model_config['bos_token_id'],
                                  length_penalty=base_model_config['length_penalty'],
                                  early_stopping=base_model_config['early_stopping'],
                                  eos_token_id=base_model_config['eos_token_id'],
                                  max_length=base_model_config['max_length'],
                                  output_hidden_states=base_model_config['output_hidden_states'],
                                  num_beams=base_model_config['num_beams'])
    logger.info(f'Model updated gen config:\n {str(gen_config)}')
    model.generation_config = gen_config

    if training_args.decoder_cold_start:
        logger.info('Reinitializing decoder weights')
        model.decoder.apply(model.decoder._init_weights)

    if model_args.dec_adapters:
        model.decoder.add_adapter("dec_adapters", set_active=True)
        model.decoder.train_adapter("dec_adapters")

    # 5. Init trainer
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

    trainer = AdditionalLossTrackerTrainer(
        args=training_args,
        model=model,
        callbacks=[layer_training_manager, early_stopping, printing_callback],
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.validation_split],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred, gen_args.wandb_predictions_to_save),
    )

    # 6. Train
    trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'tokenizer'))
    feature_extractor.save_pretrained(os.path.join(training_args.output_dir, 'feature_extractor'))

    predictions = trainer.predict(dataset[data_args.validation_split])
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'val_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)

    # 7. Eval on test
    predictions = trainer.predict(dataset[data_args.test_split])
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'test_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)
