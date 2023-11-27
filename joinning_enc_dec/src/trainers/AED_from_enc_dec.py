import math

import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, \
    AutoTokenizer, EarlyStoppingCallback, GenerationConfig, HfArgumentParser, Seq2SeqTrainer
from transformers.utils import logging

from per_utterance.ctc_encoder_plus_autoregressive_decoder import JointCTCAttentionEncoderDecoder, \
    JointCTCAttentionEncoderDecoderConfig
from trainers.training_arguments import DataTrainingArguments, GeneralTrainingArguments, GenerationArguments, \
    ModelArguments
from utils import AdditionalLossPrinterCallback, AdditionalLossTrackerTrainer, AugmentationManagerCallback, \
    Seq2SeqDataCollatorWithPadding, activate_joint_decoding, average_checkpoints, compute_metrics, fetch_AED_config, \
    postprocess_beam_outputs, prepare_dataset, save_nbests, save_predictions

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
        fix_apostrophes=data_args.fix_apostrophes,
        remove_train_unks=data_args.remove_train_unks,
        apply_augmentations=data_args.apply_augmentations,
        unk_token=tokenizer.unk_token,
        sampling_rate=sampling_rate,
        max_input_len=max_input_len,
        min_input_len=min_input_len,
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
        "output_hidden_states": True,
        "decoder_start_token_id": tokenizer.bos_token_id,
        "encoder_apply_spec_augment": training_args.apply_spec_augment,
        "decoder_pos_emb_fixed": model_args.decoder_pos_emb_fixed
    }
    if hasattr(feature_extractor, "num_mel_bins"):
        base_model_config["num_mel_bins"] = feature_extractor.num_mel_bins

    # 4. Initialize seq2seq model
    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
            config=config)
    elif model_args.from_encoder_decoder_config:
        config = fetch_AED_config(model_args.base_encoder_model, model_args.base_decoder_model, base_model_config,
                                  model_args.config_overrides)
        model = JointCTCAttentionEncoderDecoder(config=config)
    else:
        model = JointCTCAttentionEncoderDecoder.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=model_args.base_encoder_model,
            decoder_pretrained_model_name_or_path=model_args.base_decoder_model,
            **base_model_config
        )

    gen_config = GenerationConfig(bos_token_id=base_model_config['bos_token_id'],
                                  pad_token_id=base_model_config['pad_token_id'],
                                  decoder_start_token_id=base_model_config['bos_token_id'],
                                  length_penalty=base_model_config['length_penalty'],
                                  early_stopping=base_model_config['early_stopping'],
                                  eos_token_id=base_model_config['eos_token_id'],
                                  max_length=base_model_config['max_length'],
                                  output_hidden_states=base_model_config['output_hidden_states'],
                                  num_beams=base_model_config['num_beams'])

    if gen_args.rescore_ctc_eos:
        gen_config.space_token_id = tokenizer("▁")['input_ids'][0]

    logger.info(f'Model updated gen config:\n {str(gen_config)}')
    model.generation_config = gen_config

    if model_args.task and model_args.language:
        model.config.suppress_tokens = []
        model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language=model_args.language,
                                                                           task=model_args.task)

    if training_args.decoder_cold_start:
        logger.info('Reinitializing decoder weights')
        model.decoder.apply(model.decoder._init_weights)

    if model_args.dec_adapters:
        model.decoder.add_adapter("dec_adapters", set_active=True)
        model.decoder.train_adapter("dec_adapters")

    # 5. Init trainer
    callbacks = []
    if training_args.apply_spec_augment:
        callbacks.append(AugmentationManagerCallback(training_args.num_steps_to_activate_spec_augment))
    if training_args.early_stopping_patience > -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    if training_args.track_ctc_loss:
        callbacks.append(AdditionalLossPrinterCallback())
    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   tokenizer=tokenizer,
                                                   padding=True,
                                                   sampling_rate=model_args.sampling_rate,
                                                   audio_path=data_args.audio_column_name,
                                                   text_path=data_args.text_column_name,
                                                   rename_features=training_args.collator_rename_features)
    trainer_class = AdditionalLossTrackerTrainer if training_args.track_ctc_loss else Seq2SeqTrainer
    training_eval_dataset = dataset[data_args.validation_split].select(
        range(data_args.validation_slice)) if data_args.validation_slice else dataset[data_args.validation_split]
    trainer = trainer_class(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred, gen_args.wandb_predictions_to_save),
    )

    # 6. Train
    if training_args.do_train:
        if gen_args.decoding_ctc_weight > 0 and training_args.joint_decoding_during_training:
            activate_joint_decoding(model, gen_args.decoding_ctc_weight, gen_args.ctc_margin, len(tokenizer),
                                    base_model_config['eos_token_id'], None, 0)
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 7. Evaluation
    if training_args.do_evaluate:
        if gen_args.decoding_ctc_weight > 0 or gen_args.external_lm_weight > 0:
            external_lm = None
            if gen_args.external_lm is not None:
                external_lm = AutoModelForCausalLM.from_pretrained(gen_args.external_lm)
                external_lm.eval()
            activate_joint_decoding(model, gen_args.decoding_ctc_weight, gen_args.ctc_margin, len(tokenizer),
                                    base_model_config['eos_token_id'], external_lm, gen_args.external_lm_weight)

        trainer.args.per_device_eval_batch_size = math.ceil(
            trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor)
        for split in training_args.evaluation_splits:
            predictions = trainer.predict(dataset[split], output_hidden_states=True,
                                          num_beams=model.generation_config.num_beams * gen_args.eval_beam_factor)
            logger.info(f"Metrics for {split} split: {predictions.metrics}")
            save_predictions(tokenizer, predictions,
                             f'{training_args.output_dir}/'
                             f'predictions_{split}_wer{100 * predictions.metrics["test_wer"]:.2f}.csv')
    if training_args.do_generate:
        if gen_args.decoding_ctc_weight > 0 or gen_args.external_lm_weight > 0:
            external_lm = None
            if gen_args.external_lm is not None:
                external_lm = AutoModelForCausalLM.from_pretrained(gen_args.external_lm)
                external_lm.eval()
            activate_joint_decoding(model, gen_args.decoding_ctc_weight, gen_args.ctc_margin, len(tokenizer),
                                    base_model_config['eos_token_id'], external_lm, gen_args.external_lm_weight)

        gen_config.num_return_sequences = gen_args.num_predictions_to_return
        gen_config.return_dict_in_generate = True
        gen_config.num_beams = model.generation_config.num_beams * gen_args.eval_beam_factor
        gen_config.output_scores = True
        trainer.args.per_device_eval_batch_size = math.ceil(
            trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor)
        for split in training_args.evaluation_splits:
            logger.info(f"Generating predictions for split: {split}")
            dataloader = trainer.get_eval_dataloader(dataset[split])
            n_bests = []
            scores = []
            labels = []
            outputs_agg = []
            for sample in tqdm.tqdm(dataloader):
                outputs = model.generate(generation_config=gen_config, **sample)
                if gen_args.save_output_states:
                    outputs_agg.append(postprocess_beam_outputs(outputs))
                n_bests.append(outputs.sequences)
                scores.append(outputs.sequences_scores)
                labels.append(sample['labels'])
            save_nbests(gen_args.nbest_path_to_save + "_" + split, n_bests, scores, labels, tokenizer,
                        group_size=gen_args.num_predictions_to_return, outputs=outputs_agg,
                        batch_size=trainer.args.per_device_eval_batch_size)
