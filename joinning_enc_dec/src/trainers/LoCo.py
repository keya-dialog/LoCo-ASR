import os
import pickle

from datasets import load_from_disk
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCTC, AutoModelForCausalLM, \
    AutoModelForSpeechSeq2Seq, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging

from context_aware.decoder import GPT2ConfigWithContext, GPT2WithContextLMHeadModel
from context_aware.encoder import Wav2Vec2ConfigWithContext, Wav2Vec2WithContextForCTC
from context_aware.trainer import ContextAwareTrainer
from context_aware.utils import ContextContainer
from per_utterance.models import JointCTCAttentionEncoderDecoder, JointCTCAttentionEncoderDecoderConfig
from trainers.training_arguments import DataTrainingArguments, GeneralTrainingArgumentsContext, GenerationArguments, \
    ModelArgumentsContext
from utils import Seq2SeqDataCollatorWithPaddingAndConvId, compute_metrics, filter_out_sequence_from_dataset

AutoConfig.register("gpt2-with-context", GPT2ConfigWithContext)
AutoConfig.register("wav2vec2-with-context", Wav2Vec2ConfigWithContext)
AutoConfig.register("joint_aed_ctc_speech-encoder-decoder", JointCTCAttentionEncoderDecoderConfig)

AutoModelForCTC.register(Wav2Vec2ConfigWithContext, Wav2Vec2WithContextForCTC)
AutoModelForCausalLM.register(GPT2ConfigWithContext, GPT2WithContextLMHeadModel)
AutoModelForSpeechSeq2Seq.register(JointCTCAttentionEncoderDecoderConfig, JointCTCAttentionEncoderDecoder)

logger = logging.get_logger("transformers")

if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArgumentsContext, DataTrainingArguments, GeneralTrainingArgumentsContext, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Load dataset
    dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)
    for split in [data_args.train_split, data_args.validation_split, data_args.test_split]:
        dataset[split] = filter_out_sequence_from_dataset(dataset[split],
                                                          max_input_len=data_args.max_duration_in_seconds,
                                                          min_input_len=data_args.min_duration_in_seconds)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.from_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(model_args.from_pretrained)

    # 3. Initialize model config and add memory cells
    config = AutoConfig.from_pretrained(model_args.from_pretrained)
    config.encoder.update({"memory_dim": model_args.enc_memory_dim,
                           "memory_cells": model_args.enc_memory_cells_location or [],
                           "layer_norm_epsilon": config.encoder.layer_norm_eps})
    config.decoder.update({"memory_dim": model_args.dec_memory_dim,
                           "memory_cells": model_args.dec_memory_cells_location or [],
                           "attention_dropout": config.decoder.attn_pdrop,
                           "hidden_dropout": config.decoder.resid_pdrop,
                           "activation_dropout": config.decoder.resid_pdrop,
                           "hidden_act": config.decoder.activation_function,
                           "intermediate_size": config.decoder.n_inner or 4 * config.decoder.hidden_size})
    if max(config.encoder.memory_cells, default=0) >= config.encoder.num_hidden_layers or max(
            config.decoder.memory_cells, default=0) >= config.decoder.n_layer:
        raise ValueError("Memory cell location cannot be greater than number of layers")

    encoder_model_type = config.encoder.model_type + "-with-context"
    decoder_model_type = config.decoder.model_type + "-with-context"

    config.encoder = CONFIG_MAPPING[encoder_model_type].from_dict(config.encoder.to_dict())
    config.encoder.model_type = encoder_model_type
    config.decoder = CONFIG_MAPPING[decoder_model_type].from_dict(config.decoder.to_dict())
    config.decoder.model_type = decoder_model_type

    # 5. Initialize model and freeze params except memory cells
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.from_pretrained,
        config=config,
    )
    if training_args.freeze_others:
        model.freeze()
    model.activate_memory_params()

    # 4. Initialize context container and connect it with memory cells
    context_container = ContextContainer(config)
    model.connect_context_container(context_container)

    early_stopping = EarlyStoppingCallback(training_args.early_stopping_patience)
    data_collator = Seq2SeqDataCollatorWithPaddingAndConvId(feature_extractor=feature_extractor,
                                                            tokenizer=tokenizer,
                                                            padding=True, sampling_rate=model_args.sampling_rate)

    trainer = ContextAwareTrainer(
        args=training_args,
        model=model,
        callbacks=[early_stopping],
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.validation_split],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred),
        context_container=context_container
    )

    # 5. Train
    trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    predictions = trainer.predict(dataset[data_args.validation_split])
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'val_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)

    # 6. Eval on test
    predictions = trainer.predict(dataset[data_args.test_split])
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'test_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)
