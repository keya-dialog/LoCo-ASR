import glob
import os
import pickle
import shutil
from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechEncoderDecoderModel
from transformers.utils import logging

from per_utterance.models import JointCTCAttentionEncoderDecoder
from utils import Seq2SeqDataCollatorWithPadding, compute_metrics


@dataclass
class ModelArguments:
    model: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    with_ctc: bool = field(
        default=False, metadata={"help": "To use model trained with cuda"}
    )
    average_checkpoints: bool = field(
        default=False, metadata={"help": "Whether to average last checkpoints"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    validation_split: str = field(
        default="validation", metadata={"help": "Validation split to be used."}
    )
    test_split: str = field(
        default="test", metadata={"help": "Test split to be used."}
    )


def average_dicts(*dicts):
    result = {}

    # Count the number of dictionaries
    num_dicts = len(dicts)

    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value

    return result, num_dicts


def average_checkpoints(experiment_dir):
    checkpoints = glob.glob(f"{experiment_dir}/checkpoint*/pytorch_model.bin")
    state_dicts = [torch.load(checkpoint) for checkpoint in checkpoints]
    sum_state_dict, n_checkpoints = average_dicts(*state_dicts)
    del state_dicts
    average_dict = {key: sum_state_dict[key].div(n_checkpoints) for key in sum_state_dict}
    dst_path = os.path.join(experiment_dir, "average_checkpoint")
    shutil.copytree(os.path.dirname(checkpoints[0]), dst_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(experiment_dir, "tokenizer"), dst_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(experiment_dir, "feature_extractor"), dst_path, dirs_exist_ok=True)
    torch.save(average_dict, os.path.join(dst_path, "pytorch_model.bin"))
    return dst_path


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)
    model_path = model_args.model

    if model_args.average_checkpoints:
        model_path = average_checkpoints(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    if model_args.with_ctc:
        model = JointCTCAttentionEncoderDecoder.from_pretrained(model_path)
    else:
        model = SpeechEncoderDecoderModel.from_pretrained(model_path)

    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   tokenizer=tokenizer,
                                                   padding=True)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        data_collator=data_collator,
    )

    predictions = trainer.predict(dataset[data_args.validation_split], output_hidden_states=model_args.with_ctc)
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'val_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)

    predictions = trainer.predict(dataset[data_args.test_split], output_hidden_states=model_args.with_ctc)
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'test_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)
