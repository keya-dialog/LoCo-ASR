import glob
import os
import pickle
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer, HfArgumentParser, \
    LogitsProcessor, LogitsProcessorList, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.utils import logging

from per_utterance.models import JointCTCAttentionEncoderDecoder, JointCTCAttentionEncoderDecoderConfig
from utils import Seq2SeqDataCollatorWithPadding, compute_metrics, filter_out_sequence_from_dataset

AutoConfig.register("joint_aed_ctc_speech-encoder-decoder", JointCTCAttentionEncoderDecoderConfig)
AutoModelForSpeechSeq2Seq.register(JointCTCAttentionEncoderDecoderConfig, JointCTCAttentionEncoderDecoder)


class EnforceEosIfCTCStops(LogitsProcessor):
    """Logits processor (to use with HuggingFace `generate()` method :
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/
    text_generation#transformers.generation_utils.GenerationMixin).

    This logit processor simply ensure that after hitting logzero likelihood for all tokens eos is generated.

    Args:
        eos_token_id (int): ID of the EOS token.
        log_thr (float): Value to use for logzero.
    """

    def __init__(self, eos_token_id: int, log_thr: float = -10000000000.0):
        super().__init__()
        self.log_thr = log_thr
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        should_enforce_stop = scores.max(dim=1).values <= self.log_thr
        mask = should_enforce_stop.unsqueeze(dim=-1).expand(scores.size())
        eos_mask = torch.zeros_like(mask, dtype=torch.bool)
        eos_mask[:, self.eos_token_id] = True
        mask = mask & eos_mask
        scores = torch.where(~mask, scores, self.log_thr / 2)
        return scores


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
    ctc_margin: Optional[int] = field(
        default=0, metadata={"help": "Margin to stop generation."}
    )
    ctc_weight: Optional[float] = field(
        default=0, metadata={"help": "CTC weight to bias hypothesis."}
    )
    ctc_beam_width: Optional[int] = field(
        default=None, metadata={"help": "Width of the CTC beam."}
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

    for split in [data_args.validation_split, data_args.test_split]:
        dataset[split] = filter_out_sequence_from_dataset(dataset[split],
                                                          max_input_len=data_args.max_duration_in_seconds,
                                                          min_input_len=data_args.min_duration_in_seconds)

    model_path = model_args.model

    if model_args.average_checkpoints:
        model_path = average_checkpoints(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   tokenizer=tokenizer,
                                                   padding=True)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        data_collator=data_collator,
    )
    if model_args.with_ctc:
        def new_beam(*args, **kwargs):
            logits_processor = LogitsProcessorList(
                [EnforceEosIfCTCStops(tokenizer.eos_token_id,
                                      log_thr=-10000000000.0 * model_args.ctc_weight if model_args.ctc_weight > 0 else -10000000000.0)])
            kwargs.update({"logits_processor": logits_processor})
            return model.joint_beam_search(*args, **kwargs,
                                           ctc_weight=model_args.ctc_weight,
                                           margin=model_args.ctc_margin,
                                           ctc_beam_width=len(tokenizer))


        model.beam_search = new_beam

    predictions = trainer.predict(dataset[data_args.validation_split], output_hidden_states=True)
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'val_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)

    predictions = trainer.predict(dataset[data_args.test_split], output_hidden_states=True)
    logger.info(compute_metrics(tokenizer, predictions))
    with open(os.path.join(training_args.output_dir, 'test_predictions'), 'wb') as fp:  # Overwrites any existing file.
        pickle.dump(predictions, fp, pickle.HIGHEST_PROTOCOL)
