from dataclasses import dataclass, field

from datasets import load_from_disk
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechEncoderDecoderModel
from transformers.utils import logging

from models import JointCTCAttentionEncoderDecoder
from utils import Seq2SeqDataCollatorWithPadding, compute_metrics


@dataclass
class ModelArguments:
    model: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    with_ctc: bool = field(default=False, metadata={"help": "To use model trained with cuda"})


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


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model)

    if model_args.with_ctc:
        model = JointCTCAttentionEncoderDecoder.from_pretrained(model_args.model)
    else:
        model = SpeechEncoderDecoderModel.from_pretrained(model_args.model)

    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   decoder_tokenizer=tokenizer,
                                                   padding=True)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        data_collator=data_collator,
    )

    if model_args.with_ctc:
        model.config.output_hidden_states = True

    predictions = trainer.predict(dataset[data_args.validation_split])
    logger.info(compute_metrics(tokenizer, predictions))

    predictions = trainer.predict(dataset[data_args.test_split])
    logger.info(compute_metrics(tokenizer, predictions))
