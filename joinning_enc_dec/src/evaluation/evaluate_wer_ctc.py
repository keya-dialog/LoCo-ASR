from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer, BatchFeature, HfArgumentParser, \
    PreTrainedTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, Wav2Vec2FeatureExtractor

from utils import compute_metrics


@dataclass
class Seq2SeqDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.SequenceFeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizerFast`)
            The processor used for processing the text data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    Based upon: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/
                /Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: PreTrainedTokenizerFast
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000

    def _encapsulate_utterance(self, utterance):
        return utterance

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = self.feature_extractor([feature["audio"]['array'] for feature in features], padding=True,
                                                sampling_rate=self.sampling_rate)
        labels = self.tokenizer.batch_encode_plus(
            [self._encapsulate_utterance(feature['normalized_text']) for feature in features],
            return_attention_mask=True,
            padding='longest', return_tensors='pt')

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if "input_features" in batch:
            batch["input_values"] = batch["input_features"]
            del batch["input_features"]
        return batch


@dataclass
class Seq2SeqTrainingArgumentsExtended(Seq2SeqTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_part: str = field(
        default=None, metadata={"help": "The part of the dataset to use (via the datasets library)."}
    )
    model: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


if __name__ == '__main__':
    parser = HfArgumentParser(Seq2SeqTrainingArgumentsExtended)

    args = parser.parse_args_into_dataclasses()[0]
    dataset = load_dataset(args.dataset_name, args.dataset_part)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    feature_extractor.do_normalize = True
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model)
    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor, tokenizer=tokenizer,
                                                   padding=True)
    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(tokenizer, x)
    )

    predictions = trainer.evaluate(dataset["test"])
    print(predictions)
