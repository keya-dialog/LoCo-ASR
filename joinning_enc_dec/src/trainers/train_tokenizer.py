from datasets import load_dataset, load_from_disk
from huggingface_hub import repo_exists
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, trainers
from tokenizers.models import BPE, Unigram
from transformers import HfArgumentParser, PreTrainedTokenizerFast
from transformers.utils import logging

from trainers.training_arguments import TokenizerTrainingArguments


def train_tokenizer(tokenizer_type, tokenizer_name, text_iterator, vocab_size=5000, apply_regularization=False):
    if apply_regularization:
        raise NotImplementedError

    if tokenizer_type == 'BPE':
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                      special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                                      unk_token="<unk>")
        tokenizer.decoder = decoders.ByteLevel()
    elif tokenizer_type == 'unigram':
        tokenizer = Tokenizer(Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size,
                                          special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
                                          unk_token="<unk>")
        tokenizer.decoder = decoders.Metaspace()

    elif tokenizer_type == 'WPC':
        # tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        # trainer = WordPieceTrainer(special_tokens=spl_tokens)
        raise NotImplementedError

    else:
        # tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        # trainer = WordLevelTrainer(special_tokens=spl_tokens)
        raise NotImplementedError

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Replace("``", '"'),
         normalizers.Replace("''", '"'),
         normalizers.Lowercase()
         ]
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token='<pad>',
        unk_token='<unk>',
        mask_token='<mask>',
    )

    wrapped_tokenizer.push_to_hub(tokenizer_name)

    return tokenizer


if __name__ == '__main__':
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((TokenizerTrainingArguments,))

    tokenizer_args, = parser.parse_args_into_dataclasses()

    if tokenizer_args.skip_if_exists is not None and repo_exists(tokenizer_args.skip_if_exists):
        logger.warning(f"Tokenizer {tokenizer_args.skip_if_exists} already exists. Skipping training.")
        exit(0)

    # 1. Load dataset
    if tokenizer_args.dataset_config is not None:
        dataset = load_dataset(tokenizer_args.dataset_name, tokenizer_args.dataset_config, keep_in_memory=False)
    else:
        dataset = load_from_disk(tokenizer_args.dataset_name, keep_in_memory=False)

    # 2. Extract text
    text = dataset[tokenizer_args.train_split][tokenizer_args.text_column_name]
    if tokenizer_args.additional_raw_data is not None:
        text += load_dataset("text", data_files=tokenizer_args.additional_raw_data, keep_linebreaks=False)["train"][
            "text"]
    train_tokenizer(tokenizer_args.tokenizer_type, tokenizer_args.tokenizer_name, text, tokenizer_args.vocab_size)
