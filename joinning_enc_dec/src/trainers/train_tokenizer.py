from datasets import load_dataset, load_from_disk
from tokenizers.implementations import SentencePieceUnigramTokenizer
from transformers import HfArgumentParser
from transformers.utils import logging

from trainers.training_arguments import TokenizerTrainingArguments

if __name__ == '__main__':
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((TokenizerTrainingArguments,))

    tokenizer_args = parser.parse_args_into_dataclasses()
    # 1. Load dataset
    if tokenizer_args.dataset_config is not None:
        dataset = load_dataset(tokenizer_args.dataset_name, tokenizer_args.dataset_config, keep_in_memory=False)
    else:
        dataset = load_from_disk(tokenizer_args.dataset_name, keep_in_memory=False)
    text = dataset[tokenizer_args.train_split][tokenizer_args.text_column_name]

    # 2. Instantiate tokenizer
    tokenizer = SentencePieceUnigramTokenizer()

    # 3. Train tokenizer
    tokenizer.train_from_iterator(text,
                                  vocab_size=tokenizer_args.vocab_size,
                                  special_tokens=["<unk>", "<s>", "<pad>", "</s>"]
                                  )

    for k in tokenizer.get_vocab():
        print(k)
