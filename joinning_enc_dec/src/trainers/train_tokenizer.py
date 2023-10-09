import sentencepiece as spm
from datasets import load_dataset, load_from_disk
from transformers import DebertaV2Tokenizer, HfArgumentParser
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

    # 2. Extract text
    text = dataset[tokenizer_args.train_split][tokenizer_args.text_column_name]

    # 3. Save to file, sentence per line
    sentence_per_line = "\n".join(text)
    with open(tokenizer_args.raw_text_file, "w") as f:
        f.write(sentence_per_line)

    # 4. Train sentencepiece tokenizer
    spm.SentencePieceTrainer.Train(
        input=tokenizer_args.raw_text_file,
        model_prefix=tokenizer_args.model_name,
        pad_id=3,
        pad_piece='<pad>',
        vocab_size=tokenizer_args.vocab_size,
        model_type='unigram'
    )

    # 5. Instantiate tokenizer and push to hub
    tokenizer_deberta = DebertaV2Tokenizer(
        vocab_file=f"{tokenizer_args.model_name}.model",
        bos_token='<s>',
        cls_token='<s>',
        sep_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        sp_model_kwargs={
            'enable_sampling': True,
            'nbest_size': -1,
            'alpha': 0.1,
        })
    tokenizer_deberta.push_to_hub(tokenizer_args.model_name)
