from transformers import TokenClassificationPipeline, AutoTokenizer, AutoModelForTokenClassification
from argparse import ArgumentParser, Namespace
from datasets import load_from_disk
from typing import List


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Preprocessing script to generate FISHER dataset.')
    parser.add_argument('--model', required=True, type=str, help='')
    parser.add_argument('--dataset_in', required=True, type=str, help='')
    parser.add_argument('--dataset_out', required=True, type=str, help='')
    parser.add_argument('--batch_size', default=2, type=int, help='')
    parser.add_argument('--known_words_file', required=True, type=str, help='')
    return parser.parse_args()


class CustomPipeline(TokenClassificationPipeline):
    def group_entities(self, entities: List[dict]) -> List[dict]:
        return entities


def label_entities(samples):
    outputs = classifier(samples)
    outputs_joined = [[model.config.label2id[word['entity']] for word in sample] for sample in outputs]
    oov = [[0 if word in known_words else 1 for word in sample.split()] for sample in samples]
    return {"entities": outputs_joined,
            "oov": oov}


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    known_words = open(args.known_words_file).read().splitlines()
    classifier = CustomPipeline(task='ner', model=model, tokenizer=tokenizer, aggregation_strategy='max',
                                ignore_labels=[])
    dataset = load_from_disk(args.dataset_in, keep_in_memory=False)
    cols_to_be_removed = dataset['test'].column_names
    cols_to_be_removed.remove('labels')
    processed_dataset = dataset.map(label_entities, batched=True, batch_size=args.batch_size,
                                    remove_columns=cols_to_be_removed, input_columns=["labels"])
    processed_dataset.save_to_disk(args.dataset_out)
