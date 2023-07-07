from argparse import ArgumentParser, Namespace
from datasets import load_from_disk
from jiwer import compute_measures
from transformers import TokenClassificationPipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Preprocessing script to generate FISHER dataset.')
    parser.add_argument('--dataset', required=True, type=str, help='')
    parser.add_argument('--model', required=True, type=str, help='')
    return parser.parse_args()


class CustomPipeline(TokenClassificationPipeline):
    def group_entities(self, entities: List[dict]) -> List[dict]:
        return entities


def named_entity_wer(classifier, ref, hyp):
    ref_entities = classifier(ref)
    hyp_entities = classifier(hyp)
    metrics_base = compute_measures(ref, hyp)
    metrics_named_entities = compute_measures(ref_entities, hyp_entities)
    return metrics_base, metrics_named_entities

def oov_wer(classifier, ref, hyp):
    ref_oovs = classifier(ref)
    hyp_oovs = classifier(hyp)
    metrics_base = compute_measures(ref, hyp)
    metrics_named_entities = compute_measures(ref_oovs, hyp_oovs)
    return metrics_base, metrics_named_entities


if __name__ == '__main__':
    args = parse_args()
    dataset = load_from_disk(args.dataset, keep_in_memory=False)['test']
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    known_words = open(args.known_words_file).read().splitlines()
    pipeline = CustomPipeline(task='ner', model=model, tokenizer=tokenizer, aggregation_strategy='max',
                                ignore_labels=[])
    dataset = dataset.add_column('predictions', dataset['labels'])

    predictions = dataset['predictions']
    predictions[0] = 'haha test'
    labels = dataset['labels']



    dataset.to_csv("test.csv")
