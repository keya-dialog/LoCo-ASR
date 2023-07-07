from transformers import TokenClassificationPipeline, AutoTokenizer, AutoModelForTokenClassification
from argparse import ArgumentParser, Namespace
from datasets import load_from_disk
from typing import List
import json


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Preprocessing script to generate FISHER dataset.')
    parser.add_argument('--model', required=True, type=str, help='')
    parser.add_argument('--dataset_in', required=True, type=str, help='')
    parser.add_argument('--out_file', required=True, type=str, help='')
    parser.add_argument('--recordings_list', required=True, type=str, help='')
    parser.add_argument('--batch_size', default=2, type=int, help='')
    return parser.parse_args()


class CustomPipeline(TokenClassificationPipeline):
    def group_entities(self, entities: List[dict]) -> List[dict]:
        return entities


def label_entities(samples):
    outputs = classifier(samples)
    outputs_joined = [[model.config.label2id[word['entity']] for word in sample] for sample in outputs]
    return {"entities": outputs_joined}


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    recordings_list = open(args.recordings_list).read().splitlines()
    classifier = CustomPipeline(task='ner', model=model, tokenizer=tokenizer, aggregation_strategy='max',
                                ignore_labels=[])
    dataset = load_from_disk(args.dataset_in, keep_in_memory=False)['test']
    filtered_dataset = dataset.filter(lambda x: x['recording'].split('-')[0] in recordings_list)
    dataset_with_entities = dataset.map(label_entities, batched=True, batch_size=args.batch_size,
                                        input_columns=["labels"])
    id2label = model.config.id2label
    predictions = []
    for prediction in dataset_with_entities:
        prediction_object = {
            "data": {
                "topic": prediction['topic'],
                "utt_id": prediction['uttid'],
                "text": prediction['labels']
            },
            "predictions": [{"result": []}]
        }
        start = 0
        for index, word in enumerate(prediction['labels'].split()):
            end = start + len(word)
            predicted_entity = prediction['entities'][index]
            prediction_object['predictions'][0]["result"].append(
                {
                    "value": {
                        "start": start,
                        "end": end,
                        "text": word,
                        "labels": [
                            id2label[predicted_entity]
                        ]
                    },
                    "from_name": "entity",
                    "to_name": "text",
                    "type": "labels"
                },
            )
            start = end + 1
        predictions.append(prediction_object)
    json.dump(predictions, open(args.out_file, 'w'))
