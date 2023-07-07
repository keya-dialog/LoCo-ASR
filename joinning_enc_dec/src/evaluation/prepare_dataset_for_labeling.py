import json
from argparse import ArgumentParser, Namespace
from datasets import load_from_disk

id2label = {
    0: "B-LOC",
    1: "B-MISC",
    2: "B-ORG",
    3: "I-LOC",
    4: "I-MISC",
    5: "I-ORG",
    6: "I-PER",
    7: "O"
}


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Preprocessing script to generate FISHER dataset.')
    parser.add_argument('--dataset', required=True, type=str, help='')
    parser.add_argument('--out_file', required=True, type=str, help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = load_from_disk(args.dataset, keep_in_memory=False)['test']
    predictions = []
    for prediction in dataset:
        prediction_object = {
            "data": {
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
