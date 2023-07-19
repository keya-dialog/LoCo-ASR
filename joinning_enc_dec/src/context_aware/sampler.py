from argparse import ArgumentParser, Namespace

import datasets
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Preprocessing script to generate FISHER dataset.')
    parser.add_argument('--dataset', required=True, type=str, help='')
    parser.add_argument('--len_column', type=str, default='n_turns', help='')
    parser.add_argument('--batch_size', default=2, type=int, help='')
    parser.add_argument('--split', type=str, default='train', help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)
    sampler = LengthGroupedSampler(
        batch_size=args.batch_size,
        dataset=dataset[args.split],
        lengths=dataset[args.split][args.len_column]
    )
    dataloader = DataLoader(dataset=dataset[args.split], batch_size=args.batch_size, sampler=sampler,
                            collate_fn=lambda x: x)
    iterator = iter(dataloader)
    batch_of_conv = next(iterator)
    print(batch_of_conv)
