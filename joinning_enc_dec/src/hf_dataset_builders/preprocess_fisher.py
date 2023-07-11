from argparse import ArgumentParser, Namespace
import datasets


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Preprocessing script to generate FISHER dataset.')
    parser.add_argument('dataset', type=str, help='')
    parser.add_argument('metadata_dir', type=str, help='')
    parser.add_argument('output_dir', type=str, help='')
    parser.add_argument('--num_proc', type=int, default=1, help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_dataset(args.dataset, keep_in_memory=False,
                                    metadata_dir=args.metadata_dir, num_proc=args.num_proc)
    dataset.save_to_disk(args.output_dir, num_proc=args.num_proc)
