#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 29 Jun 2023
# Last modified : 29 Jun 2023

"""
Split fisher data (train_all/ in Kaldi structure) into
training, dev and test splits given the recording ids
"""

import os
import glob
import argparse


def write_list_to_file(lst, fname):
    """Write list line by line in the file"""

    with open(fname, "w", encoding="utf-8") as fpw:
        fpw.write("\n".join(lst) + "\n")
    print(fname, "saved.")


def load_list(fname):
    """Load file into a list line-by-line"""

    print("Loading from", fname)
    lst = []
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            if line.strip():
                lst.append(line.strip())
    return lst


def get_subset(fname, subset_ids):
    """Filter lines based on subset of rec ids"""

    subset_ids = set(subset_ids)

    print("subset ids to look for", len(subset_ids))

    subset_lines = []

    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            rid_or_uid, _ = line.split(" ", maxsplit=1)

            recid = rid_or_uid.split("-")[0]

            if recid in subset_ids:
                subset_lines.append(line)

    print("Lines found", len(subset_lines))

    return subset_lines


def main():
    """main method"""

    args = parse_arguments()

    os.makedirs(args.out_data_dir, exist_ok=True)

    splits = {}
    for split_name in args.sets:
        splits[split_name] = load_list(
            os.path.join(args.split_dir, f"{split_name}.recids")
        )
        print(split_name, "rec ids:", len(splits[split_name]))
        os.makedirs(os.path.join(args.out_data_dir, split_name), exist_ok=True)

    for fname in glob.glob(args.data_dir + "/*"):
        if os.path.isfile(fname):

            base = os.path.basename(fname)

            for split_name in splits:
                out_fname = os.path.join(args.out_data_dir, f"{split_name}/{base}")

                if base in ("frame_shift"):
                    os.system(f"cp -v {fname} {out_fname}")
                else:

                    subset_lines = get_subset(fname, splits[split_name])

                    write_list_to_file(subset_lines, out_fname)


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "split_dir",
        help="fisher topic split dir with train, dev, test recids (eg: fisher_topic_split/)",
    )
    parser.add_argument(
        "data_dir",
        help="data dir where all the fisher data is prepared in kaldi/espnet format (eg: data/train_all/)",
    )
    parser.add_argument(
        "out_data_dir",
        help="out data dir that follows the same structure as data dir but with splits accorinding to contents from split dir",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["train", "dev", "test"],
        help="the set name should match the file base name in split_dir",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
