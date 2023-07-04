
import logging
from datetime import datetime


def create_logger(log_file_base, verbose: bool):
    """Create logger"""

    now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=f"{log_file_base}_{now_str}.log",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    if verbose:
        logger.addHandler(logging.StreamHandler())
    return logger


def load_text(fname):
    """Load plain text line-by-line"""

    texts = []
    with open(fname, 'r', encoding='utf-8') as fpr:
        for line in fpr:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def load_key_file(fname, sort_by_recids=True):
    """Load kaldi-formatted text file, where first col is utterance ID"""

    utt2text = {}
    recid2uttids = {}

    with open(fname, 'r', encoding='utf-8') as fpr:
        for line in fpr:
            line = line.strip()

            uttid, text = line.split(" ", maxsplit=1)
            recid = uttid.split("-", maxsplit=1)

            if uttid not in utt2text:
                utt2text[uttid] = text


def test_key_file():

    load_key_file(args.fname)


if __name__ == "__main__":

    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", default="", help="input file")
    parser.add_argument("--test_key_file", action="store_true")

    args = parser.parse_args()


    if args.test_key_file:
        if os.path.isfile(args.fname):
            print("--fname is required and must point to a file")
        else:
            test_key_file(args.fname)
