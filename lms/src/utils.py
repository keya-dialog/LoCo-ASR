import logging
from datetime import datetime
import numpy as np
from typing import List


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


def load_text(fname, tokenizer, ignore_one_tok: bool = True):
    """Load plain text line-by-line"""

    text_ids = []
    lengths = []
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            text = line.strip()
            if text:
                tok_ids = tokenizer(text)["input_ids"]
                tok_ids.append(tokenizer.eos_token_id)

                if len(tok_ids) > 1:
                    text_ids.append(tok_ids)
                    lengths.append(len(tok_ids))

                else:
                    ign += 1

    if ignore_one_tok:
        print("Ignored single-token sentences", ign)

    lengths = np.asarray(lengths)
    sort_ixs = np.argsort(lengths)

    sorted_text_ids = []
    for i in sort_ixs:
        sorted_text_ids.append(text_ids[i])

    return sorted_text_ids, lengths[sort_ixs]


def load_key_text_file(fname, tokenizer, ignore_one_tok: bool = True):
    """Load key text file, tokenize and add end_of_token"""

    utt_id2text = {}
    utt_ids = []
    text_ids = []
    lengths = []
    ign = 0

    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            utt_id, text = line.split(" ", maxsplit=1)
            if utt_id in utt_id2text:
                print("Duplicate utt id:", utt_id, utt_id2text[utt_id], "ignoring")

            else:
                tok_ids = tokenizer(text)["input_ids"]
                tok_ids.append(tokenizer.eos_token_id)

                if len(tok_ids) > 1:
                    utt_ids.append(utt_id)
                    utt_id2text[utt_id] = text
                    text_ids.append(tok_ids)
                    lengths.append(len(tok_ids))

                else:
                    ign += 1

    if ignore_one_tok:
        print("Ignored single-token sentences", ign)

    lengths = np.asarray(lengths)
    sort_ixs = np.argsort(lengths)

    sorted_text_ids = []
    sorted_utt_ids = []
    for i in sort_ixs:
        sorted_text_ids.append(text_ids[i])
        sorted_utt_ids.append(utt_ids[i])

    return sorted_text_ids, lengths[sort_ixs], sorted_utt_ids


def compute_ppl_per_recording(nlls: list, utt_ids: list) -> List[dict]:
    """Compute perplexity (PPL) per recording, given the token-level nlls,
    and the corresponding utternace (or block) ids.
    Expects that rec_id can be inferred from utt_id as follows
    `rec_id = utt_id.split("-")[0]`

    Args:
        nlls (list): A nested list, where inside list corresponds the negLLH
          of every token in the utterance or block
        utt_ids (list): List of utterance IDs or block ID, where utterances
          and block belong to recording

    Returns:
        dict: rec_id to nll mapping (huge dictionary)
        dict: rec_id to ppl mapping (small dictionary)
    """

    logger = logging.getLogger()

    rec_id2nlls = {}
    for i, utt_id in enumerate(utt_ids):
        rec_id = utt_id.split("-", maxsplit=1)[0]

        if rec_id not in rec_id2nlls:
            rec_id2nlls[rec_id] = []
        rec_id2nlls[rec_id].extend(nlls[i])

    rec_id2ppl = {}
    ppls = []
    for rec_id, rec_ppls in rec_id2nlls.items():
        ppl = np.exp(np.mean(rec_ppls))
        rec_id2ppl[rec_id] = ppl
        ppls.append(ppl)

    logger.info(
        "Avg. PPL of recordings: %.1f | std.dev: %.1f | min PPL %.1f | max PPL %.1f",
        np.mean(ppls),
        np.std(ppls),
        np.min(ppls),
        np.max(ppls),
    )
    return [rec_id2nlls, rec_id2ppl]


# def test_key_file():

# load_key_file(args.fname)


# if __name__ == "__main__":

# import os
# import sys
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--fname", default="", help="input file")
# parser.add_argument("--test_key_file", action="store_true")

# args = parser.parse_args()


# if args.test_key_file:
#    if os.path.isfile(args.fname):
#        print("--fname is required and must point to a file")
#    else:
#        test_key_file(args.fname)
