import logging
import sys
import os
import torch
import glob
import numpy as np
from typing import Any, List, Tuple, Dict
from datetime import datetime
from collections import defaultdict
from torch.utils.data import IterableDataset


def save_checkpoint(obj, exp_dir: str, keep: int = 5) -> None:
    r"""Save checkpoint (variables)

    Args:
        exp_dir (str): Path to experiment's directory
        keep (int): Number of how many checkpoint we want keep (Default: 5)
    """
    logger = logging.getLogger()
    now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fname = exp_dir / f"checkpoint-{now_str}.pt"

    torch.save(obj, fname)

    checkpoints = glob.glob(str(exp_dir / f"checkpoint-*"))
    if len(checkpoints) > keep:
        for c in checkpoints[: len(checkpoints) - keep]:
            os.remove(c)

    return


def load_checkpoint(exp_dir: str) -> Dict[str, Any]:
    logger = logging.getLogger()
    checkpoints = glob.glob(str(exp_dir / f"checkpoint-*"))
    if checkpoints:
        for i in range(1, len(checkpoints) + 1):
            try:
                obj = torch.load(checkpoints[-i])
                logger.info(f"Checkpoint {checkpoints[-i]} loaded")
                break
            except RuntimeError:
                logger.info(
                    f"Checkpoint {checkpoints[-i]} is corrupted ... try another one"
                )
    else:
        obj = {}

    return obj


class FisherTextDatasetIndep(IterableDataset):
    """Dataset"""

    def __init__(self, fname: str, tokenizer, batch_size: int = 128):
        """Init"""

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_ids, self.lengths, self.utt_ids = self._load_key_text_file(fname)
        self.bins, self.counts = np.unique(self.lengths, return_counts=True)
        self.sentence_done = 0

    def __iter__(self):
        return iter(self._next_batch())

    def __len__(self):
        return len(self.utt_ids)

    def _next_batch(self):
        """Generate batches"""

        bno = 1  # batch number
        offset = 0
        for bin, count in zip(self.bins, self.counts):
            for i in range(offset, offset + count, self.batch_size):
                start = i
                end = i + self.batch_size
                if i + self.batch_size > offset + count:
                    end = offset + count
                # print(f"\r batch {bno:5d} length bin {bin:3d} start ix {start:8d} : {end:8d} / {len(self.text_ids):8d}", end=" ", file=sys.stderr)
                yield self.text_ids[start:end]
                bno += 1
            offset += count
            # print()

    def _load_key_text_file(self, fname, ignore_one_tok: bool = True):
        """Load key text file, tokenize and add end_of_token"""

        utt_id2text = {}
        utt_ids = []
        text_ids = []
        lengths = []
        ign = 0

        with open(fname, "r", encoding="utf-8") as fpr:
            for line in fpr:
                line = line.strip()
                utt_id, text = line.split(None, maxsplit=1)
                if utt_id in utt_id2text:
                    print(
                        "Duplicate utt id:",
                        utt_id,
                        utt_id2text[utt_id],
                        "ignoring",
                        file=sys.stderr,
                    )

                else:
                    tok_ids = self.tokenizer(text)["input_ids"]
                    tok_ids.append(self.tokenizer.eos_token_id)
                    tok_ids.insert(0, self.tokenizer.bos_token_id)

                    if len(tok_ids) > 1:
                        utt_ids.append(utt_id)
                        utt_id2text[utt_id] = tok_ids
                        text_ids.append(tok_ids)
                        lengths.append(len(tok_ids))
                    else:
                        ign += 1

        if ignore_one_tok:
            print("Ignored single-token sentences", ign, file=sys.stderr)

        lengths = np.asarray(lengths)
        sort_ixs = np.argsort(lengths)

        sorted_text_ids = []
        sorted_utt_ids = []
        for i in sort_ixs:
            sorted_text_ids.append(text_ids[i])
            sorted_utt_ids.append(utt_ids[i])

        return sorted_text_ids, lengths[sort_ixs], sorted_utt_ids


class FisherTextDatasetMaxLen(IterableDataset):
    """Dataset"""

    def __init__(
        self,
        fname: str,
        tokenizer,
        batch_size: int = 128,
        context_type: str = "indep",
        max_len: int = 128,
    ):
        """Init"""
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.context_type = context_type
        self.utt_id2text, self.rec_id2tokens = self._load_text(fname)
        self.nrecording = len(self.rec_id2tokens)

        self.ids_done = []
        self.sentence_done = 0

    def __iter__(self):
        return iter(self._next_batch())

    def _load_text(self, fname):
        def recid_time(x):
            rec, _, start, end = x[0].split("-")
            return "-".join((rec, start, end))

        utt_id2text = {}
        rec_id2tokens = defaultdict(list)

        with open(fname, "r", encoding="utf-8") as fd:
            for line in fd:
                utt_id, text = line.strip().split(None, maxsplit=1)
                if utt_id in utt_id2text:
                    print(f"Duplicate utt id: {utt_id} ignoring", file=sys.stderr)
                else:
                    utt_id2text[utt_id] = text

        utt_ids_sorted = sorted(utt_id2text.items(), key=lambda x: recid_time(x))

        for utt_id, text in utt_ids_sorted:
            rec_id = utt_id.split("-", maxsplit=1)[0]
            rec_id2tokens[rec_id].extend(self.tokenizer(text)["input_ids"])
            rec_id2tokens[rec_id].append(self.tokenizer.eos_token_id)

        return utt_id2text, rec_id2tokens

    def __len__(self):
        nsentence = 0
        for _, v in self.rec_id2tokens.items():
            if len(v) < self.max_len:
                nsentence += 1
            else:
                nsentence += 1 + (len(v) - self.max_len)

        return nsentence

    def _next_batch(self):
        for k, v in self.rec_id2tokens.items():
            if k in self.ids_done:
                continue

            if len(v) < self.max_len:
                yield [v], [k]
                self.sentence_done += 1
                continue

            buffer = v[: self.max_len]
            batch = []
            rec_ids = []
            for ii in range(self.max_len, len(v)):
                output_seq = buffer[:]

                batch.append(output_seq)
                rec_ids.append(k)

                if len(batch) == self.batch_size:
                    yield batch, rec_ids
                    self.sentence_done += len(batch)
                    batch.clear()
                    rec_ids.clear()
                buffer.pop(0)
                buffer.append(v[ii])

            # Non-complete batch
            if batch:
                yield batch, rec_ids
                self.sentence_done += len(batch)

            self.ids_done.append(k)


def create_logger(log_file_base, verbose: bool):
    """Create logger"""

    now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=f"{log_file_base}_{now_str}",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    if verbose:
        logger.addHandler(logging.StreamHandler())
    return logger


def compute_ppl_per_recording(nlls: list, utt_ids: list) -> Tuple[dict, dict]:
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
    for rec_id in rec_id2nlls:
        ppl = np.exp(np.mean(rec_id2nlls[rec_id]))
        rec_id2ppl[rec_id] = ppl
        ppls.append(ppl)

    logger.info(
        f"Avg. PPL of recordings: {np.mean(ppls):.2f} std.dev: {np.std(ppls):.2f} min PPL: {np.min(ppls):.2f} max PPL: {np.max(ppls):.2f}"
    )
    return rec_id2nlls, rec_id2ppl


if __name__ == "__main__":
    ...
