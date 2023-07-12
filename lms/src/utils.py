import logging
import sys
import numpy as np
from typing import List, Tuple
from datetime import datetime
from collections import defaultdict
from torch.utils.data import IterableDataset

class FisherTextDatasetIndep(IterableDataset):
    """ Dataset """

    def __init__(self, fname: str, tokenizer, batch_size: int = 128):
        """ Init """

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_ids, self.lengths, self.utt_ids = self._load_key_text_file(fname)
        self.bins, self.counts = np.unique(self.lengths, return_counts=True)

    def __iter__(self):
        return iter(self._next_batch())

    def _next_batch(self):
        """ Generate batches """

        bno = 1  # batch number
        offset = 0
        for bin, count in zip(self.bins, self.counts):
            for i in range(offset, offset+count, self.batch_size):
                start = i
                end = i+self.batch_size
                if i+self.batch_size > offset+count:
                    end = offset+count
                print(f"\r batch {bno:5d} length bin {bin:3d} start ix {start:8d} : {end:8d} / {len(self.text_ids):8d}", end=" ", file=sys.stderr)
                yield self.text_ids[start:end]
                bno += 1
            offset += count
            print()

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
                    print("Duplicate utt id:", utt_id, utt_id2text[utt_id], "ignoring", file=sys.stderr)

                else:
                    tok_ids = self.tokenizer(text)['input_ids']
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

    def print_recid2tokens(self):
        recid2tokens = defaultdict(list)
        recid2tokens = defaultdict(int)
        for text_ids, utt_id in zip(self.text_ids, self.utt_ids):
            rec_id = utt_id.split("-", maxsplit=1)[0]
            recid2tokens[rec_id] += text_ids

        return recid2tokens


class FisherTextDatasetMaxLen(IterableDataset):
    """ Dataset """

    def __init__(self, fname: str, tokenizer, max_len: int = 128, batch_size: int = 5):
        """ Init """
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.rec_id2text, self.nsentence = self._load_key_text_as_rec(fname)
        self.nrecording = len(self.rec_id2text)

    def __iter__(self):
        return iter(self._next_batch())

    def _load_key_text_as_rec(self, fname):

        def recid_time(x):
            rec, _, start, end = x[0].split("-")
            return "-".join((rec, start, end))

        utt_ids = {}
        rec_id2text = defaultdict(list)

        with open(fname, "r", encoding="utf-8") as fd:
            for line in fd:
                utt_id, text = line.strip().split(None, maxsplit=1)
                if utt_id in utt_ids:
                    print(f"Duplicate utt id: {utt_id} ignoring", file=sys.stderr)
                else:
                    utt_ids[utt_id] = text

        utt_ids_sorted = sorted(utt_ids.items(), key=lambda x: recid_time(x))

        for utt_id, text in utt_ids_sorted:
            rec_id = utt_id.split("-", maxsplit=1)[0]
            rec_id2text[rec_id].extend(self.tokenizer(text)['input_ids'])
            rec_id2text[rec_id].append(self.tokenizer.eos_token_id)

        nsentence = 0
        for _, v in rec_id2text.items():
            if len(v) < self.max_len:
                nsentence += 1
            else:
                nsentence += (1 + (len(v) - self.max_len))

        return rec_id2text, nsentence

    def _next_batch(self):
        for k, v in self.rec_id2text.items():
            first_batch = True
            last_batch = False
            if len(v) < self.max_len:
                last_batch = True
                yield [v], [k], first_batch, last_batch
                continue

            buffer = v[:self.max_len]
            batch = []
            rec_ids = []
            for i, ii in enumerate(range(self.max_len, len(v))):
                output_seq = buffer[:]

                if first_batch and i != 0:
                    first_batch = False

                if ii == len(v) - 1:
                    last_batch = True

                batch.append(output_seq)
                rec_ids.append(k)

                if first_batch:
                    yield batch, rec_ids, first_batch, last_batch
                    batch.clear()
                    rec_ids.clear()
                elif len(batch) == self.batch_size:
                    yield batch, rec_ids, first_batch, last_batch
                    batch.clear()
                    rec_ids.clear()
                elif last_batch:
                    yield batch, rec_ids, first_batch, last_batch
                    batch.clear()
                    rec_ids.clear()
                buffer.pop(0)
                buffer.append(v[ii])

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
