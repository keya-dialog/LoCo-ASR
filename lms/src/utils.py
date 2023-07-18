import logging
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Iterator, List, Tuple
from datetime import datetime
from collections import defaultdict
from torch.utils.data import IterableDataset


class Processor:
    def __init__(self, model, dataloader, device, checkpoint):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.checkpoint = checkpoint

        self.nlls = []
        self.ids = []

        self.load_checkpoint()

    def fwd(self):
        # stime = time()
        pbar = tqdm(self.dataloader, initial=self.dataloader.sentence_done)
        # pbar.update(self.dataloader.sentence_done)
        for i, (batch_text_ids, batch_ids, i_record) in enumerate(pbar):
            batch_text_ids = torch.LongTensor(batch_text_ids).to(device=self.device)
            target_ids = batch_text_ids.clone()

            outputs = self.model(batch_text_ids)

            xen_loss = torch.nn.CrossEntropyLoss(reduction='none')

            # ignore the logits for the last token
            shifted_logits = outputs.logits[..., :-1, :].contiguous()

            # ignore the targets for the first token
            shifted_targets = target_ids[..., 1:].contiguous()

            # compute cross entropy loss and return it for every token
            neg_llh = xen_loss(torch.transpose(shifted_logits, 1, 2), shifted_targets)

            self.nlls.extend(neg_llh.cpu().numpy().tolist())
            self.ids.extend(batch_ids)

            pbar.update(batch_text_ids.shape[0])
            pbar.set_postfix({'Recordings': f"{i_record}/{self.dataloader.nrecording}"})

            if i % 10 == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        # print("(save) nlls_score", len(self.nlls))
        # print("(save) ids_score", len(self.ids))
        # print("(save) sentence_processed", self.dataloader.sentence_done)
        torch.save({
            "id_processed": self.dataloader.ids_done,
            "sentence_processed": self.dataloader.sentence_done,
            "nlls_score": self.nlls,
            "ids_score": self.ids,
        }, self.checkpoint)

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint):
            checkpoint = torch.load(self.checkpoint)
            self.dataloader.ids_done = checkpoint["id_processed"]
            self.dataloader.sentence_done = checkpoint["sentence_processed"]
            self.nlls = checkpoint["nlls_score"]
            self.ids = checkpoint["ids_score"]
            # print("(load) nlls_score", len(self.nlls))
            # print("(load) ids_score", len(self.ids))
            # print("(load) sentence_processed", self.dataloader.sentence_done)

class FisherTextDataset(IterableDataset):
    """ Dataset """

    def __init__(self,
                 fname: str, tokenizer, batch_size: int = 128,
                 context_type: str = "indep", max_len: int = 128):
        """ Init """
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
            rec_id2tokens[rec_id].extend(self.tokenizer(text)['input_ids'])
            rec_id2tokens[rec_id].append(self.tokenizer.eos_token_id)

        return utt_id2text, rec_id2tokens

    def __len__(self):
        nsentence = 0
        for _, v in self.rec_id2tokens.items():
            if len(v) < self.max_len:
                nsentence += 1
            else:
                nsentence += (1 + (len(v) - self.max_len))

        return nsentence

    def _next_batch(self):
        for i, (k, v) in enumerate(self.rec_id2tokens.items(), start=1):

            if k in self.ids_done:
                continue

            if len(v) < self.max_len:
                yield [v], [k], i
                self.sentence_done += 1
                continue

            buffer = v[:self.max_len]
            batch = []
            rec_ids = []
            for ii in range(self.max_len, len(v)):
                output_seq = buffer[:]

                batch.append(output_seq)
                rec_ids.append(k)

                if len(batch) == self.batch_size:
                    yield batch, rec_ids, i
                    self.sentence_done += len(batch)
                    batch.clear()
                    rec_ids.clear()
                buffer.pop(0)
                buffer.append(v[ii])

            # Non-complete batch
            if batch:
                yield batch, rec_ids, i
                self.sentence_done += len(batch)

            self.ids_done.append(k)


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
