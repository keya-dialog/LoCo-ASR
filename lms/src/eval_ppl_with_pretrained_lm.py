#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
# Date created : 29 Jun 2023
# Last modified : 29 Jun 2023

"""
"""

import os
import sys
import logging
import pickle
import json
import argparse
from functools import partial
from time import time
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import (
    load_key_text_file,
    create_logger,
    compute_ppl_per_recording,
)


def yield_indep_sent_batches(text_ids, lengths, bsize):
    """Return batches of independent sentences"""

    logger = logging.getLogger()
    # smart batching:
    # create batches based on sentence lengths
    diff_ixs = np.diff(lengths)
    bnd_ixs = np.where(diff_ixs > 0)[0]

    j = 0
    start = 0
    end = start + bsize
    total = 0
    bno = 1  # batch number

    while start < len(text_ids):
        logger.info(
            """batch {bno:5d} length bin {lengths[start]:3d}
start ix {start:8d} : {end:8d} / {len(text_ids):8d}"""
        )
        batch_text_ids = text_ids[start:end]

        total += len(batch_text_ids)
        bno += 1

        start = end
        end += bsize

        if j < len(bnd_ixs):
            if end >= bnd_ixs[j] + 1:
                end = bnd_ixs[j] + 1
                j += 1
        else:
            if end > len(text_ids):
                end = len(text_ids)

        yield batch_text_ids


def yield_max_input_len_batches(text_ids, lengths, bsize):
    """Prepare input seqs and targets based on max seq length of the model"""

    pass


def main():
    """main method"""

    args = parse_arguments()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if args.cuda else "cpu"

    # prepare common file prefix based on input args

    model_id = args.model
    base = os.path.basename(args.in_file).rsplit(".", 1)[0]
    context = args.context_type
    pfx = f"{model_id}_{context}_{base}"

    # Create logger
    logger = create_logger(os.path.join(args.out_dir, f"{pfx}"), args.verbose)
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "NONE")
    )

    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    # Load the input text line-by-line
    # Tokenize the input text and convert to torch Tensors
    sorted_text_ids, sorted_lengths, sorted_utt_ids = load_key_text_file(
        args.in_file, tokenizer, ignore_one_tok=True
    )

    logger.info("In text lines in %s: %d", args.in_file, len(sorted_lengths))

    if context == "indep":
        logger.info(
            "Treating each sentence independently. Sentences with 1 word will be ignored."
        )
        batcher = partial(
            yield_indep_sent_batches, sorted_text_ids, sorted_lengths, args.bsize
        )

    elif context == "max_len":
        max_len = model.config.n_positions
        logger.info("Max seq length of the model is %d", max_len)

        batcher = partial(
            yield_max_input_len_batches, sorted_text_ids, sorted_lengths, args.bsize
        )

    else:
        print("Context type", context, "not implemented", file=sys.stderr)
        sys.exit()

    stime = time()
    # forward and get negative llh for each input sequence
    nlls = []
    for input_ids in batcher():
        # input ids shape: B x T  (batch, num_tokens)
        input_ids = torch.LongTensor(input_ids).to(device=device)

        # target ids shape: B X T (batch, num_tokens) - same as input
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids)

            xen_loss = torch.nn.CrossEntropyLoss(reduction="none")

            # ignore the logits for the last token
            shifted_logits = outputs.logits[..., :-1, :].contiguous()

            # ignore the targets for the first token
            shifted_targets = target_ids[..., 1:].contiguous()

            # compute cross entropy loss and return it for every token
            neg_llh = xen_loss(torch.transpose(shifted_logits, 1, 2), shifted_targets)

        # extend the list of nlls
        nlls.extend(neg_llh.cpu().numpy().tolist())

    compute_ppl_per_recording(nlls, sorted_utt_ids)

    rec_id2nlls, rec_id2ppl = compute_ppl_per_recording(nlls, sorted_utt_ids)

    with open(os.path.join(args.out_dir, f"{pfx}_rec_id2nlls.pkl"), "wb") as fpw:
        pickle.dump(rec_id2nlls, fpw)

    with open(
        os.path.join(args.out_dir, f"{pfx}_rec_id2ppl.json"), "w", encoding="utf-8"
    ) as fpw:
        json.dump(rec_id2ppl, fpw, indent=2, ensure_ascii=False)

    print(f"Saved in {args.out_dir}. Time taken {(time()-stime):.2f} sec")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--in_file",
        required=True,
        help="path to input text file on which PPL shall be computed",
    )
    parser.add_argument(
        "--ftype",
        choices=["key"],
        required=True,
        help="""input file type. text is plain text line by line,
                        key is where first column in utterance with time_stamp (kaldi format)
                        which will help in figuring out the chronology of utterances in
                        in a recording""",
    )
    parser.add_argument(
        "--out_dir", required=True, help="path to out dir where results are stored"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="huggingface model",
    )
    parser.add_argument(
        "--context_type",
        choices=["indep", "max_len"],
        default="indep",
        type=str,
        help="""How much context to use? indep is independent utterances,
max_len is max length constrained by the pretrained model which could be
512 or 1024 tokens""",
    )
    parser.add_argument("--bsize", type=int, default=128, help="max batch size")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="do not use cuda device, by default uses cuda",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print logging info on stdout"
    )
    args = parser.parse_args()

    args.cuda = not args.no_cuda

    if args.context_type == "max_len":
        if args.ftype != "key":
            print("Input ftype must be 'key' when using 'max_len' context")
            sys.exit()

    return args


if __name__ == "__main__":
    main()
