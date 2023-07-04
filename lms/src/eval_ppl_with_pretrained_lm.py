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
import argparse
import numpy as np
import torch
from tqdm import tqdm
from utils import load_text, create_logger
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def prepare_indep_utt_dataset(texts, tokenizer):
    """Prepare dataset that treats utterances independently"""

    logger = logging.getLogger()

    bsize = 32
    start = 0
    end = start + bsize
    input_ids = []
    # input_lens = []
    target_ids = []

    ign = 0

    while start < len(texts):

        if end > len(texts):
            end = len(texts)

        encodings = tokenizer(texts[start:end])

        for inp in encodings.input_ids:
            if len(inp) == 1:
                ign += 1
                continue
            input_ids.append(torch.LongTensor(inp))
            target_ids.append(torch.LongTensor(inp))
            # input_lens.append(len(inp))

        start = end
        end += bsize

    if ign > 0:
        logger.info("Ignoring single word utterances %d", ign)

    return input_ids, target_ids


def prepare_max_seq_len_dataset(texts: list, tokenizer, max_seq_len):
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
    logger.info("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES"))

    # out file to save negative LLHs for each input - can be used later to compute PPL
    out_nll_file = os.path.join(args.out_dir, f"{pfx}_nlls.txt")

    # Load the input text line-by-line
    texts = load_text(args.in_file)
    logger.info("In text lines in %s: %d", args.in_file, len(texts))

    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    # Tokenize the input text and convert to torch Tensors
    if context == "indep":
        logger.info(
            "Treating each sentence independently. Sentences with 1 word will be ignored."
        )
        all_input_ids, all_target_ids = prepare_indep_utt_dataset(texts, tokenizer)

    elif context == "max_len":
        max_len = model.config.n_positions
        logger.info("Max seq length of the model is %d", max_len)
        all_input_ids, all_target_ids = prepare_max_seq_len_dataset(texts, tokenizer)
    else:
        print("Context type", context, "not implemented", file=sys.stderr)
        sys.exit()

    # forward and get negative llh for each input sequence
    nlls = []
    for i in tqdm(range(len(all_input_ids))):

        input_ids = all_input_ids[i].to(device=device)
        target_ids = all_target_ids[i].to(device=device)

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over
            # valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels,
            # because it internally shifts the labels to the left by 1.
            neg_llh = outputs.loss

        nlls.append(neg_llh)

    ppl = torch.exp(torch.stack(nlls).mean())
    np.savetxt(out_nll_file, torch.stack(nlls).cpu().numpy())
    logger.info("PPL: %.2f", ppl.cpu().item())


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
    parser.add_argument("--ftype", choices=['text', 'key'], required=True,
                        help="""input file type. text is plain text line by line,
                        key is where first column in utterance with time_stamp (kaldi format)
                        which will help in figuring out the chronology of utterances in
                        in a recording"""
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
        help="""How much context to use? indep is independent utterances, max_len is
                        max length constrained by the pretrained model which could be 512 or 1024 tokens""",
    )
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
