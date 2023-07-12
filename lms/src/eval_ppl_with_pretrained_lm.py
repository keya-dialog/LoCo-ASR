#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju, Jan Svec
# e-mail : kcraj2[AT]gmail[DOT]com, kesiraju[AT]fit[DOT]vutbr[DOT]cz
#        : john.swabe[AT]gmail[DOT]com, isvecjan[AT]fit[DOT]vutbr[DOT]cz
# Date created : 29 Jun 2023
# Last modified : 29 Jun 2023

"""
"""

import os
import sys
import json
import torch
import pickle
import argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from pathlib import Path
from time import time
from utils import (
    FisherTextDatasetIndep, FisherTextDatasetMaxLen,
    create_logger,
    compute_ppl_per_recording)

from utils import (
    load_key_text_file,
    create_logger,
    compute_ppl_per_recording,
)

def main():
    """main method"""

    args = parse_arguments()

    if args.download_only:
        model_id = args.model
        _ = GPT2LMHeadModel.from_pretrained(model_id)
        _ = GPT2TokenizerFast.from_pretrained(model_id)
        sys.exit(0)

    # if args.offline:
    #     os.environ["HF_HUB_OFFLINE"] = "1"
    #     os.environ["HF_DATASETS_OFFLINE"] = "1"
    #     os.environ["TRANSFORMERS_OFFLINE"] = "1"
    #     os.environ["HF_EVALUATE_OFFLINE"] = "1"

    os.makedirs(args.out_dir, exist_ok=True)

    path_out_dir = Path(args.out_dir)
    device = "cuda" if args.cuda else "cpu"

    # prepare common file prefix based on input args
    model_id = args.model
    base = os.path.basename(args.in_file).rsplit(".", 1)[0]
    context = args.context_type
    pfx = f"{model_id}_{context}_{base}"

    # Create logger
    logger = create_logger(path_out_dir / f"{pfx}.log", args.verbose)

    # Get GPU (BUT style)
    if args.cuda and args.but_gpu:
        from safe_gpu import safe_gpu
        safe_gpu.claim_gpus(logger=logger)

    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    if args.context_type == "indep":
        dataset = FisherTextDatasetIndep(args.in_file, tokenizer, batch_size=args.bsize)
        utt_ids = dataset.utt_ids
    elif args.context_type == "max_len":
        dataset = FisherTextDatasetMaxLen(args.in_file, tokenizer, max_len=model.config.n_positions, batch_size=args.bsize)
    else:
        raise ValueError(f"Wrong set 'context_type' as {args.context_type}")

    nlls = []
    stime = time()
    with torch.no_grad():
        if args.context_type == "indep":
            for batch_text_ids in dataset:
                batch_text_ids = torch.LongTensor(batch_text_ids).to(device=device)
                target_ids = batch_text_ids.clone()

                outputs = model(batch_text_ids)

                xen_loss = torch.nn.CrossEntropyLoss(reduction='none')

                # ignore the logits for the last token
                shifted_logits = outputs.logits[..., :-1, :].contiguous()

                # ignore the targets for the first token
                shifted_targets = target_ids[..., 1:].contiguous()

                # compute cross entropy loss and return it for every token
                neg_llh = xen_loss(torch.transpose(shifted_logits, 1, 2), shifted_targets)

                # append to the list of nlls
                nlls.extend(neg_llh.cpu().numpy().tolist())
        elif args.context_type == "max_len":
            i_recording = 0
            rec_ids = []
            nsamples = 0
            act_rec = None
            for i, (batch_text_ids, rec_id, first_batch, last_batch) in enumerate(dataset):
                print(f"\r Batch: {i} , Actual rec: {act_rec} ({i_recording}/{dataset.nrecording}), Procesed sentences: {nsamples} / {dataset.nsentence}", end=" ", file=sys.stderr)
                nsamples += len(batch_text_ids)

                batch_text_ids = torch.LongTensor(batch_text_ids).to(device=device)
                target_ids = batch_text_ids.clone()

                outputs = model(batch_text_ids)

                xen_loss = torch.nn.CrossEntropyLoss(reduction='none')
                if first_batch:
                    xen_loss = torch.nn.CrossEntropyLoss(reduction='none')
                    print()
                    act_rec = rec_id[0]
                    i_recording += 1

                    # ignore the logits for the last token
                    shifted_logits = outputs.logits[..., :-1, :].contiguous()

                    # ignore the targets for the first token
                    shifted_targets = target_ids[..., 1:].contiguous()

                    # compute cross entropy loss and return it for every token
                    neg_llh = xen_loss(torch.transpose(shifted_logits, 1, 2), shifted_targets)

                    # append to the list of nlls
                    nlls.extend(neg_llh.cpu().numpy().tolist())
                    rec_ids.extend(rec_id)
                else:
                    xen_loss = torch.nn.CrossEntropyLoss(reduction='none')
                    # ignore the logits for the last token
                    shifted_logits = outputs.logits[..., :-1, :].contiguous()

                    # ignore the targets for the first token
                    shifted_targets = target_ids[..., 1:].contiguous()

                    # compute cross entropy loss and return it for every token
                    neg_llh = xen_loss(torch.transpose(shifted_logits, 1, 2), shifted_targets)

                    # append to the list of nlls
                    nlls.extend([[_v] for _v in neg_llh[:,-1].cpu().numpy().tolist()])
                    rec_ids.extend(rec_id)
        else:
            raise NotImplementedError
        print()

    if args.context_type == "indep":
        ids = utt_ids
    elif args.context_type == "max_len":
        ids = rec_ids

    assert len(nlls) == len(ids), f"nlls {len(nlls)} != utt_ids {len(ids)}"

    rec_id2nlls, rec_id2ppl = compute_ppl_per_recording(nlls, ids)

    with open(path_out_dir / "rec_id2nlls.pkl", "wb") as fpw:
        pickle.dump(rec_id2nlls, fpw)

    with open(path_out_dir / "rec_id2ppl.json", "w", encoding="utf-8") as fpw:
        json.dump(rec_id2ppl, fpw, indent=2, ensure_ascii=False)

    logger.info(f"Saved in {args.out_dir} Time taken {time() - stime:.2f} sec")

    return 0

def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--in_file", "-in_file", "-i",
        required=True,
        help="path to input text file on which PPL shall be computed",
    )
    # parser.add_argument(
    #     "--ftype",
    #     choices=['text', 'key'],
    #     required=True,
    #     help="""input file type. text is plain text line by line,
    #     key is where first column in utterance with time_stamp (kaldi format)
    #     which will help in figuring out the chronology of utterances in
    #     in a recording""",
    # )
    parser.add_argument(
        "--out_dir", "--out_dir", "-o",
        required=True,
        help="path to out dir where results are stored",
    )
    parser.add_argument(
        "--bsize", "--batch_size", "-bsize", "-batch_size", "--sb", "-sb",
        type=int,
        default=128,
        help="max batch size"
    )
    parser.add_argument(
        "--model", "-model", "-m",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="huggingface model",
    )
    parser.add_argument(
        "--context_type", "-context_type", "--ct", "-ct",
        choices=["indep", "max_len"],
        default="indep",
        type=str,
        help="""How much context to use? indep is independent utterances, max_len is
        max length constrained by the pretrained model which could be 512 or 1024 tokens""",
    )
    parser.add_argument("--bsize", type=int, default=128, help="max batch size")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="do not use cuda device, by default uses cuda",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not use internet(connection), by default yes",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Download only model from HuggingFace",
    )
    parser.add_argument(
        "--but_gpu", "-but_gpu",
        action="store_true",
        help="Use BUT's procedure for access to GPU",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="print logging info on stdout",
    )
    args = parser.parse_args()

    args.cuda = not args.no_cuda

    # if args.context_type == "max_len" and args.ftype != "key":
    #     print("Input ftype must be 'key' when using 'max_len' context")
    #     sys.exit()

    return args


if __name__ == "__main__":
    sys.exit(main())
