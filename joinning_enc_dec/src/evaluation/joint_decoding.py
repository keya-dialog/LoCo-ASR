from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import tqdm
from datasets import load_from_disk
from jiwer import cer, compute_measures
from safe_gpu import safe_gpu
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer, BeamSearchScorer,
                          HfArgumentParser, LogitsProcessor, LogitsProcessorList, MaxLengthCriteria,
                          StoppingCriteriaList)
from transformers.trainer_pt_utils import find_batch_size

from per_utterance.models import JointCTCAttentionEncoderDecoder, JointCTCAttentionEncoderDecoderConfig
from trainers.AED_from_enc_dec import DataTrainingArguments, ModelArguments
from utils import Seq2SeqDataCollatorWithPadding, filter_out_sequence_from_dataset

AutoConfig.register("joint_aed_ctc_speech-encoder-decoder", JointCTCAttentionEncoderDecoderConfig)
AutoModelForSpeechSeq2Seq.register(JointCTCAttentionEncoderDecoderConfig, JointCTCAttentionEncoderDecoder)


class EnforceEosIfCTCStops(LogitsProcessor):
    """Logits processor (to use with HuggingFace `generate()` method :
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/
    text_generation#transformers.generation_utils.GenerationMixin).

    This logit processor simply ensure that after hitting logzero likelihood for all tokens eos is generated.

    Args:
        eos_token_id (int): ID of the EOS token.
        log_thr (float): Value to use for logzero.
    """

    def __init__(self, eos_token_id: int, log_thr: float = -10000000000.0):
        super().__init__()
        self.log_thr = log_thr
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        should_enforce_stop = scores.max(dim=1).values <= self.log_thr
        mask = should_enforce_stop.unsqueeze(dim=-1).expand(scores.size())
        eos_mask = torch.zeros_like(mask, dtype=torch.bool)
        eos_mask[:, self.eos_token_id] = True
        mask = mask & eos_mask
        scores = torch.where(~mask, scores, self.log_thr / 2)
        return scores


@dataclass
class GenerationArguments:
    num_beams: Optional[int] = field(
        default=2, metadata={"help": "Num beams for decoding."}
    )
    max_len: Optional[int] = field(
        default=200, metadata={"help": "Max number of generated tokens."}
    )
    max_len_factor: Optional[float] = field(
        default=1.5, metadata={"help": "Factor of max tokens to number of encoder frames."}
    )
    ctc_margin: Optional[int] = field(
        default=0, metadata={"help": "Margin to stop generation."}
    )
    ctc_weight: Optional[float] = field(
        default=0, metadata={"help": "CTC weight to bias hypothesis."}
    )
    ctc_beam_width: Optional[int] = field(
        default=None, metadata={"help": "Width of the CTC beam."}
    )
    batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size."}
    )
    dataloader_num_workers: Optional[int] = field(
        default=1, metadata={"help": "Number of workers for dataloader."}
    )
    use_cuda: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gpu for decoding."}
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=0, metadata={"help": "No repeat ngram size."}
    )
    out_path: Optional[str] = field(
        default="predictions.csv", metadata={"help": "Path to save output."}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GenerationArguments))

    model_args, data_args, gen_args = parser.parse_args_into_dataclasses()

    if gen_args.use_cuda:
        safe_gpu.claim_gpus(1)
        if not torch.cuda.is_available():
            raise Exception("No cuda device available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.from_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(model_args.from_pretrained)

    data_collator = Seq2SeqDataCollatorWithPadding(feature_extractor=feature_extractor,
                                                   tokenizer=tokenizer,
                                                   padding=True, sampling_rate=model_args.sampling_rate)

    # 3. Initialize seq2seq model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_args.from_pretrained).to(device)

    # dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)
    dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)
    for split in [data_args.train_split, data_args.validation_split, data_args.test_split]:
        dataset[split] = filter_out_sequence_from_dataset(dataset[split],
                                                          max_input_len=data_args.max_duration_in_seconds,
                                                          min_input_len=data_args.min_duration_in_seconds)

    dataloader = DataLoader(
        dataset[data_args.test_split],
        batch_size=gen_args.batch_size,
        collate_fn=data_collator,
        num_workers=gen_args.dataloader_num_workers,
    )

    encoder = model.get_encoder()
    logits_processor = LogitsProcessorList(
        [EnforceEosIfCTCStops(tokenizer.eos_token_id,
                              log_thr=-10000000000.0 * gen_args.ctc_weight if gen_args.ctc_weight > 0 else -10000000000.0)])

    ref = []
    hyp = []
    for batch in tqdm.tqdm(iter(dataloader)):
        batch = batch.to(device)
        input_ids = torch.ones((find_batch_size(batch), 1), device=device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {
            "encoder_outputs": encoder(
                **batch, return_dict=True, output_hidden_states=True),
            "logit_lens": encoder._get_feat_extract_output_lengths(
                batch['attention_mask'].sum(-1)).to(
                torch.long),
            "output_attentions": gen_args.ctc_margin > 0,
            "margin": gen_args.ctc_margin,
            "ctc_beam_width": gen_args.ctc_beam_width or len(tokenizer),
            "ctc_weight": gen_args.ctc_weight,
        }
        actual_batch_size = model_kwargs["logit_lens"].size(0)

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=min(gen_args.max_len,
                                                                                   int(model_kwargs[
                                                                                           "logit_lens"].max() * gen_args.max_len_factor)))])

        input_ids, model_kwargs = model._expand_inputs_for_generation(expand_size=gen_args.num_beams,
                                                                      is_encoder_decoder=True,
                                                                      input_ids=input_ids,
                                                                      **model_kwargs)

        beam_scorer = BeamSearchScorer(
            batch_size=actual_batch_size,
            num_beams=gen_args.num_beams,
            device=device,
            do_early_stopping=True
        )
        with torch.no_grad():
            search_alg = model.joint_beam_search if gen_args.ctc_weight > 0 else model.beam_search
            outputs = search_alg(input_ids, beam_scorer,
                                 logits_processor=logits_processor,
                                 stopping_criteria=stopping_criteria,
                                 **model_kwargs)
        labels_batch = batch['labels']
        labels_batch[labels_batch == -100] = tokenizer.pad_token_id
        ref.extend(tokenizer.batch_decode(labels_batch.tolist(), skip_special_tokens=True))
        hyp.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    metrics = compute_measures(ref, hyp)
    del metrics['ops']
    del metrics['truth']
    del metrics['hypothesis']

    metrics = {"cer": cer(ref, hyp), **metrics}
    print(metrics)
    pd.DataFrame({"ref": ref, "hyp": hyp}).to_csv(gen_args.out_path, index=False)
