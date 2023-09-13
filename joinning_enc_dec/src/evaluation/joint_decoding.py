from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from safe_gpu import safe_gpu
from torch.utils.data import DataLoader
from transformers import (AutoFeatureExtractor, AutoTokenizer, BeamSearchScorer, HfArgumentParser, LogitsProcessor,
                          LogitsProcessorList)
from transformers.trainer_pt_utils import find_batch_size

from per_utterance.models import JointCTCAttentionEncoderDecoder
from trainers.AED_from_enc_dec import DataTrainingArguments, ModelArguments
from utils import Seq2SeqDataCollatorWithPadding


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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
    model = JointCTCAttentionEncoderDecoder.from_pretrained(
        model_args.from_pretrained)
    model.to(device)

    dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)

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

    for batch in iter(dataloader):
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
            "max_length": gen_args.max_len,
        }

        input_ids, model_kwargs = model._expand_inputs_for_generation(expand_size=gen_args.num_beams,
                                                                      is_encoder_decoder=True,
                                                                      input_ids=input_ids,
                                                                      **model_kwargs)

        beam_scorer = BeamSearchScorer(
            batch_size=gen_args.batch_size,
            num_beams=gen_args.num_beams,
            device=device,
            do_early_stopping=True
        )
        outputs = model.joint_beam_search(input_ids, beam_scorer,
                                          logits_processor=logits_processor,
                                          **model_kwargs)
        labels_batch = batch['labels']
        labels_batch[labels_batch == -100] = tokenizer.pad_token_id

        print(f"Reference: {tokenizer.batch_decode(labels_batch.tolist(), skip_special_tokens=True)}\n"
              f"Hypothesis: {tokenizer.batch_decode(outputs, skip_special_tokens=True)}")
        break
