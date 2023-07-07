

## Recipes for training/fine-tuning and evaluating perplexity on Fisher dev and test

- To prepare Fisher data, see [../../data/fisher/README.md](../../data/fisher/README.md)

### Evaluating PPL on Fisher independent utterances using GPT-2

- Results are [here](https://docs.google.com/spreadsheets/d/1Zv0dwYPTRTECHzfk1j6mpBnJskTPKYGjWYgo_-cbelw/edit?usp=sharing)

- Expects the Fisher data (utterance-by-utternace) present in the following dir in Kaldi-format (utt text)
`data/fisher_topic_splits/{train,dev,test}/text`

```bash
model=gpt-2  # gpt2-medium, gpt2-large, gpt2-xl
queue.pl -q all.q --gpu 1 \
  exp/gpt2/fisher_ppl/${model}_fisher_dev.log \
  python src/eval_ppl_with_pretrained_lm.py \
  --in_file data/fisher_topic_splits/dev.txt \
  --ftype key \
  --out_dir exp/gpt2/fisher_ppl/ \
  --verbose \
  --model "${model}"
```

### Evaluating PPL on Fisher conversations, subject to max sequence length of the model

```bash

```