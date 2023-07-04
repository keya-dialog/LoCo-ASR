## Recipes for training/fine-tuning and evaluating perplexity on Fisher dev and test

- To prepare Fisher data, see [../../data/fisher/README.md](../../data/fisher/README.md)

### Evaluating PPL on Fisher independent utterances using GPT-2

- Expects the Fisher data (utterance-by-utternace) present in the following dir
`data/fisher_topic_splits/{train,dev,test}.txt`

```bash
model=gpt-2  # gpt2-medium, gpt2-large, gpt2-xl
queue.pl -q all.q --gpu 1 \
  exp/gpt2/fisher_ppl/${model}_fisher_dev.log \
  python src/eval_ppl_with_pretrained_lm.py \
  --in_file data/fisher_topic_splits/dev.txt \
  --ftype text \
  --out_dir exp/gpt2/fisher_ppl/ \
  --verbose \
  --model "${model}"
```

### Evaluating PPL on Fisher conversations, subject to max sequence length of the model

```bash

```