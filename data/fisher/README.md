## FISHER CONVERSATIONAL SPEECH

- The data preparation script assumes you have access to the original Fisher Phase 1 and 2
corpora in its original strucure.
- The scripts here are taken from Kaldi, and should work without downloading or installing Kaldi.

### Requirements
- sph2pipe


### Usage
- This will create `data/train_all/` directory in the current working directory, with Kaldi-formatted
file structure

```bash
scripts/fisher_data_prep.sh <PATH_TO_FISHER/>
```

- This will take the recording id splits from `fisher_topic_split/` and creates the Kaldi-formatted
file structure in the target directory `data_topic_splits/`

```python3
scripts/split_fisher_data_based_on_ids.py fisher_topic_split/ data/train_all/ data_topic_splits/ --sets train dev test train_500 dev_6
```

- `dev_6` is intended for monitoring the validation loss/wer during training progess. Contains one recording per topic.
- `train_500` is a smaller training covering all the topics in `train`, and is intended to train and test different architectures rather quickly.