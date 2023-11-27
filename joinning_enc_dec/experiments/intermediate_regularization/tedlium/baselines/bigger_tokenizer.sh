#!/usr/bin/bash
#SBATCH --job-name TED_LM
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu
#SBATCH --time 01:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/ted_tokenizer.out

EXPERIMENT="gpt2_tedlium88"
PROJECT="TED_CLM"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"
USER="lakoc"
TOKENIZER_NAME="ted_uni1000"
LM_DATA="${WORK_DIR}/data/ted_lm_raw.txt"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

conda deactivate
source activate loco_asr

cd $WORK_DIR

python joinning_enc_dec/src/trainers/train_tokenizer.py \
  --dataset_name="LIUM/tedlium" \
  --dataset_config="release3" \
  --tokenizer_name=$TOKENIZER_NAME \
  --vocab_size=1000 \
  --tokenizer_type="unigram" \
  --text_column_name="text" \
  --train_split="train" \
  --additional_raw_data $LM_DATA \
  --skip_if_exists="${USER}/${TOKENIZER_NAME}"
