#!/usr/bin/bash
#SBATCH --job-name CommonVoice
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu
#SBATCH --time 24:00:00

EXPERIMENT="librispeech_asr"
PROJECT="CommonVoice"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"
USER="lakoc"
TOKENIZER_NAME="ted_uni500"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID=$EXPERIMENT
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

conda deactivate
source activate loco_asr

cd $WORK_DIR
  
python \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --dataset_name="librispeech_asr" \
  --dataset_config="" \
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k" \
  --tokenizer_name="${USER}/${TOKENIZER_NAME}" \
  --output_dir=$EXPERIMENT_PATH \
  --report_to="none" \
  --length_column_name="input_len" \
  --text_column_name="text" \
  --preprocessing_num_workers="128" \
  --fix_apostrophes \
  --train_split train \
  --preprocess_dataset_only
