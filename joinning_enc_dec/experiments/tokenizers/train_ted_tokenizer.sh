#!/usr/bin/bash
#SBATCH --job-name LoCo
#SBATCH --account OPEN-28-58
#SBATCH --partition qcpu
#SBATCH --time 1:00:00

WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"

export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"

ml Anaconda3/2021.05
source activate loco_asr

cd $WORK_DIR
python joinning_enc_dec/src/trainers/train_tokenizer.py --dataset_name="LIUM/tedlium" --dataset_config="release3" --model_name="Lakoc/ted_tokenizer"
