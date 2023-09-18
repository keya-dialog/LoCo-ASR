#!/bin/bash
#$ -N LoCo-ASR
#$ -q long.q@@gpu
#$ -l ram_free=8G,mem_free=8G
#$ -l matylda5=2
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/eval_pretrained_model_joint0.o
#$ -e /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/eval_pretrained_model_joint0.e

# Job should finish in 24 hours
ulimit -t 86400

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/loco_asr/

SRC_DIR="/mnt/matylda5/ipoloka/projects/LoCo-ASR"
DATASET_DIR="${SRC_DIR}/datasets/fisher"

export HF_HOME="${SRC_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/joinning_enc_dec/src"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export $(/mnt/matylda4/kesiraju/bin/gpus 1) || exit 1

cd $SRC_DIR

python joinning_enc_dec/src/evaluation/evaluate_wer_base_seq2seq.py \
  --dataset_name="${DATASET_DIR}" \
  --model="models/checkpoint-88000" \
  --output_dir="test_small0" \
  --predict_with_generate="True" \
  --validation_split="dev_6" \
  --per_device_eval_batch_size="8" \
  --generation_num_beams="5" \
  --dataloader_num_workers="4" \
  --with_ctc \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0"
