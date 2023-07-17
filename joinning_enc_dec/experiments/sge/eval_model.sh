#!/bin/bash
#$ -N LoCo-ASR
#$ -q all.q@@gpu
#$ -l ram_free=8G,mem_free=8G
#$ -l matylda5=2
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/xpolok03/projects/LoCo-ASR/experiments/eval_pretrained_model.o
#$ -e /mnt/matylda5/xpolok03/projects/LoCo-ASR/experiments/eval_pretrained_model.e

# Job should finish in 5 hours
ulimit -t 18000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/xpolok03/miniconda3/bin/activate /mnt/matylda5/xpolok03/envs/loco_asr/

SRC_DIR="/mnt/matylda5/xpolok03/projects/LoCo-ASR"
DATASET_DIR="${SRC_DIR}/datasets/fisher"

export HF_HOME="${SRC_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/joinning_enc_dec/src"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export $(/mnt/matylda4/kesiraju/bin/gpus) || exit 1

cd $SRC_DIR
torchrun --nproc_per_node=1 joinning_enc_dec/src/evaluation/evaluate_wer_base_seq2seq.py \
  --dataset_name="${DATASET_DIR}" \
  --model="checkpoint-27000" \
  --output_dir="test_xx" \
  --predict_with_generate="True" \
  --with_ctc="True" \
  --validation_split="dev_6" \
  --per_device_eval_batch_size="8" \
  --generation_num_beams="1" \
  --dataloader_num_workers="4"
