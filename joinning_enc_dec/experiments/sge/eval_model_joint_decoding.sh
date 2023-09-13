#!/bin/bash
#$ -N LoCo-ASR
#$ -q long.q@@gpu
#$ -l ram_free=8G,mem_free=8G
#$ -l matylda5=2
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/eval_pretrained_model_joint.o
#$ -e /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/eval_pretrained_model_joint.e

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

python joinning_enc_dec/src/evaluation/joint_decoding.py \
  --dataset_name="${DATASET_DIR}" \
  --from_pretrained="models/checkpoint-88000" \
  --ctc_weight="0.3" \
  --batch_size="2" \
  --dataloader_num_workers="1" \
  --num_beams="3" \
  --use_cuda="true"
  --out_path="predictions_joint_decoding_0.3"