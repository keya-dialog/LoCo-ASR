#!/usr/bin/bash
#SBATCH --job-name LoCo
#SBATCH --account OPEN-28-58
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --time 4:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/decoding/logits_averaging.out

PROJECT="LoCo-ASR_v2"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
DATASET_DIR="${WORK_DIR}/datasets/fisher"
MODEL="LoCo-ASR_v2_ASR_v2_AED_80M_label_smoothing_MELUXINA_mel_fe_augmentations_3_lm_heads/checkpoint-111000"
MODEL_PATH="${WORK_DIR}/experiments/${MODEL}"

export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

ml Anaconda3/2021.05
source activate loco_asr

cd $WORK_DIR

python joinning_enc_dec/src/evaluation/evaluate_wer_base_seq2seq.py \
  --dataset_name="${DATASET_DIR}" \
  --model="${MODEL_PATH}" \
  --output_dir="test_logits_averaging" \
  --predict_with_generate="True" \
  --validation_split="dev_6" \
  --per_device_eval_batch_size="32" \
  --generation_num_beams="5" \
  --dataloader_num_workers="4" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --decoder_average_logits
