#!/bin/bash
#$ -N LoCo-ASR
#$ -q long.q@@gpu
#$ -l ram_free=48G,mem_free=48G
#$ -l matylda5=10
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/LoCo_v20_bigger_mem.o
#$ -e /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/LoCo_v20_bigger_mem.e

# Job should finish in 2 days - 172800 seconds
ulimit -t 172800

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

ulimit -u 4096

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/loco_asr/

SRC_DIR="/mnt/matylda5/ipoloka/projects/LoCo-ASR"
SCRATCH_DIR="/mnt/matylda5/ipoloka/projects/LoCo-ASR"
DATASET_DIR="${SRC_DIR}/datasets/fisher"
MODEL_CHECKPOINT="/mnt/matylda5/ipoloka/projects/LoCo-ASR/models/checkpoint-88000"
EXPERIMENT="LoCo_v20_bigger_mem"

cd $SRC_DIR

export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/joinning_enc_dec/src"
export $(/mnt/matylda4/kesiraju/bin/gpus 1) || exit 1

export HF_HOME="${SRC_DIR}/huggingface_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="LoCo-ASR_v20"

python joinning_enc_dec/src/trainers/LoCo.py \
  --dataset_name="${DATASET_DIR}" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --output_dir="${SRC_DIR}/experiments/${EXPERIMENT}" \
  --gradient_accumulation_steps="1" \
  --learning_rate="1e-5" \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="200" \
  --evaluation_strategy="steps" \
  --eval_steps="200" \
  --auto_find_batch_size="True" \
  --per_device_train_batch_size="64" \
  --per_device_eval_batch_size="64" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="4" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_loss" \
  --early_stopping_patience="10" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs=5 \
  --max_len="128" \
  --from_pretrained=$MODEL_CHECKPOINT \
  --conv_ids_column_name="recording" \
  --turn_index_column_name="turn_index" \
  --enc_memory_dim 64 \
  --dec_memory_dim 64 \
  --enc_memory_cells_location 11 \
  --dec_memory_cells_location 5 \
  --train_split="train_500" \
  --validation_split="dev_6" \
  --fp16 \
  --predict_with_generate \
  --generation_num_beams 1 \
  --freeze_others
