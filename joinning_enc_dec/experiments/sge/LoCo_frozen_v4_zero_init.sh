#!/bin/bash
#$ -N LoCo-ASR
#$ -q long.q@@gpu
#$ -l ram_free=80G,mem_free=80G
#$ -l matylda5=10
#$ -l gpu=1,gpu_ram=40G
#$ -o /mnt/matylda5/xpolok03/projects/LoCo-ASR/experiments/LoCo_frozen_v4_zero_init.o
#$ -e /mnt/matylda5/xpolok03/projects/LoCo-ASR/experiments/LoCo_frozen_v4_zero_init.e

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
source /mnt/matylda5/xpolok03/miniconda3/bin/activate /mnt/matylda5/xpolok03/envs/loco_asr/

SRC_DIR="/mnt/matylda5/xpolok03/projects/LoCo-ASR"
SCRATCH_DIR="/mnt/matylda5/xpolok03/projects/LoCo-ASR"
DATASET_DIR="${SRC_DIR}/datasets/fisher"
MODEL_CHECKPOINT="/mnt/matylda5/xpolok03/projects/LoCo-ASR/models/XLS-R+GPT2_withCTC"
EXPERIMENT="LoCo_frozen_v4_zero_init"

cd $SRC_DIR

export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/joinning_enc_dec/src"
export $(/mnt/matylda4/kesiraju/bin/gpus 1) || exit 1

export HF_HOME="${SRC_DIR}/huggingface_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="LoCo-ASR_v2"

python joinning_enc_dec/src/trainers/LoCo_v4.py \
  --dataset_name="${DATASET_DIR}" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --base_encoder_model="facebook/wav2vec2-xls-r-300m" \
  --feature_extractor_name="facebook/wav2vec2-xls-r-300m" \
  --base_decoder_model="gpt2" \
  --tokenizer_name="gpt2" \
  --enc_layers_to_freeze="24" \
  --dec_layers_to_freeze="12" \
  --steps_to_freeze_enc="-1" \
  --steps_to_freeze_dec="-1" \
  --output_dir="${SRC_DIR}/experiments/${EXPERIMENT}" \
  --gradient_accumulation_steps="2" \
  --learning_rate="1e-4" \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="1000" \
  --evaluation_strategy="steps" \
  --eval_steps="1000" \
  --auto_find_batch_size="True" \
  --per_device_train_batch_size="16" \
  --per_device_eval_batch_size="16" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="4" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_loss" \
  --early_stopping_patience="10" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs=2 \
  --num_beams="1" \
  --max_len="128" \
  --greater_is_better="False" \
  --train_split="train_500" \
  --validation_split="dev_6" \
  --length_column_name="n_turns" \
  --resume_from_checkpoint=$MODEL_CHECKPOINT \
  --ctc_weight="0.2" \
  --freeze_cross_attention \
  --freeze_others \
  --conv_ids_column_name="recording" \
  --turn_index_column_name="turn_index" \
  --enable_enc_context \
  --reinit_context_weights