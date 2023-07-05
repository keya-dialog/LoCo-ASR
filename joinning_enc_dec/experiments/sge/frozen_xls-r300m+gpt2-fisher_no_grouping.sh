#!/bin/bash
#$ -N XLSR_DeePsy
#$ -q long.q@@gpu
#$ -l ram_free=8G,mem_free=8G
#$ -l matylda5=10
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/xpolok03/projects/LoCo-ASR/experiments/xlsr+gpt2_no_grouping.o
#$ -e /mnt/matylda5/xpolok03/projects/LoCo-ASR/experiments/xlsr+gpt2_no_grouping.e

# Job should finish in 10 days - 432000 seconds
ulimit -t 432000

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
SCRATCH_DIR="/mnt/matylda5/xpolok03/projects/LoCo-ASR"
DATASET_DIR="${SRC_DIR}/datasets/fisher"
EXPERIMENT="XLS-R300m+cold_decoder_no_grouping"

cd $SRC_DIR
export HF_HOME="${SRC_DIR}/huggingface_cache"

WANDB_RUN_ID=$EXPERIMENT WANDB_PROJECT="LoCo-ASR" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline python joinning_enc_dec/src/trainers/xlsr+gpt2.py \
  --dataset_name="${DATASET_DIR}" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --base_encoder_model="facebook/wav2vec2-xls-r-300m" \
  --feature_extractor_name="facebook/wav2vec2-xls-r-300m" \
  --base_decoder_model="gpt2" \
  --tokenizer_name="gpt2" \
  --enc_layers_to_freeze="24" \
  --steps_to_freeze_enc="-1" \
  --output_dir="${SRC_DIR}/experiments/${EXPERIMENT}" \
  --gradient_accumulation_steps="8" \
  --learning_rate="1e-5" \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="10000" \
  --evaluation_strategy="steps" \
  --eval_steps="10000" \
  --auto_find_batch_size="True" \
  --per_device_train_batch_size="8" \
  --per_device_eval_batch_size="8" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="4" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_wer" \
  --early_stopping_patience="5" \
  --remove_unused_columns="False" \
  --predict_with_generate="True" \
  --save_total_limit="5" \
  --num_train_epochs=10 \
  --num_beams="1" \
  --max_len="512" \
  --greater_is_better="False" \
  --train_split="train_500" \
  --validation_split="dev_6" \
  --n_gpus="1"
