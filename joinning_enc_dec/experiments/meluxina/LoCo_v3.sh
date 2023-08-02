#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --time=1-24:00:00                  # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200186                  # project account
#SBATCH --qos=default                      # SLURM qos

EXPERIMENT="LoCo_v3"
PROJECT="LoCo-ASR_v2"
SRC_DIR="/home/users/u100959/projects/LoCo-ASR"
WORK_DIR="/project/home/p200186"
DATASET_DIR="${WORK_DIR}/data/fisher_conv"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"
MODEL_CHECKPOINT="$WORK_DIR/experiments/LoCo-ASR_v2_XLS-R300m+GPT2+CTC_MELUXINA/best_checkpoint"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID=$EXPERIMENT
export HF_HOME="${WORK_DIR}/huggingface_cache"
source /home/users/u100959/miniconda3/bin/activate /project/home/p200186/envs/loco_asr

export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

cd $SRC_DIR

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
   joinning_enc_dec/src/trainers/LoCo_v3.py \
  --dataset_name="${DATASET_DIR}" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --base_encoder_model="facebook/wav2vec2-xls-r-300m" \
  --feature_extractor_name="facebook/wav2vec2-xls-r-300m" \
  --base_decoder_model="gpt2" \
  --tokenizer_name="gpt2" \
  --enc_layers_to_freeze="24" \
  --steps_to_freeze_enc="-1" \
  --dec_layers_to_freeze="12" \
  --steps_to_freeze_dec="-1" \
  --output_dir="${EXPERIMENT_PATH}" \
  --gradient_accumulation_steps="4" \
  --learning_rate="1e-4" \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="1000" \
  --evaluation_strategy="steps" \
  --eval_steps="1000" \
  --per_device_train_batch_size="8" \
  --per_device_eval_batch_size="8" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --auto_find_batch_size="True" \
  --dataloader_num_workers="16" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_loss" \
  --early_stopping_patience="10" \
  --remove_unused_columns="False" \
  --save_total_limit="2" \
  --num_train_epochs=1 \
  --num_beams="1" \
  --max_len="128" \
  --group_by_length="True" \
  --greater_is_better="False" \
  --train_split="train_500" \
  --validation_split="dev_6" \
  --length_column_name="n_turns" \
  --resume_from_checkpoint=$MODEL_CHECKPOINT \
  --freeze_cross_attention \
  --freeze_others \
  --ctc_weight="0.2" \
  --reinit_context_weights \
  --ddp_find_unused_parameters=False