#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --time=1-24:00:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200186                  # project account
#SBATCH --qos=default                      # SLURM qos

EXPERIMENT="AED_80M_label_smoothing_MELUXINA_bigger_lr"
PROJECT="LoCo-ASR_v2"
SRC_DIR="/home/users/u100959/projects/LoCo-ASR"
WORK_DIR="/project/home/p200186"
DATASET_DIR="${WORK_DIR}/data/fisher"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID=$EXPERIMENT
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

source /home/users/u100959/miniconda3/bin/activate /project/home/p200186/envs/loco_asr
cd $SRC_DIR

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  joinning_enc_dec/src/trainers/xlsr+gpt2+ctc.py \
  --dataset_name="${DATASET_DIR}" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --base_encoder_model="Lakoc/fisher_enc_12_layers" \
  --feature_extractor_name="facebook/wav2vec2-xls-r-300m" \
  --base_decoder_model="Lakoc/fisher_dec_6_layers" \
  --tokenizer_name="Lakoc/fisher_bpe" \
  --output_dir=$EXPERIMENT_PATH \
  --gradient_accumulation_steps="1" \
  --learning_rate="5e-4" \
  --warmup_steps="5000"
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="1000" \
  --evaluation_strategy="steps" \
  --eval_steps="1000" \
  --per_device_train_batch_size="24" \
  --per_device_eval_batch_size="24" \
  --group_by_length="True" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="16" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_loss" \
  --early_stopping_patience="5" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs="100" \
  --num_beams="5" \
  --max_len="128" \
  --greater_is_better="False" \
  --train_split="train_500" \
  --validation_split="dev_6" \
  --bf16 \
  --ctc_weight="0.2" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --pad_token="<pad>" \
  --lsm_factor="0.1"