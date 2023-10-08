#!/usr/bin/bash
#SBATCH --job-name LoCo
#SBATCH --account OPEN-28-58
#SBATCH --partition qgpu
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/LoCo-ASR_v2-AED_ebranchformer_68M_label_smoothing_MELUXINA_mel_fe_augmentations_lm_head3.out

EXPERIMENT="AED_ebranchformer_68M_label_smoothing_MELUXINA_mel_fe_augmentations_lm_head3"
PROJECT="LoCo-ASR_v2"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
DATASET_DIR="${WORK_DIR}/datasets/fisher"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}_reset"
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

ml Anaconda3/2021.05
source activate loco_asr

cd $WORK_DIR

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --dataset_name="${DATASET_DIR}" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --base_decoder_model="Lakoc/fisher_dec_6_layers_multi_head" \
  --tokenizer_name="Lakoc/fisher_bpe" \
  --output_dir=$EXPERIMENT_PATH \
  --gradient_accumulation_steps="1" \
  --learning_rate="2e-3" \
  --warmup_steps="25000" \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="1000" \
  --evaluation_strategy="steps" \
  --eval_steps="1000" \
  --per_device_train_batch_size="64" \
  --per_device_eval_batch_size="32" \
  --group_by_length="True" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="16" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_wer" \
  --early_stopping_patience="10" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs="100" \
  --num_beams="5" \
  --max_len="128" \
  --greater_is_better="False" \
  --train_split="train_500" \
  --validation_split="dev_6" \
  --bf16 \
  --ctc_weight="0.3" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --pad_token="<pad>" \
  --lsm_factor="0.1" \
  --use_fbanks \
  --apply_augmentations \
  --audio_column_name="input_values" \
  --predict_with_generate \

cp /mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/LoCo-$EXPERIMENT.out $EXPERIMENT_PATH/
