#!/usr/bin/bash
#SBATCH --job-name CommonVoice
#SBATCH --account OPEN-28-58
#SBATCH --partition qgpu
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/common_voice_AED_ebranchformer_regularized.out

EXPERIMENT="common_voice_AED_ebranchformer_regularized"
PROJECT="CommonVoice"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID=$EXPERIMENT
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

source ~/miniconda3/bin/activate ~/miniconda3/envs/loco_asr

cd $WORK_DIR

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --dataset_name="mozilla-foundation/common_voice_13_0" \
  --dataset_config="cs" \
  --max_duration_in_seconds="35.0" \
  --min_duration_in_seconds="0.0" \
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
  --save_steps="5000" \
  --evaluation_strategy="steps" \
  --eval_steps="5000" \
  --per_device_train_batch_size="64" \
  --per_device_eval_batch_size="32" \
  --group_by_length="True" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="24" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_wer" \
  --early_stopping_patience="100" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs="100" \
  --num_beams="5" \
  --max_len="128" \
  --greater_is_better="False" \
  --bf16 \
  --ctc_weight="0.3" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --pad_token="<pad>" \
  --lsm_factor="0.1" \
  --use_fbanks \
  --apply_augmentations \
  --audio_column_name="input_values" \
  --text_column_name="labels" \
  --predict_with_generate

cp /mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/LoCo-$EXPERIMENT.out $EXPERIMENT_PATH/
