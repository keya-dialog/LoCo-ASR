#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-58
#SBATCH --partition qcpu
#SBATCH --time 24:00:00

WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
EXPERIMENT_PATH="process_TED"

export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

source activate loco_asr

cd $WORK_DIR

python \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --dataset_name="LIUM/tedlium" \
  --dataset_config="release3" \
  --max_duration_in_seconds="45.0" \
  --min_duration_in_seconds="0.0" \
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --base_decoder_model="Lakoc/fisher_dec_6_layers" \
  --tokenizer_name="Lakoc/ted_tokenizer_v2" \
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
  --report_to="none" \
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
  --group_by_length="True" \
  --bf16 \
  --ctc_weight="0.3" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --pad_token="<pad>" \
  --lsm_factor="0.1" \
  --use_fbanks \
  --apply_augmentations \
  --predict_with_generate \
  --preprocessing_num_workers="128" \
  --fix_apostrophes \
  --remove_train_unks \
  --preprocess_dataset_only
