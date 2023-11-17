#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --time 08:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/ebranchformer8_ctc04.out

EXPERIMENT="ebranchformer8_ctc04"
PROJECT="TED"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

conda deactivate
source activate loco_asr

cd $WORK_DIR

python \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --dataset_name="LIUM/tedlium" \
  --dataset_config="release3" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="0.0" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --tokenizer_name="Lakoc/ted_uni500" \
  --from_pretrained="/mnt/proj1/open-28-58/lakoc/LoCo-ASR/experiments/TED_tedlium_ebranchformer_tiny_esp_aug_uni500_fixed_pos_proper_scoring_regularization_averaging_v8/checkpoint-126808" \
  --output_dir=$EXPERIMENT_PATH \
  --gradient_accumulation_steps="1" \
  --learning_rate="2e-3" \
  --warmup_steps="15000" \
  --logging_steps="10" \
  --save_strategy="epoch" \
  --evaluation_strategy="epoch" \
  --per_device_train_batch_size="64" \
  --per_device_eval_batch_size="24" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="4" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_wer" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs="150" \
  --num_beams="40" \
  --max_len="512" \
  --greater_is_better="False" \
  --group_by_length="True" \
  --bf16 \
  --ctc_weight="0.3" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --pad_token="<pad>" \
  --mask_token="<mask>" \
  --lsm_factor="0.1" \
  --use_fbanks \
  --apply_augmentations \
  --predict_with_generate \
  --early_stopping_patience="50" \
  --preprocessing_num_workers="128" \
  --fix_apostrophes \
  --remove_train_unks \
  --wandb_predictions_to_save=600 \
  --weight_decay="1e-6" \
  --max_grad_norm="5.0" \
  --decoder_pos_emb_fixed \
  --do_evaluate \
  --decoding_ctc_weight="0.4" \
  --eval_beam_factor="1" \
  --evaluation_splits validation test\
  --joint_decoding_during_training \
  --apply_spec_augment \
  --num_steps_to_activate_spec_augment=5000

cp /mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/LoCo-$EXPERIMENT.out $EXPERIMENT_PATH/
