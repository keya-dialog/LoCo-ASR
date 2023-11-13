#!/usr/bin/bash
#SBATCH --job-name CommonVoice
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/common_voice_AED_ebranchformer_cs4.out

EXPERIMENT="common_voice_AED_ebranchformer_cs4"
PROJECT="CommonVoice"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"
USER="lakoc"
TOKENIZER_NAME="CV_cs_uni500"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID=$EXPERIMENT
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

conda deactivate
source activate loco_asr

cd $WORK_DIR

python joinning_enc_dec/src/trainers/train_tokenizer.py \
  --dataset_name="mozilla-foundation/common_voice_13_0" \
  --dataset_config="cs" \
  --tokenizer_name=$TOKENIZER_NAME \
  --vocab_size=500 \
  --tokenizer_type="unigram" \
  --text_column_name="sentence" \
  --train_split="other" \
  --skip_if_exists="${USER}/${TOKENIZER_NAME}"

python \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --dataset_name="mozilla-foundation/common_voice_13_0" \
  --dataset_config="cs" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="0.0" \
  --base_encoder_model="Lakoc/ebranchformer_enc_8_layers" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --base_decoder_model="Lakoc/gpt2_tiny_decoder_4_layers" \
  --tokenizer_name="Lakoc/CV_cs_uni500" \
  --output_dir=$EXPERIMENT_PATH \
  --gradient_accumulation_steps="1" \
  --learning_rate="2e-4" \
  --warmup_steps="5000" \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="500" \
  --evaluation_strategy="steps" \
  --eval_steps="500" \
  --per_device_train_batch_size="96" \
  --per_device_eval_batch_size="48" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="8" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_wer" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs="200" \
  --num_beams="4" \
  --max_len="512" \
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
  --early_stopping_patience="100" \
  --text_column_name="sentence" \
  --preprocessing_num_workers="128" \
  --fix_apostrophes \
  --wandb_predictions_to_save=100 \
  --from_encoder_decoder_config \
  --weight_decay="1e-6" \
  --max_grad_norm="5.0" \
  --do_train \
  --train_split other \
  --evaluation_splits validation test \
  --do_eval \
  --decoding_ctc_weight="0.3" \
  --eval_beam_factor="5" \
  --validation_slice 500 \
  --track_ctc_loss \
  --config_overrides="decoder_average_logits=True"

cp /mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/LoCo-$EXPERIMENT.out $EXPERIMENT_PATH/
