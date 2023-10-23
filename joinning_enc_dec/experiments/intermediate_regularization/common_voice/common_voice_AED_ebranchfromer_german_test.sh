#!/usr/bin/bash
#SBATCH --job-name CommonVoice
#SBATCH --account OPEN-28-58
#SBATCH --partition qgpu_exp
#SBATCH --gpus 8
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/common_voice_AED_ebranchformer_german_test.out

EXPERIMENT="common_voice_AED_ebranchformer_german_test"
PROJECT="CommonVoice"
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

#python joinning_enc_dec/src/trainers/train_tokenizer.py \
#  --dataset_name="mozilla-foundation/common_voice_13_0" \
#  --dataset_config="de" \
#  --tokenizer_name="Lakoc/common_voice_german0.5k" \
#  --vocab_size=500 \
#  --tokenizer_type="unigram" \
#  --text_column_name="sentence" \
#  --train_split="train"

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --dataset_name="mozilla-foundation/common_voice_13_0" \
  --dataset_config="de" \
  --max_duration_in_seconds="45.0" \
  --min_duration_in_seconds="0.0" \
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --base_decoder_model="Lakoc/fisher_dec_6_layers" \
  --tokenizer_name="Lakoc/common_voice_german0.5k" \
  --output_dir=$EXPERIMENT_PATH \
  --gradient_accumulation_steps="1" \
  --learning_rate="2e-3" \
  --warmup_steps="25000" \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="1000" \
  --evaluation_strategy="steps" \
  --eval_steps="1000" \
  --per_device_train_batch_size="8" \
  --per_device_eval_batch_size="8" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="24" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_wer" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs="50" \
  --num_beams="5" \
  --max_len="128" \
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
  --early_stopping_patience="100" \
  --text_column_name="sentence" \
  --preprocessing_num_workers="128" \
  --validation_slice="2000" \
  --from_encoder_decoder_config

cp /mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/LoCo-$EXPERIMENT.out $EXPERIMENT_PATH/