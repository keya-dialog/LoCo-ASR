#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-58
#SBATCH --partition qgpu
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --time 4:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/joint_decoding_wer_test_shallow_fusion_5.out

EXPERIMENT="joint_decoding_wer_test_shallow_fusion_5"
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

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  joinning_enc_dec/src/trainers/AED_from_enc_dec.py \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="0.0" \
  --dataset_name="LIUM/tedlium" \
  --dataset_config="release3" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --tokenizer_name="Lakoc/ted_uni500" \
  --from_pretrained="/mnt/proj1/open-28-58/lakoc/LoCo-ASR/experiments/TED_tedlium_ebranchformer_tiny_esp_no_aug_uni500_fixed_pos_proper_scoring/checkpoint-39550" \
  --external_lm="Lakoc/TED_CLM_gpt2_tedlium3" \
  --output_dir=$EXPERIMENT_PATH \
  --per_device_eval_batch_size="8" \
  --dataloader_num_workers="4" \
  --length_column_name="input_len" \
  --preprocessing_num_workers="128" \
  --remove_unused_columns="False" \
  --group_by_length="True" \
  --bf16 \
  --ctc_weight="0.3" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --pad_token="<pad>" \
  --mask_token="<mask>" \
  --use_fbanks \
  --decoder_pos_emb_fixed \
  --fix_apostrophes \
  --remove_train_unks \
  --predict_with_generate \
  --num_beams="40" \
  --max_len="512" \
  --external_lm_weight="0.5" \
  --decoding_ctc_weight="0.3" \
  --evaluation_splits test \
  --do_eval
