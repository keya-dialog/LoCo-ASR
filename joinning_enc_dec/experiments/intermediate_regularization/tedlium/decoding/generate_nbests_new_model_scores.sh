#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --time 6:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/ebranchformer_nbests_scores.out

EXPERIMENT="ebranchformer_nbests_scores"
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
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="0.0" \
  --dataset_name="LIUM/tedlium" \
  --dataset_config="release3" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --tokenizer_name="Lakoc/ted_uni500" \
  --from_pretrained="Lakoc/AED_ebranchformer_tedlium" \
  --external_lm="Lakoc/TED_CLM_gpt2_tedlium5" \
  --output_dir=$EXPERIMENT_PATH \
  --per_device_eval_batch_size="8" \
  --dataloader_num_workers="4" \
  --length_column_name="input_len" \
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
  --num_beams="50" \
  --max_len="512" \
  --external_lm_weight="0.15" \
  --decoding_ctc_weight="0.3" \
  --num_predictions_to_return="50" \
  --do_generate \
  --evaluation_splits validation test \
  --nbest_path_to_save="${EXPERIMENT_PATH}/nbests"
