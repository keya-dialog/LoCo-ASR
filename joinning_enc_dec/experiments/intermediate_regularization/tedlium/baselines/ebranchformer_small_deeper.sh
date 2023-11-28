#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/tedlium_ebranchformer_16l_256h_gpt2_8l_256h_restart_specaug_fix.out

EXPERIMENT="tedlium_ebranchformer_16l_256h_gpt2_8l_256h_restart_specaug_fix"
PROJECT="TED2"
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
  --dataset_name="LIUM/tedlium" \
  --dataset_config="release3" \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="0.0" \
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed" \
  --feature_extractor_name="Lakoc/fisher_log_mel_extractor" \
  --base_decoder_model="Lakoc/gpt2_tiny_decoder_6_layers" \
  --tokenizer_name="Lakoc/ted_uni500" \
  --output_dir=$EXPERIMENT_PATH \
  --gradient_accumulation_steps="1" \
  --learning_rate="2e-3" \
  --warmup_steps="15000" \
  --logging_steps="10" \
  --save_strategy="epoch" \
  --evaluation_strategy="epoch" \
  --per_device_train_batch_size="48" \
  --per_device_eval_batch_size="48" \
  --report_to="wandb" \
  --optim="adamw_torch" \
  --dataloader_num_workers="24" \
  --length_column_name="input_len" \
  --load_best_model_at_end="True" \
  --metric_for_best_model="eval_wer" \
  --remove_unused_columns="False" \
  --save_total_limit="5" \
  --num_train_epochs="150" \
  --num_beams="4" \
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
  --from_encoder_decoder_config \
  --weight_decay="1e-6" \
  --max_grad_norm="5.0" \
  --decoder_pos_emb_fixed \
  --do_train \
  --do_evaluate \
  --decoding_ctc_weight="0.3" \
  --eval_beam_factor="10" \
  --evaluation_splits validation test \
  --joint_decoding_during_training \
  --apply_spec_augment \
  --num_steps_to_activate_spec_augment=5000 \
  --config_overrides="encoder_num_hidden_layers=16,decoder_n_layer=8" \
  --restart_from="/mnt/proj1/open-28-58/lakoc/LoCo-ASR/experiments/TED2_tedlium_ebranchformer_16l_256h_gpt2_8l_256h/checkpoint-41910"

cp /mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/LoCo-$EXPERIMENT.out $EXPERIMENT_PATH/
