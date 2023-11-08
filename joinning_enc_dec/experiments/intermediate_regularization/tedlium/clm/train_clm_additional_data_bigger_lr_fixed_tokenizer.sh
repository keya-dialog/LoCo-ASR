#!/usr/bin/bash
#SBATCH --job-name TED_LM
#SBATCH --account OPEN-28-58
#SBATCH --partition qgpu
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/gpt2_tedlium_bigger_lr_fixed_tokenizer.out

EXPERIMENT="gpt2_tedlium_bigger_lr_fixed_tokenizer"
PROJECT="TED_CLM"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}_${EXPERIMENT}"
LM_DATA="${WORK_DIR}/data/ted_lm_raw.txt"
USER="lakoc"
TOKENIZER_NAME="ted_uni500"

export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"
export HF_HOME="${WORK_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/joinning_enc_dec/src"
export OMP_NUM_THREADS=64

conda deactivate
source activate loco_asr

cd $WORK_DIR

python joinning_enc_dec/src/trainers/train_tokenizer.py \
  --dataset_name="LIUM/tedlium" \
  --dataset_config="release3" \
  --tokenizer_name=$TOKENIZER_NAME \
  --vocab_size=500 \
  --tokenizer_type="unigram" \
  --text_column_name="text" \
  --train_split="train" \
  --additional_raw_data $LM_DATA \
  --skip_if_exists="${USER}/${TOKENIZER_NAME}"

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  joinning_enc_dec/src/trainers/train_clm.py \
  --model_type gpt2 \
  --config_overrides="n_embd=512,n_head=8,n_layer=16,vocab_size=500,bos_token_id=0,eos_token_id=1,n_positions=1024" \
  --tokenizer_name="${USER}/${TOKENIZER_NAME}" \
  --dataset_name LIUM/tedlium \
  --pad_token_id=3 \
  --dataset_config_name release3 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --auto_find_batch_size \
  --do_train \
  --do_eval \
  --logging_steps="5" \
  --save_strategy="steps" \
  --save_steps="3000" \
  --evaluation_strategy="steps" \
  --eval_steps="3000" \
  --num_train_epochs=15 \
  --warmup_steps=20000 \
  --learning_rate="4e-3" \
  --bf16 \
  --save_total_limit="2" \
  --output_dir $EXPERIMENT_PATH \
  --load_best_model_at_end \
  --preprocessing_num_workers 64 \
  --additional_raw_data $LM_DATA \
  --skip_if_exists="Lakoc/${EXPERIMENT}" \
  --push_to_hub \
  --early_stopping_patience="5"
