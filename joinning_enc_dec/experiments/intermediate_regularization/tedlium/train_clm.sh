#!/usr/bin/bash
#SBATCH --job-name TED_LM
#SBATCH --account OPEN-28-58
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/LoCo-ASR/outputs/tedlium_clm1.out

EXPERIMENT="tedlium_clm1"
PROJECT="TED_CLM"
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
  joinning_enc_dec/src/trainers/train_clm.py \
  --model_type gpt2 \
  --config_overrides="n_embd=1024,n_head=8,n_layer=16,vocab_size=500,bos_token_id=0,eos_token_id=1,n_positions=256" \
  --tokenizer_name="Lakoc/ted_bpe500" \
  --dataset_name LIUM/tedlium \
  --dataset_config_name release3 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --auto_find_batch_size \
  --do_train \
  --do_eval \
  --num_train_epochs=40 \
  --warmup_steps=25000 \
  --learning_rate="1e-3" \
  --bf16 \
  --output_dir $EXPERIMENT_PATH
