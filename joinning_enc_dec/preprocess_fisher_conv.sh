#!/bin/bash

#$ -N XLSR_data
#$ -q long.q@@blade
#$ -l ram_free=0.2G,mem_free=0.2G
#$ -l matylda5=0.1,matylda2=0.1
#$ -pe smp 24
#$ -o /mnt/matylda5/xpolok03/projects/LoCo-ASR/joinning_enc_dec/fisher_conv.o
#$ -e /mnt/matylda5/xpolok03/projects/LoCo-ASR/joinning_enc_dec/fisher_conv.e

# Limit job runtime to 24 h -> 86400 s, send SIGXCPU and SIGKILL if limit is breached
ulimit -t 86400

# Enable opening multiple files
ulimit -n 4096

# Enable bigger arrow shards
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/xpolok03/miniconda3/bin/activate /mnt/matylda5/xpolok03/envs/xlsr_pretraining/

# Ensure work directory exists
METADATA_DIR="/mnt/matylda4/kesiraju/tools/espnet/egs2/fisher_english/asr1"
OUT_DIR="/mnt/matylda5/xpolok03/projects/LoCo-ASR/datasets/fisher_conv"
WORK_DIR="/mnt/matylda5/xpolok03/projects/LoCo-ASR/joinning_enc_dec"


cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit
}

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HOME="${WORK_DIR}/../huggingface_cache" python $WORK_DIR/src/hf_dataset_builders/preprocess_fisher.py $WORK_DIR/src/hf_dataset_builders/conv_fisher $METADATA_DIR $OUT_DIR --num_proc 24
