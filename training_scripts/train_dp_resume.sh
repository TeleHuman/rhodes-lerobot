#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

DATASET_REPO_ID="lerobot/pusht"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="$MY_HOME/train_dp/test_consume"

# Training Parameters
BATCH_SIZE=24
TOTAL_STEPS=100000
SAVE_FREQ=5000
# EVAL_FREQ=1000
# EVAL_BATCH_SIZE=10

python lerobot/scripts/train.py \
    --resume=true \
    --config_path=$OUTPUT_DIR/checkpoints/last/pretrained_model \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ
