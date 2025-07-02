#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

DATASET_REPO_ID="RoboTwin/all_tasks_50ep"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

MY_HOME="/gemini/space/users/ysy/data/"
# Output directory
OUTPUT_DIR="$MY_HOME/train_flowbc/rt"

# Training Parameters
BATCH_SIZE=32
TOTAL_STEPS=100000
SAVE_FREQ=5000
# EVAL_FREQ=1000
# EVAL_BATCH_SIZE=10
NUM_WORKERS=12

python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.type="flowbc" \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ \
    --num_workers=$NUM_WORKERS
