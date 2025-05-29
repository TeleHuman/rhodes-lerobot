#!/bin/zsh

# Dataset configuration
DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

# Pretrained model configuration  
POLICY_PATH="/data/zhangyang/huggingface_cache/hub/models--lerobot--pi0"


# Output directory
OUTPUT_DIR="outputs/train_pi0/run_lerobot_aloha_mobile_cabinet_test"

python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR