#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

DATASET_REPO_ID="RoboMind/clean_table"
DATASET_ROOT="/data/zhangyang/RoboMind-lerobot/benchmark1_0_release/tienkung_gello_1rgb/clean_table_2_241211"

# Pretrained model configuration  
POLICY_PATH="/data/zhangyang/huggingface_cache/hub/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="outputs/train_pi0/robomind_test"

# Training Parameters
BATCH_SIZE=1



python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE