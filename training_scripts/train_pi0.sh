#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

DATASET_REPO_ID="RoboMind/tienkung_gello_1rgb_normkey"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="outputs/train_pi0/test_robomind"

# Training Parameters
BATCH_SIZE=8
TOTAL_STEPS=280000
SAVE_FREQ=20000


python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ \
    --policy.max_state_dim=64 --policy.max_action_dim=64 --policy.train_expert_from_scratch=True
