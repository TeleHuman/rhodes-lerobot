#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

DATASET_REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
# DATASET_REPO_ID="Rhodes_H1_2/simplified_6tasks/build_lego_dual_hands_tactus"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="$MY_HOME/train_pi0/bridge_data_ft_bs32_steps300k"

# Training Parameters
BATCH_SIZE=32
TOTAL_STEPS=300000
SAVE_FREQ=20000
NUM_WORKERS=8

CUDA_VISIBLE_DEVICES=1 python -m ipdb lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ
