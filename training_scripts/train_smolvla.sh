#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

# DATASET_REPO_ID="RoboMind/tienkung_gello_1rgb_normkey"
DATASET_REPO_ID="realman/pick_coffee_normkey"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--smolvla_base"

# Output directory
OUTPUT_DIR="outputs/train_smolvla/pick_coffee_test_1"

# Training Parameters
BATCH_SIZE=16
TOTAL_STEPS=280000
SAVE_FREQ=10000
ACTION_CHUNK_SIZE=50
NUM_WORKERS=16
USE_TENSORBOARD=false

CUDA_VISIBLE_DEVICES=0,1,2,3 python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=true \
    --output_dir=$OUTPUT_DIR --use_tensorboard=$USE_TENSORBOARD \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE
