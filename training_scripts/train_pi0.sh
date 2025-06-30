#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

# DATASET_REPO_ID="RoboMind/tienkung_gello_1rgb_normkey"
DATASET_REPO_ID="rm75_3rgb/pick_coffee_normkey"
DATASET_ROOT="/gemini/space/users/loy/WAIC/rm75_3rgb/pick_coffee_normkey"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="outputs/train_pi0/test_single_gpu_3"

# Training Parameters
BATCH_SIZE=16
TOTAL_STEPS=280000
SAVE_FREQ=1000
ACTION_CHUNK_SIZE=20
NUM_WORKERS=16

CUDA_VISIBLE_DEVICES=0,1,2,3 python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=true \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --policy.max_state_dim=64 --policy.max_action_dim=64 --policy.train_expert_from_scratch=False \
    --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE
