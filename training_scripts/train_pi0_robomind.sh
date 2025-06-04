#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

DATASET_REPO_ID="RoboMind/tienkung_gello_1rgb"
# DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
DATASET_ROOT="/gemini/space/shared_dataset/RoboMind_2_lerobot/benchmark1_0_release/tienkung_gello_1rgb/clean_table_2_241211"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"


# Output directory
OUTPUT_DIR="outputs/train_pi0/test_robomind"

# Training Parameters
BATCH_SIZE=2



CUDA_VISIBLE_DEVICES=1 python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE
