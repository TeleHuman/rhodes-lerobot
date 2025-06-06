#!/bin/zsh

# Dataset configuration
DATASET_REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="outputs/train_pi0/0603_bridge_data_ft"

# Training Parameters
BATCH_SIZE=8

# Task DIY
POLICY_TYPE=pi0
TASK_NAME=rhodes_${DATASET_NAME}_${POLICY_TYPE}_gpus${GPUS}

### accelerate launch arguments
GPUS=2
MIXED_PRECISION="bf16" # fp16 or bf16
MAIN_PROCESS_PORT=29500

### accelerate launch command
# accelerate launch \
#     --multi_gpu \
#     --num_processes=$GPUS \
#     --main_process_port=$MAIN_PROCESS_PORT \
#     --mixed_precision=$MIXED_PRECISION \
#     lerobot/scripts/train.py \
#     --dataset.repo_id=$DATASET_REPO_ID \
#     --dataset.root=$DATASET_ROOT \
#     --policy.path=$POLICY_PATH \
#     --policy.local_files_only=True \
#     --output_dir=$OUTPUT_DIR \
#     --batch_size=$BATCH_SIZE

### accelerate launch command with deepspeed
accelerate launch \
    --use_deepspeed \
    --num_processes=$GPUS \
    --deepspeed_config_file training_scripts/deepspeed_config/zero_stage1_config.json \
    --gradient_clipping=1.0 \
    --main_process_port=$MAIN_PROCESS_PORT \
    --mixed_precision=$MIXED_PRECISION \
    lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE


# OFFLINE_STEPS=100000
# EVAL_FREQ=1000
# BATCH_SIZE=8
# EVAL_BATCH_SIZE=10
# SAVE_FREQ=10000

# python -m accelerate.commands.launch --num_processes=$GPUS --mixed_precision=fp16 --main_process_port=$PORT lerobot/scripts/train.py \
#      --policy.type=$POLICY  \
#      --dataset.repo_id=$REPO_ID \
#      --env.type=$ENV \
#      --env.task=$TASK \
#      --output_dir=$TRAIN_DIR \
#      --batch_size=$BATCH_SIZE \
#      --steps=$OFFLINE_STEPS \
#      --eval_freq=$EVAL_FREQ --save_freq=$SAVE_FREQ --eval.batch_size=$EVAL_BATCH_SIZE --eval.n_episodes=$EVAL_BATCH_SIZE 