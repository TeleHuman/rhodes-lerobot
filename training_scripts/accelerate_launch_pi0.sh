#!/bin/zsh
# preparation on os.environ
# export GLOO_SOCKET_IFNAME=eth0
# # 启用 NCCL debug 日志
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Dataset configuration
# DATASET_REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
DATASET_REPO_ID="RoboMind/tienkung_gello_1rgb_normkey"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="$MY_HOME/train_pi0/multi_gpu_test_1"

### accelerate launch arguments
GPUS=1
MIXED_PRECISION="bf16" # fp16 or bf16 or no
MAIN_PROCESS_PORT=29510
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CLIPPING=10.0

### Lerobot Training Parameters
BATCH_SIZE=8
TOTAL_STEPS=280000
SAVE_FREQ=200
LEARNING_RATE=0.0005


### accelerate launch command
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes=$GPUS \
    --config_file=training_scripts/accelerate_configs/accelerate_ds_stage1.yaml \
    --main_process_port=$MAIN_PROCESS_PORT \
    --mixed_precision=$MIXED_PRECISION \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --gradient_clipping=$GRADIENT_CLIPPING \
    lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR \
    --optimizer.lr=$LEARNING_RATE \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ