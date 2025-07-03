#!/bin/zsh

# Dataset configuration
DATASET_REPO_ID="rm75_3rgb/pick_baozi"
DATASET_ROOT="/gemini/space/shared_dataset/Realman/WAIC/lerobot_result/waic_80/$DATASET_REPO_ID"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--smolvla_base"

# Training Parameters
BATCH_SIZE=64
TOTAL_STEPS=100000
SAVE_FREQ=10000
ACTION_CHUNK_SIZE=50
NUM_WORKERS=24
LEARNING_RATE=2.5e-5
USE_TENSORBOARD=false
SEED=42

# 
DATE1=$(date "+%y-%m-%d")
TIME1=$(date "+%H-%M-%S")

MODEL_NAME="pi0"
DATASET_NAME="${DATASET_REPO_ID##*/}"
GPU_NUM="gpu1"
BS="bs${BATCH_SIZE}"
STEPS_K=$(($TOTAL_STEPS / 1000))
STEPS="s${STEPS_K}K"
LR_SCI=$(printf "%.0e" $LEARNING_RATE | sed 's/e-0/e-/')
LR="lr${LR_SCI}"
CHUNK_SIZE="ck${ACTION_CHUNK_SIZE}"
SEED_STR="seed${SEED}"

# Output directory
OUTPUT_DIR="$MY_HOME/train_pi0/rm75_${DATASET_NAME}/${DATE1}_${TIME1}_${MODEL_NAME}_${GPU_NUM}_${CHUNK_SIZE}_${LR}_${BS}_${STEPS}_${SEED_STR}"
echo "Output dir: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 python lerobot/scripts/train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=true \
    --policy_optimizer_lr=$LEARNING_RATE \
    --output_dir=$OUTPUT_DIR --use_tensorboard=$USE_TENSORBOARD \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --seed=$SEED \
    --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE
