#!/bin/zsh

# Dataset configuration
ROBOMIND_ROOT="/gemini/space"
REPO_IDS=(
    "huggingface_cache/rhodes_lerobot/IPEC-COMMUNITY/fractal20220817_data_lerobot"
    # "huggingface_cache/rhodes_lerobot/IPEC-COMMUNITY/bridge_orig_lerobot"
    # "huggingface_cache/rhodes_lerobot/realman/3tasks"
)

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="$MY_HOME/train_pi0/test_multi_dataset"

# Training Parameters
BATCH_SIZE=128

# Transform REPO_IDS to a string that can be parsed into list[str]
REPO_IDS_STR="["
for repo_id in "${REPO_IDS[@]}"; do
    REPO_IDS_STR="${REPO_IDS_STR}'${repo_id}', "
done
REPO_IDS_STR="${REPO_IDS_STR%, }]"  # remove the last comma and space, add the ending bracket

python lerobot/scripts/train.py \
    --dataset.repo_id=$REPO_IDS_STR \
    --dataset.root=$ROBOMIND_ROOT \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=True \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE \
    --num_workers=0
