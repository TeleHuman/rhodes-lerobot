#!/bin/zsh

# Dataset configuration
ROBOMIND_ROOT="/gemini/space/shared_dataset/RoboMind_2_lerobot/benchmark1_0_release"
EMBODIMENT="tienkung_gello_1rgb"

# Automatically discover all dataset folders under the embodiment directory
EMBODIMENT_PATH="$ROBOMIND_ROOT/$EMBODIMENT"
REPO_IDS=()

# Check if the embodiment directory exists
if [[ -d "$EMBODIMENT_PATH" ]]; then
    # Iterate through all directories in the embodiment folder
    for dataset_dir in "$EMBODIMENT_PATH"/*; do
        if [[ -d "$dataset_dir" ]]; then
            # Extract just the folder name
            dataset_name=$(basename "$dataset_dir")
            # Add to REPO_IDS array in the required format
            REPO_IDS+=("$EMBODIMENT/$dataset_name")
        fi
    done
else
    echo "Error: Embodiment directory $EMBODIMENT_PATH does not exist"
    exit 1
fi

# Print discovered datasets for verification
# echo "Discovered datasets:"
# for repo_id in "${REPO_IDS[@]}"; do
#     echo "  - $repo_id"
# done
# echo "Total datasets found: ${#REPO_IDS[@]}"

# Pretrained model configuration  
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi0"

# Output directory
OUTPUT_DIR="$MY_HOME/train_pi0/train_tienkung_gello_1rgb"

# Training Parameters
BATCH_SIZE=32

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
    --batch_size=$BATCH_SIZE
