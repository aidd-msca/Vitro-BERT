#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-15     # 4 split types * 4 epochs = 16 combinations (0-15)
#SBATCH --output=/scratch/work/masooda1/ToxBERT/output/MLP_with_BERT_%a.out

VENV_PATH="/scratch/work/masooda1/.conda_envs/ToxBERT"
SCRIPT_PATH="/scratch/work/masooda1/ToxBERT/scripts/MLP_with_BERT.py"
CONFIG_PATH="/scratch/work/masooda1/ToxBERT/scripts/config/MLP_TG_with_BERT.yml"

# Define arrays
split_types=("Split_Structure" "Split_ATC" "Split_Time" "Split_RandomPick")
feature_epochs=("init" 0 4 9)  # Mixed string and integers

# Calculate indices for split_type and feature_epoch
split_idx=$((SLURM_ARRAY_TASK_ID / 4))
epoch_idx=$((SLURM_ARRAY_TASK_ID % 4))

SPLIT_TYPE=${split_types[$split_idx]}
FEATURE_EPOCH=${feature_epochs[$epoch_idx]}

# Debug prints
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Split Type: $SPLIT_TYPE"
echo "Feature Epoch: $FEATURE_EPOCH"
echo "Node: $SLURM_NODELIST"
echo "Config Path: $CONFIG_PATH"

# Load and activate conda environment
module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create a temporary config file with the current split type and feature epoch
TMP_CONFIG="/tmp/config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"
cp $CONFIG_PATH $TMP_CONFIG
sed -i "s/split_type: .*$/split_type: \"$SPLIT_TYPE\"/" $TMP_CONFIG

# Handle feature_epoch differently based on whether it's "init" or a number
if [ "$FEATURE_EPOCH" = "init" ]; then
    sed -i "s/feature_epoch: .*$/feature_epoch: \"$FEATURE_EPOCH\"/" $TMP_CONFIG
else
    sed -i "s/feature_epoch: .*$/feature_epoch: $FEATURE_EPOCH/" $TMP_CONFIG
fi

echo "Running MLP training script for split type: $SPLIT_TYPE and feature epoch: $FEATURE_EPOCH"
srun python $SCRIPT_PATH --config $TMP_CONFIG

# Clean up temporary config
rm $TMP_CONFIG

echo 'Script execution completed.'