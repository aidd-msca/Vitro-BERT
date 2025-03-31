#!/bin/bash
#SBATCH --time=00:60:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-15     # 4 split types * 4 epochs = 16 combinations
#SBATCH --output=/scratch/work/masooda1/ToxBERT/output/A_MLP_final_%a.out

VENV_PATH="/scratch/work/masooda1/.conda_envs/ToxBERT"
SCRIPT_PATH="/scratch/work/masooda1/ToxBERT/scripts/MLP_with_BERT_best_model.py"
CONFIG_PATH="/scratch/work/masooda1/ToxBERT/scripts/config/MLP_TG_with_BERT.yml"

# Define arrays
split_types=("Split_Structure" "Split_ATC" "Split_Time" "Split_RandomPick")
feature_epochs=("init" 0 4 9)

# Calculate indices
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

# Load and activate conda environment
module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create temporary config
TMP_CONFIG="/tmp/final_config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"
cp $CONFIG_PATH $TMP_CONFIG

# Update split_type and feature_epoch in config
sed -i "s/split_type: .*$/split_type: \"$SPLIT_TYPE\"/" $TMP_CONFIG

if [ "$FEATURE_EPOCH" = "init" ]; then
    sed -i "s/feature_epoch: .*$/feature_epoch: \"$FEATURE_EPOCH\"/" $TMP_CONFIG
else
    sed -i "s/feature_epoch: .*$/feature_epoch: $FEATURE_EPOCH/" $TMP_CONFIG
fi

echo "Running final MLP training script"
srun python $SCRIPT_PATH --config $TMP_CONFIG --split-type $SPLIT_TYPE --feature-epoch $FEATURE_EPOCH

# Clean up
rm $TMP_CONFIG

echo 'Script execution completed.'