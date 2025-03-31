#!/bin/bash
#SBATCH --time=00:60:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-16g,gpu-v100-32g  
#SBATCH --array=900-1439    # 90 hp combinations * 4 split types * 4 epochs = 900 jobs
#SBATCH --output=/scratch/work/masooda1/ToxBERT/output/MLP_hp_search_%a.out

VENV_PATH="/scratch/work/masooda1/.conda_envs/ToxBERT"
SCRIPT_PATH="/scratch/work/masooda1/ToxBERT/scripts/MLP_with_BERT_hp_search.py"
CONFIG_PATH="/scratch/work/masooda1/ToxBERT/scripts/config/MLP_TG_with_BERT.yml"

# Define hyperparameter arrays
ALPHA_VALUES=(0.0 0.25 0.5 0.75 1.0)
GAMMA_VALUES=(1.0 2.0 3.0)
L2_VALUES=(0.01 0.05 0.1 0.25 0.5 1.0)
DROPOUT_VALUES=(0.5)

# Calculate total hyperparameter combinations
N_ALPHA=${#ALPHA_VALUES[@]}    # 5
N_GAMMA=${#GAMMA_VALUES[@]}    # 3
N_L2=${#L2_VALUES[@]}         # 6
N_DROPOUT=${#DROPOUT_VALUES[@]} # 1
HP_TOTAL=$((N_ALPHA * N_GAMMA * N_L2 * N_DROPOUT))

# Define arrays for splits and epochs
split_types=("Split_Structure" "Split_ATC" "Split_Time" "Split_RandomPick")
feature_epochs=("init" 0 4 9)

# Calculate indices
job_idx=$SLURM_ARRAY_TASK_ID
split_idx=$((job_idx / (HP_TOTAL * 4)))
remainder=$((job_idx % (HP_TOTAL * 4)))
epoch_idx=$((remainder / HP_TOTAL))
hp_idx=$((remainder % HP_TOTAL))

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

# Create temporary config
TMP_CONFIG="/tmp/config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"
cp $CONFIG_PATH $TMP_CONFIG

# Calculate specific hyperparameter values for this job
alpha_idx=$((hp_idx / (N_GAMMA * N_L2 * N_DROPOUT)))
remainder=$((hp_idx % (N_GAMMA * N_L2 * N_DROPOUT)))
gamma_idx=$((remainder / (N_L2 * N_DROPOUT)))
remainder=$((remainder % (N_L2 * N_DROPOUT)))
l2_idx=$((remainder / N_DROPOUT))
dropout_idx=$((remainder % N_DROPOUT))

# Add hp_search section to config
cat >> $TMP_CONFIG << EOL

hp_search:
  alpha: [${ALPHA_VALUES[$alpha_idx]}]
  gamma: [${GAMMA_VALUES[$gamma_idx]}]
  optm_l2_lambda: [${L2_VALUES[$l2_idx]}]
  dropout_p: [${DROPOUT_VALUES[$dropout_idx]}]
EOL

# Update split_type and feature_epoch
sed -i "s/split_type: .*$/split_type: \"$SPLIT_TYPE\"/" $TMP_CONFIG

if [ "$FEATURE_EPOCH" = "init" ]; then
    sed -i "s/feature_epoch: .*$/feature_epoch: \"$FEATURE_EPOCH\"/" $TMP_CONFIG
else
    sed -i "s/feature_epoch: .*$/feature_epoch: $FEATURE_EPOCH/" $TMP_CONFIG
fi

# Export HP index for Python script
export HP_IDX=$hp_idx

echo "Running MLP training script for split type: $SPLIT_TYPE and feature epoch: $FEATURE_EPOCH"
srun python $SCRIPT_PATH --config $TMP_CONFIG

# Clean up
rm $TMP_CONFIG

echo 'Script execution completed.'