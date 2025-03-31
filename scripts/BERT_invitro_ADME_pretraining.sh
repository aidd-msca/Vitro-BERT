#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0     # Modified: 6 fractions × 1 seed × 3 splits = 18 jobs
#SBATCH --output=/scratch/work/masooda1/ToxBERT/output/BERT_invitro_ADME_pretraining_single_gpu.out

VENV_PATH="/scratch/work/masooda1/.conda_envs/ToxBERT"
CONFIG_PATH="/scratch/work/masooda1/ToxBERT/scripts/config/BERT_init_masking_physchem_invitro_head.yaml"

# Get number of GPUs using SLURM_GPUS_PER_NODE
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
#BASE_LR=0.00003  # 3.0e-5 in decimal form
#SCALED_LR=$(echo "$BASE_LR * $NUM_GPUS" | bc -l)  # Added -l flag for floating point arithmetic

# Debug prints
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"
#echo "Base learning rate: $BASE_LR"
#echo "Scaled learning rate: $SCALED_LR"

# Update learning rate in config file
#sed -i "s/learning_rate: .*$/learning_rate: $SCALED_LR/" $CONFIG_PATH

module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

echo 'Running training script...'
srun python /scratch/work/masooda1/ToxBERT/scripts/BERT_invitro_ADME_pretraining.py

# Optionally, restore original learning rate after training
#sed -i "s/learning_rate: .*$/learning_rate: $BASE_LR/" $CONFIG_PATH

echo 'Script execution completed successfully.'