#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=/scratch/work/masooda1/ToxBERT/output/BERT_invitro_ADME_pretraining_single_gpu.out

# Check if all arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <base_folder> <venv_path> <config_path> <pretrained_weights>"
    echo "Example: sbatch $0 /scratch/work/masooda1/ToxBERT_github/data /scratch/work/masooda1/.conda_envs/ToxBERT /scratch/work/masooda1/ToxBERT/scripts/config/BERT_init_masking_physchem_invitro_head.yaml /scratch/work/masooda1/ToxBERT/MolBERT_checkpoints/molbert_100epochs/checkpoints/last.ckpt"
    exit 1
fi

BASE_FOLDER="$1"
VENV_PATH="$2"
CONFIG_PATH="$3"
PRETRAINED_WEIGHTS="$4"

# Get the project root directory (parent of base_folder)
PROJECT_ROOT="${BASE_FOLDER%/*}"

# Check if paths exist
if [ ! -d "$BASE_FOLDER" ]; then
    echo "Error: Base folder '$BASE_FOLDER' does not exist"
    exit 1
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Conda environment path '$VENV_PATH' does not exist"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' does not exist"
    exit 1
fi

if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "Error: Pretrained weights file '$PRETRAINED_WEIGHTS' does not exist"
    exit 1
fi

# Get number of GPUs using SLURM_GPUS_PER_NODE
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Debug prints
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"

module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Set environment variables
export TOXBERT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo 'Running training script...'
srun python "${PROJECT_ROOT}/scripts/BERT_invitro_ADME_pretraining.py" \
    --config_path "${CONFIG_PATH}" \
    --data_dir "${BASE_FOLDER}/data/pretraining_data" \
    --output_dir "${BASE_FOLDER}/model_outputs" \
    --pretrained_weights "${PRETRAINED_WEIGHTS}" \
    --wandb_key "27edf9c66b032c03f72d30e923276b93aa736429"

if [ $? -ne 0 ]; then
    echo 'Error: Training script failed.'
    exit 1
fi

echo 'Script execution completed successfully.'