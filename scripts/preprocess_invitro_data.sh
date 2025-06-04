#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0    
#SBATCH --output=/scratch/work/masooda1/ToxBERT/output/split_data.out

# Check if all arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <base_folder> <venv_path> <pretrained_MolBERT_weights>"
    echo "Example: sbatch $0 /scratch/work/masooda1/ToxBERT_github/data /scratch/work/masooda1/.conda_envs/ToxBERT /scratch/work/masooda1/ToxBERT/MolBERT_checkpoints/molbert_100epochs/checkpoints/last.ckpt"
    exit 1
fi

BASE_FOLDER="$1"
VENV_PATH="$2"
PRETRAINED_WEIGHTS="$3"

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

if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "Error: Pretrained weights file '$PRETRAINED_WEIGHTS' does not exist"
    exit 1
fi

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

echo 'Running preprocessing script...'
srun python "${PROJECT_ROOT}/scripts/preprocess_invitro_data.py" \
    --invitro_input_path "${BASE_FOLDER}/rawdata/chembl20.parquet" \
    --invivo_input_path "${BASE_FOLDER}/rawdata/TG_GATES_SMILES.csv" \
    --output_path "${BASE_FOLDER}/pretraining_data/chembl20_selected_assays_with_normalzied_smiles.parquet" \
    --invitro_smiles_column smiles \
    --invivo_smiles_column SMILES \
    --min_pos_neg_per_assay 10 \
    --save_plots \
    --plot_path "${BASE_FOLDER}/pretraining_data/my_distribution_plots.png"

if [ $? -ne 0 ]; then
    echo 'Error: Preprocessing script failed.'
    exit 1
fi

echo 'Running featurizer script...'
srun python "${PROJECT_ROOT}/scripts/featurizer.py" \
    --input_path "${BASE_FOLDER}/pretraining_data/chembl20_selected_assays_with_normalzied_smiles.parquet" \
    --output_dir "${BASE_FOLDER}/pretraining_data" \
    --pretrained_MolBERT_weights "${PRETRAINED_WEIGHTS}"

if [ $? -ne 0 ]; then
    echo 'Error: Featurizer script failed.'
    exit 1
fi

echo 'Running training script...'
srun python "${PROJECT_ROOT}/scripts/split_data.py" \
    --input_path "${BASE_FOLDER}/pretraining_data/Chembl20_filtered_for_MolBERT.pkl" \
    --output_dir "${BASE_FOLDER}/pretraining_data" \
    --split_type "Random" \
    --test_size 0.05

if [ $? -ne 0 ]; then
    echo 'Error: Training script failed.'
    exit 1
fi

echo 'Script execution completed successfully.'



