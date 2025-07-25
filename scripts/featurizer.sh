#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0     # Modified: 6 fractions × 1 seed × 3 splits = 18 jobs
#SBATCH --output=/scratch/work/masooda1/ToxBERT/output/featurizer.out

### Activate your env ###
### you might need to change as per your system ###
VENV_PATH="/scratch/work/masooda1/.conda_envs/ToxBERT"

module load mamba
echo "Activating conda environment: ${VENV_PATH}"
source activate "${VENV_PATH}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

### Run the script ###
echo 'Running training script...'
srun python /scratch/work/masooda1/ToxBERT/scripts/featurizer.py

echo 'Script execution completed successfully.'