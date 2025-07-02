# VitroBERT - Modeling DILI by pretraining BERT on invitro data

This repository contains the code of VitroBERT, a pretrained BERT based model with the ability to use biological and chemical data during pretraining stage

<p align="center">
  <img src="ToxBERT.png" width="50%" alt="VitroBERT Architecture"/>
</p>

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Reproducing Results](#reproducing-results)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [Citation](#citation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aidd-msca/VitroBERT.git
cd VitroBERT
```

2. Create and activate a conda environment:
```bash
conda create -y -q -n VitroBERT -c rdkit rdkit=2019.03.1.0 python=3.7.3
conda activate VitroBERT
```
if Aalto University (triton):
```bash
module load mamba # specific to Aalto University (triton)
source activate VitroBERT
```

3. Install dependencies:
```bash
pip install -e . 
```

4. Download and store pretrianed MolBERT model from [here](https://ndownloader.figshare.com/files/25611290)

## Data Preparation

The model requires two types of data:

1. Pretraining data: In-vitro data with binary labels
2. Fine-tuning data: Preclinical and Clinical data with binary labels

### Public data
Download downstream and pretrianing data from [here](https://figshare.com/articles/dataset/VitroBERT_-_Pretraining_and_downstream_data/28692518) and place in the `data/` directory

### Data Format
The input data should be with the following structure:
```python
{
    'SMILES': [...],  # List of SMILES strings
    'target1': [...],  # Binary labels (0 or 1)
    'target2': [...],
    # ... additional properties
}
```

### Data-preprocessing of pretraining data
This step is performed to 
- Filter assays and mlecules
- Filter incompatible molecules (metals, salt and >128)
- Split data into training and validation 

```bash
# Run the complete preprocessing pipeline
sbatch scripts/preprocess_invitro_data.sh \
    /path/to/data \
    /path/to/conda/env \
    /path/to/pretrained/MolBERT/weights
```
#### Or run individual scripts:
```bash
# 1. Normalize SMILES, filter assays and remove downstream molecules
python scripts/preprocess_invitro_data.py \
    --invitro_input_path /path/to/input/chembl20.parquet \
    --invivo_input_path /path/to/input/TG_GATES_SMILES.csv\
    --output_path /path/to/output/pretraining_data/invitro_selected_assays.parquet \
    --invitro_smiles_column smiles \
    --invivo_smiles_column SMILES \
    --min_pos_neg_per_assay 10 \
    --save_plots \
    --plot_path /path/to/output/pretraining_data/distribution_plots.png

# 2. Filter metals, salts, and molecules > 128, and compute MolBERT features (Baseline)
python scripts/featurizer.py \
    --input_path /path/to/output/pretraining_data/invitro_selected_assays.parquet \
    --output_dir /path/to/output/pretraining_data \
    --pretrained_MolBERT_weights /path/to/pretrained/weights

# 3. Split invitro data into train and validation sets
python scripts/split_data.py \
    --input_path /path/to/output/pretraining_data/invitro_filtered.pkl \
    --output_dir /path/to/output/pretraining_data \
    --split_type Random \
    --test_size 0.05

# Alternative split types available:
# --split_type Stratified  # Maintains label distribution across splits
# --split_type Scaffold    # Groups molecules by Murcko scaffolds (recommended for chemical data)
```

The pipeline will generate the following files:
- `invitro_selected_assays.parquet`: Preprocessed invitro data
- `invitro_filtered.pkl`: Filtered data with MolBERT features
- `invitro_train.pkl`: Training set
- `invitro_val.pkl`: Validation set
- `split_ratio_Random.csv`: Statistics about the data split
- `distribution_plots.png`: Distribution plots of the data

### Data directory Structure
Place your prepared data files in the `data/` directory:
```
data/
├── rawdata
    ├── TG_GATES_SMILES.csv
    ├── chembl20.parquet
├── pretraining_data
    ├── invitro_train.pkl
    ├── invitro_val.pkl
    ├── invitro_pos_weight_distribution.csv
└── downstream_data
```

### Downstream Data Preprocessing
For detailed instructions on preprocessing downstream data (TG-GATES histopathology and biochemistry data), see the comprehensive guide in [`src/datasets/README.md`](src/datasets/README.md).

## Model Architecture

The model consists of three main components:
1. BERT encoder for molecular representation
2. Masked language modeling head for pre-training
3. Task-specific heads for:
   - ADME property prediction
   - Physicochemical property prediction

Architecture details:
- BERT output dimension: 768
- Maximum sequence length: 128
- Hidden layer size: 2048
- Number of attention heads: 12

## Training the Model

Use the following script to pretrain VitroBERT by using invitro data
```bash
sbatch scripts/BERT_invitro_ADME_pretraining.sh \
    /path/to/invitro_data \
    /path/to/conda/env \
    /scripts/config/BERT_init_masking_physchem_invitro_head.yaml \
    /molbert_100epochs
```

## Downstream Task Training

After pretraining, train MLP heads for downstream tasks (TG-GATES data) using the pretrained BERT features.

### 1. Hyperparameter Search
First, perform hyperparameter search using cross-validation to find optimal parameters:

```bash
sbatch scripts/MLP_with_BERT_hp_search.sh
```

This script runs a comprehensive hyperparameter search across:
- Alpha values: [0.0, 0.25, 0.5, 0.75, 1.0] (focal loss parameter)
- Gamma values: [1.0, 2.0, 3.0] (focal loss parameter)
- L2 regularization: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
- Dropout rates: [0.5]
- All split types: Structure, ATC, Time, RandomPick
- Different pretrained epochs: init, 0, 4, 9

### 2. Final MLP Model Training
After hyperparameter search, train the final models using the best parameters:

```bash
sbatch scripts/MLP_with_BERT_best_model.sh
```

This script:
- Finds the best hyperparameters from the search results
- Trains final models with multiple random seeds (0-4)
- Evaluates on test sets for each split type and pretrained epoch
- Saves comprehensive results including metrics for all seeds

### Generated Results
The downstream training generates:
- Hyperparameter search results: `hp_search_{split_type}_epoch_{epoch}/hp_*.json`
- Final model results: `best_model_results_{split_type}_pretrained_epoch_{epoch}.json`
- Model checkpoints and training logs via Weights & Biases

## Reproducing Results

## Citation

If you use this code in your research, please cite:
```bibtex
@article{VitroBERT,
    title={VitroBERT - Modeling DILI by pretraining BERT on invitro data},
    author={Muhammad Arslan Masood, Samuel Kaski, Anamya Ajjolli Nagaraja, Katia Belaid, Natalie Mesens, Hugo Ceulemans, Dorota Herman, Markus Heinonen},
    journal={under review},
    year={2025}
}
```