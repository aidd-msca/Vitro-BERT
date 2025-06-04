# ToxBERT - Modeling DILI by pretraining BERT on invitro data

This repository contains the code of ToxBERT, a pretrained BERT based model with the ability to use biological and chemical data during pretraining stage

<p align="center">
  <img src="ToxBERT.png" width="50%" alt="ToxBERT Architecture"/>
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
git clone https://github.com/aidd-msca/ToxBERT.git
cd ToxBERT
```

2. Create and activate a conda environment:
```bash
conda create -y -q -n ToxBERT -c rdkit rdkit=2019.03.1.0 python=3.7.3
conda activate ToxBERT
```
if Aalto University (triton):
```bash
module load mamba # specific to Aalto University (triton)
source activate ToxBERT
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
Download downstream and pretrianing data from [here](https://figshare.com/articles/dataset/ToxBERT_-_Pretraining_and_downstream_data/28692518) and place in the `data/` directory


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

Use the following script to pretrain ToxBERT by using invitro data
```bash
sbatch scripts/BERT_invitro_ADME_pretraining.sh \
    /path/to/invitro_data \
    /path/to/conda/env \
    /scripts/config/BERT_init_masking_physchem_invitro_head.yaml \
    /molbert_100epochs
```

## Pretraining by using public data
```bash
# Pretraining data can be downloaded from:
https://figshare.com/articles/dataset/Pretraining_data/28334303
```

## Downstream data and ToxBERT Embeddings
```bash
https://figshare.com/articles/dataset/ToxBERT_-_Pretraining_and_downstream_data/28692518
```

## Citation

If you use this code in your research, please cite:
```bibtex
@article{ToxBERT,
    title={ToxBERT - Modeling DILI by pretraining BERT on invitro data},
    author={Muhammad Arslan Masood, Samuel Kaski, Anamya Ajjolli Nagaraja, Katia Belaid, Natalie Mesens, Hugo Ceulemans, Dorota Herman, Markus Heinonen},
    journal={under review},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Muhammad Arslan Masood - arslan.asood@aalto.fi