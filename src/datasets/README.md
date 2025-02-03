# Datasets

## Data Preparation Steps

### 1. Download Required Datasets
Download the following datasets from [Open TG-GATEs Database](https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html):
- Histopathology data (`open_tggates_hematology.zip`)
- Biochemistry data (`open_tggates_biochemistry.zip`)

After downloading, extract the files to `data/rawdata/`.

### 2. Extract SMILES from PubChem
```bash
python src/datasets/Extract_SMILES_from_PubChem.py \
    --input "data/rawdata/tx2c00378_si_001.xlsx" \
    --output "data/rawdata/TG_GATES_SMILES.csv"
```

### 3. Generate Binary Labels for Histopathology Data
```bash
python src/datasets/generate_histopathology_binary_labels.py \
    --input "data/rawdata/" \
    --output "data/binary_data/histopathology_binary_data.csv"
```

## Directory Structure
```
bert-invitro-adme/
├── data/
│   ├── rawdata/
│   │   ├── tx2c00378_si_001.xlsx           # Extracted histopathology data
│   │   ├── open_tggates_biochemistry.csv   # Extracted biochemistry data
│   │   └── TG_GATES_SMILES.csv             # Extracted SMILES data
│   └── binary_data/
│       └── histopathology_binary_data.csv  # Generated binary labels
```
