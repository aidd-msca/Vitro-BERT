# Datasets

## Data Preparation Steps

### 1. Download Required Datasets
Download the following datasets from [Open TG-GATEs Database](https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html):
- Histopathology data (`open_tggates_hematology.zip`)
- Biochemistry data (`open_tggates_biochemistry.zip`)

After downloading, extract the files to `data/rawdata/`.

### 2. Extract SMILES from PubChem
Extract canonical SMILES strings for compounds using PubChemPy:

```bash
python src/datasets/Extract_SMILES_from_PubChem.py \
    --input "data/rawdata/tx2c00378_si_001.xlsx" \
    --output "data/rawdata/TG_GATES_SMILES.csv"
```

**Options:**
- `--sheet`: Sheet name in Excel file (default: "Sheet1")
- `--limit`: Limit number of compounds to process (for testing)

### 3. Generate Binary Labels for Histopathology Data
Process histopathology data to create binary toxicity labels:

```bash
python src/datasets/generate_histopathology_binary_labels.py \
    --input "data/rawdata/" \
    --output "data/downstream_data/histopathology_binary_data.csv"
```

### 4. Generate Binary Labels for Blood Markers
Process biochemistry data to create binary labels for blood markers with comprehensive pipeline:

```bash
python src/datasets/generate_blood_markers_binary_labels.py \
    --smiles-file "data/rawdata/Animal_GAN_TGGATES_SMILES.xlsx" \
    --biochemistry-file "data/rawdata/open_tggates_biochemistry.csv" \
    --splits-file "data/rawdata/Animal_GAN_TGGATES_splits.xlsx" \
    --output-dir "data/downstream_data/" \
    --molbert-checkpoint "/path/to/molbert/checkpoint.ckpt" \
    --batch-size 5
```

**Options:**
- `--molbert-checkpoint`: Path to MolBERT model for SMILES validation (optional)
- `--batch-size`: Batch size for MolBERT processing (default: 5)
- `--skip-molbert`: Skip MolBERT filtering step entirely

**Generated Files:**
- `TG_train_Split_RandomPick.csv` / `TG_test_Split_RandomPick.csv`
- `TG_train_Split_Structure.csv` / `TG_test_Split_Structure.csv`
- `TG_train_Split_ATC.csv` / `TG_test_Split_ATC.csv`
- `TG_train_Split_Time.csv` / `TG_test_Split_Time.csv`
- `TG_data_pos_neg_ratio.csv` (class weights)

**Blood Markers Included:**
- `ALP(IU/L)` - Alkaline Phosphatase
- `AST(IU/L)` - Aspartate Transaminase
- `ALT(IU/L)` - Alanine Transaminase
- `GTP(IU/L)` - Gamma-Glutamyl Transpeptidase
- `TC(mg/dL)` - Total Cholesterol
- `TBIL(mg/dL)` - Total Bilirubin
- `DBIL(mg/dL)` - Direct Bilirubin

## Directory Structure
```
src/datasets/
├── data/
│   ├── rawdata/
│   │   ├── tx2c00378_si_001.xlsx                    # Histopathology data
│   │   ├── open_tggates_biochemistry.csv            # Biochemistry data
│   │   ├── Animal_GAN_TGGATES_SMILES.xlsx          # SMILES data
│   │   ├── Animal_GAN_TGGATES_splits.xlsx          # Dataset splits
│   │   └── TG_GATES_SMILES.csv                     # Extracted SMILES
│   ├── binary_data/
│   └── downstream_data/
│       ├── histopathology_binary_data.csv          # Histopathology binary labels
│       ├── TG_train_Split_RandomPick.csv           # Training sets
│       ├── TG_test_Split_RandomPick.csv            # Test sets
│       ├── TG_train_Split_Structure.csv            # (for each split type)
│       ├── TG_test_Split_Structure.csv
│       ├── TG_train_Split_ATC.csv
│       ├── TG_test_Split_ATC.csv
│       ├── TG_train_Split_Time.csv
│       ├── TG_test_Split_Time.csv
│       └── TG_data_pos_neg_ratio.csv               # Class weights
```

## Scripts Overview

### Extract_SMILES_from_PubChem.py
- Extracts canonical SMILES from compound names using PubChemPy
- Handles batch processing with progress tracking
- Filters out invalid compounds

### generate_histopathology_binary_labels.py
- Processes histopathology findings into binary toxicity labels
- Filters findings by minimum frequency
- Creates compound-level binary matrices

### generate_blood_markers_binary_labels.py
- Complete pipeline with:
  - SMILES normalization using RDKit
  - Optional MolBERT filtering for SMILES validation
  - Biochemistry data processing with toxicity thresholds
  - Multiple dataset splits (Random, Structure, ATC, Time)
  - Class weight calculation for imbalanced datasets
  - Comprehensive logging and error handling

## Quick Start Example

For a complete blood markers pipeline:

```bash
# 1. Extract SMILES (if not already done)
python src/datasets/Extract_SMILES_from_PubChem.py \
    --input data/rawdata/tx2c00378_si_001.xlsx \
    --output data/rawdata/TG_GATES_SMILES.csv

# 2. Process blood markers with all splits
python src/datasets/generate_blood_markers_binary_labels.py \
    --smiles-file data/rawdata/Animal_GAN_TGGATES_SMILES.xlsx \
    --biochemistry-file data/rawdata/open_tggates_biochemistry.csv \
    --splits-file data/rawdata/Animal_GAN_TGGATES_splits.xlsx \
    --output-dir data/downstream_data/ \
    --skip-molbert  # Skip if MolBERT not available
```

## Notes

- **SMILES Normalization**: Uses RDKit for standardization, salt removal, and canonicalization
- **Toxicity Thresholds**: Different thresholds for each biomarker based on clinical significance
- **Bilirubin Handling**: Special logic for TBIL and DBIL based on control group means
- **Missing Data**: Handled gracefully with appropriate null value assignment
- **Multiple Splits**: Supports random, structural, ATC, and temporal dataset splits
- **Class Imbalance**: Automatic calculation of positive/negative ratios for weighting
