#!/usr/bin/env python3
"""
Generate binary labels for blood markers from TG-GATES data.

This script processes TG-GATES biochemistry data to create binary toxicity labels
for various blood markers. It includes SMILES normalization, MolBERT filtering,
and dataset splitting functionality.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, Sequence, Any, Dict, Union, Optional
from multiprocessing import Pool

import pandas as pd
import numpy as np
import yaml
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.info')

# Add project path for MolBERT imports
sys.path.append('/scratch/work/masooda1/ToxBERT/src')

try:
    from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
    from molbert.models.smiles import SmilesMolbertModel
except ImportError:
    print("Warning: MolBERT modules not found. SMILES filtering will be skipped.")
    SmilesMolbertModel = None
    SmilesIndexFeaturizer = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MolBertFeaturizer:
    """
    MolBERT featurizer for transforming SMILES to embeddings.
    """
    
    def __init__(self, model, featurizer, device: str = None, embedding_type: str = 'pooled',
                 max_seq_len: Optional[int] = None, permute: bool = False) -> None:
        """
        Initialize MolBERT featurizer.
        
        Args:
            model: Trained MolBERT model
            featurizer: SMILES index featurizer
            device: Computing device ('cpu' or 'cuda')
            embedding_type: Method to reduce MolBERT encoding
            max_seq_len: Maximum sequence length for tokenizer
            permute: Whether to permute SMILES
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_type = embedding_type
        self.max_seq_len = max_seq_len
        self.permute = permute
        self.featurizer = featurizer
        self.model = model

    def __getstate__(self):
        """Handle pickling for multiprocessing."""
        self.__dict__.update({'model': self.model.to('cpu')})
        self.__dict__.update({'device': 'cpu'})
        return self.__dict__

    def transform_single(self, smiles: str) -> Tuple[np.ndarray, bool]:
        """Transform single SMILES string."""
        features, valid = self.transform([smiles])
        return features, valid[0]

    def transform(self, molecules: Sequence[Any]) -> Tuple[Union[Dict, np.ndarray], np.ndarray]:
        """Transform batch of molecules to features."""
        input_ids, valid = self.featurizer.transform(molecules)
        input_ids = self.trim_batch(input_ids, valid)

        # Prepare tensors
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        attention_mask = np.zeros_like(input_ids, dtype=np.int64)
        attention_mask[input_ids != 0] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model.model.bert(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )

        sequence_output, pooled_output = outputs

        # Filter invalid outputs
        valid_tensor = torch.tensor(
            valid, dtype=sequence_output.dtype, device=sequence_output.device, requires_grad=False
        )
        pooled_output = pooled_output * valid_tensor[:, None]
        
        return pooled_output.detach().cpu().numpy(), valid

    @staticmethod
    def trim_batch(input_ids, valid):
        """Trim batch to remove unnecessary padding."""
        if any(valid):
            _, cols = np.where(input_ids[valid] != 0)
        else:
            cols = np.array([0])
        
        max_idx = int(cols.max().item() + 1)
        return input_ids[:, :max_idx]


def standardize_smiles(smiles: str, remover: SaltRemover = None) -> str:
    """
    Standardize SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        remover: Salt remover instance
        
    Returns:
        Standardized SMILES string or np.nan if failed
    """
    if remover is None:
        remover = SaltRemover()
    
    config = {
        "StandardizeSmiles": True,
        "FragmentParent": False,
        "SaltRemover": True,
        "isomericSmiles": False,
        "kekuleSmiles": True,
        "canonical": True
    }
    
    try:
        if config["StandardizeSmiles"]:
            smiles = rdMolStandardize.StandardizeSmiles(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
            
        # Remove salts
        if config["SaltRemover"]:
            mol = remover.StripMol(mol, dontRemoveEverything=False)

        if config["FragmentParent"]:
            mol = rdMolStandardize.FragmentParent(mol)

        if config["kekuleSmiles"]:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            
        normalized_smiles = Chem.MolToSmiles(
            mol,
            isomericSmiles=config["isomericSmiles"],
            kekuleSmiles=config["kekuleSmiles"],
            canonical=config["canonical"],
            allHsExplicit=False
        )
        
        return normalized_smiles if normalized_smiles else np.nan
        
    except Exception as e:
        logger.debug(f"Failed to standardize SMILES {smiles}: {e}")
        return np.nan


def normalize_smiles_parallel(smiles_list: list) -> list:
    """
    Normalize SMILES list in parallel.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of normalized SMILES strings
    """
    with Pool() as pool:
        results = []
        total = len(smiles_list)
        with tqdm(total=total, ncols=80, desc="Normalizing SMILES") as pbar:
            for normalized_smiles in pool.imap(standardize_smiles, smiles_list):
                results.append(normalized_smiles)
                pbar.update(1)
    return results


def load_molbert_model(checkpoint_path: str) -> Tuple[Any, Any]:
    """
    Load MolBERT model and featurizer.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Tuple of (model, featurizer)
    """
    if SmilesMolbertModel is None:
        raise ImportError("MolBERT modules not available")
        
    model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    hparams_path = os.path.join(model_dir, 'hparams.yaml')
    
    # Load config
    with open(hparams_path) as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    config_dict["pretrained_model_path"] = checkpoint_path

    # Load model
    model = SmilesMolbertModel(config_dict)
    checkpoint = torch.load(config_dict["pretrained_model_path"], map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    device = "cpu"
    model.eval()
    model.freeze()
    model = model.to(device)

    # Load featurizer
    featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(126, permute=False)
    
    return model, featurizer


def filter_smiles_with_molbert(smiles_df: pd.DataFrame, checkpoint_path: str, 
                              batch_size: int = 5) -> pd.DataFrame:
    """
    Filter SMILES using MolBERT to remove invalid ones.
    
    Args:
        smiles_df: DataFrame with SMILES
        checkpoint_path: Path to MolBERT checkpoint
        batch_size: Batch size for processing
        
    Returns:
        Filtered DataFrame
    """
    try:
        model, featurizer = load_molbert_model(checkpoint_path)
        f = MolBertFeaturizer(model=model, featurizer=featurizer, device="cpu")
        
        smiles_list = smiles_df['Normalized_SMILES'].tolist()
        batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]
        
        logger.info(f"Processing {len(smiles_list)} SMILES in {len(batches)} batches")
        
        masks_all = []
        for batch_smiles in tqdm(batches, desc="Filtering SMILES"):
            _, masks = f.transform(batch_smiles)
            torch.cuda.empty_cache()
            masks_all.extend(masks.tolist())
        
        # Filter data
        filtered_df = smiles_df[masks_all].reset_index(drop=True)
        
        filtered_count = len(smiles_list) - sum(masks_all)
        logger.info(f"Filtered out {filtered_count} invalid SMILES")
        logger.info(f"Remaining SMILES: {sum(masks_all)}")
        
        return filtered_df
        
    except Exception as e:
        logger.warning(f"MolBERT filtering failed: {e}. Returning original data.")
        return smiles_df


def tbil_label(value: np.ndarray, control_mean: float) -> np.ndarray:
    """Generate TBIL toxicity labels based on control mean."""
    if 0 <= control_mean < 0.1:
        return value > 0.1
    elif 0.1 <= control_mean < 0.25:
        return value >= 0.35
    else:
        return value >= 0.5


def dbil_label(value: np.ndarray, control_mean: float) -> np.ndarray:
    """Generate DBIL toxicity labels based on control mean."""
    if control_mean <= 0.1:
        return value >= 0.15
    else:
        return value >= 0.3


def process_biochemistry_data(data_path: str) -> pd.DataFrame:
    """
    Process biochemistry data to generate binary toxicity labels.
    
    Args:
        data_path: Path to biochemistry CSV file
        
    Returns:
        DataFrame with binary toxicity labels
    """
    logger.info(f"Loading biochemistry data from {data_path}")
    tg_data = pd.read_csv(data_path, encoding="cp1252")
    
    # Select relevant columns
    desired_columns = [
        'COMPOUND_NAME', 'INDIVIDUAL_ID', 'SACRIFICE_PERIOD', 'DOSE_LEVEL',
        'ALP(IU/L)', 'TC(mg/dL)', 'TG(mg/dL)', 'TBIL(mg/dL)',
        'DBIL(mg/dL)', 'AST(IU/L)', 'ALT(IU/L)', 'LDH(IU/L)', 'GTP(IU/L)'
    ]
    selected_data = tg_data[desired_columns]
    
    # Validate control groups
    group = selected_data.groupby(['COMPOUND_NAME', "SACRIFICE_PERIOD"])
    missing_control_groups = group.filter(lambda x: "Control" not in x['DOSE_LEVEL'].values)
    if not missing_control_groups.empty:
        raise ValueError("Missing control groups found in data")
    
    # Group by compound-dose-time
    group = selected_data.groupby(['COMPOUND_NAME', "SACRIFICE_PERIOD", "DOSE_LEVEL"])
    
    # Define toxicity thresholds
    columns_ratio = [
        ('ALP(IU/L)', 1.5), ('AST(IU/L)', 2), ('ALT(IU/L)', 2),
        ('GTP(IU/L)', 3), ('TC(mg/dL)', 1.5), ('TG(mg/dL)', 3),
        ('TBIL(mg/dL)', tbil_label), ('DBIL(mg/dL)', dbil_label)
    ]
    
    # Process toxicity labels
    compounds = selected_data.COMPOUND_NAME.unique()
    dose_levels = sorted(selected_data.DOSE_LEVEL.unique())
    sacrifice_periods = selected_data.SACRIFICE_PERIOD.unique()
    animals_threshold = 1
    
    results = []
    
    logger.info("Processing toxicity labels...")
    for compound in tqdm(compounds, desc="Processing compounds"):
        for dose in dose_levels:
            for time in sacrifice_periods:
                if dose == "Control":
                    continue
                    
                try:
                    # Get control data
                    control = group.get_group((compound, time, "Control"))
                    compound_data = group.get_group((compound, time, dose))
                    
                    record = {
                        'COMPOUND_NAME': compound,
                        'DOSE_LEVEL': dose,
                        'SACRIFICE_PERIOD': time
                    }
                    
                    # Generate labels for each biomarker
                    for finding, threshold in columns_ratio:
                        if finding in ['DBIL(mg/dL)', 'TBIL(mg/dL)']:
                            # Special handling for bilirubin markers
                            if finding == 'DBIL(mg/dL)':
                                labels = dbil_label(
                                    compound_data[finding].values,
                                    control[finding].mean()
                                )
                            else:  # TBIL
                                labels = tbil_label(
                                    compound_data[finding].values,
                                    control[finding].mean()
                                )
                            record[finding] = int(labels.sum() > animals_threshold)
                        else:
                            # Standard ratio-based thresholding
                            labels = (compound_data[finding].values > 
                                    threshold * control[finding].mean())
                            record[finding] = int(labels.sum() > animals_threshold)
                            
                except KeyError:
                    # Missing data for this compound-dose-time combination
                    record = {
                        'COMPOUND_NAME': compound,
                        'DOSE_LEVEL': dose,
                        'SACRIFICE_PERIOD': time
                    }
                    findings = [key for key, _ in columns_ratio]
                    record.update({finding: None for finding in findings})
                
                results.append(record)
    
    # Convert to DataFrame and aggregate
    results_df = pd.DataFrame(results)
    
    # Nominal toxicity: if positive at any dose/time, consider positive
    nominal_toxicity = results_df.groupby(['COMPOUND_NAME']).sum() >= 1
    nominal_toxicity = nominal_toxicity.astype(int).reset_index()
    
    logger.info("Generated binary toxicity labels")
    return nominal_toxicity


def load_splits_data(splits_path: str) -> pd.DataFrame:
    """Load dataset splits information."""
    logger.info(f"Loading splits data from {splits_path}")
    splits = pd.read_excel(splits_path)
    splits = splits[["COMPOUND_NAME", "Split_RandomPick", "Split_Structure", "Split_ATC", "Split_Time"]]
    splits = splits.dropna().reset_index(drop=True)
    splits = splits.groupby("COMPOUND_NAME").agg('first').reset_index()
    return splits


def create_train_test_splits(combined_data: pd.DataFrame, output_dir: str):
    """
    Create train/test splits and save to files.
    
    Args:
        combined_data: DataFrame with SMILES, labels, and split information
        output_dir: Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    split_types = ["Split_RandomPick", "Split_Structure", "Split_ATC", "Split_Time"]
    selected_cols = [
        'Normalized_SMILES', 'ALP(IU/L)', 'AST(IU/L)', 'ALT(IU/L)', 
        'GTP(IU/L)', 'TC(mg/dL)', 'TBIL(mg/dL)', 'DBIL(mg/dL)'
    ]
    
    logger.info("Creating train/test splits...")
    for split_type in split_types:
        train = combined_data[combined_data[split_type] == "Training"].reset_index(drop=True)
        test = combined_data[combined_data[split_type] == "Test"].reset_index(drop=True)
        
        train = train[selected_cols].rename(columns={"Normalized_SMILES": "SMILES"})
        test = test[selected_cols].rename(columns={"Normalized_SMILES": "SMILES"})
        
        train_path = output_path / f"TG_train_{split_type}.csv"
        test_path = output_path / f"TG_test_{split_type}.csv"
        
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        
        logger.info(f"Saved {split_type}: {len(train)} train, {len(test)} test samples")


def calculate_class_weights(data: pd.DataFrame, output_path: str):
    """Calculate and save class weights for imbalanced datasets."""
    selected_cols = [
        'ALP(IU/L)', 'AST(IU/L)', 'ALT(IU/L)', 'GTP(IU/L)', 
        'TC(mg/dL)', 'TBIL(mg/dL)', 'DBIL(mg/dL)'
    ]
    
    pos = (data[selected_cols] == 1).sum()
    neg = (data[selected_cols] == 0).sum()
    pos_neg_ratio = pos / neg
    
    weights_df = pd.DataFrame({
        "Targets": pos_neg_ratio.index,
        "weights": pos_neg_ratio.values
    })
    
    weights_df.to_csv(output_path, index=False)
    logger.info(f"Saved class weights to {output_path}")


def main():
    """Main function to orchestrate the data processing pipeline."""
    parser = argparse.ArgumentParser(description='Generate binary labels for blood markers')
    
    parser.add_argument('--smiles-file', type=str, required=True,
                       help='Excel file with SMILES data')
    parser.add_argument('--biochemistry-file', type=str, required=True,
                       help='CSV file with biochemistry data')
    parser.add_argument('--splits-file', type=str, required=True,
                       help='Excel file with dataset splits')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed files')
    parser.add_argument('--molbert-checkpoint', type=str,
                       help='Path to MolBERT checkpoint for SMILES filtering')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Batch size for MolBERT processing')
    parser.add_argument('--skip-molbert', action='store_true',
                       help='Skip MolBERT filtering step')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and normalize SMILES
    logger.info("Loading SMILES data...")
    smiles_df = pd.read_excel(args.smiles_file)
    
    logger.info("Normalizing SMILES...")
    smiles_df['Normalized_SMILES'] = normalize_smiles_parallel(smiles_df['SMILES'].tolist())
    
    # Step 2: Filter SMILES with MolBERT (optional)
    if not args.skip_molbert and args.molbert_checkpoint:
        smiles_df = filter_smiles_with_molbert(
            smiles_df, args.molbert_checkpoint, args.batch_size
        )
    
    # Step 3: Process biochemistry data
    toxicity_labels = process_biochemistry_data(args.biochemistry_file)
    
    # Step 4: Merge SMILES with toxicity labels
    logger.info("Merging SMILES with toxicity labels...")
    combined_data = pd.merge(
        smiles_df, toxicity_labels,
        left_on="CompoundName", right_on="COMPOUND_NAME",
        how="left"
    )
    
    # Step 5: Add splits information
    splits_df = load_splits_data(args.splits_file)
    combined_data = pd.merge(combined_data, splits_df, on="COMPOUND_NAME", how="left")
    
    # Step 6: Create train/test splits
    create_train_test_splits(combined_data, args.output_dir)
    
    # Step 7: Calculate class weights
    weights_path = output_dir / "TG_data_pos_neg_ratio.csv"
    calculate_class_weights(combined_data, weights_path)
    
    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main()


# In[ ]:




