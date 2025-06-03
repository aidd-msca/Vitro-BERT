import pandas as pd
import math
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.MolStandardize import rdMolStandardize
#IPythonConsole.drawOptions.comicMode=True
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')
import rdkit
from rdkit.Chem.SaltRemover import SaltRemover
print(rdkit.__version__)
import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import tarfile
import gzip
import shutil
from typing import Union

def standardize_smiles(smiles: str, remover=SaltRemover()) -> str:
    """
    Standardize a SMILES string using RDKit's MolStandardize.
    
    Args:
        smiles (str): Input SMILES string
        remover (SaltRemover): RDKit SaltRemover instance
        
    Returns:
        str: Standardized SMILES string or np.nan if standardization fails
    """
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
        
    except:
        return np.nan

def normalize_smiles_parallel(smiles_list: list) -> list:
    """
    Process a list of SMILES strings in parallel using multiprocessing.
    
    Args:
        smiles_list (list): List of SMILES strings to normalize
        
    Returns:
        list: List of normalized SMILES strings
    """
    with Pool() as pool:
        results = []
        with tqdm(total=len(smiles_list), ncols=80, desc="Processing") as pbar:
            for normalized_smiles in pool.imap(standardize_smiles, smiles_list):
                results.append(normalized_smiles)
                pbar.update(1)
    return results

def filter_assays(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """
    Filter assays based on minimum number of positive and negative samples.
    
    Args:
        df (pd.DataFrame): Input DataFrame with assays
        min_samples (int): Minimum number of positive/negative samples required
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    pos_counts = (df.iloc[:, 1:] == 1).sum(axis=0)
    neg_counts = (df.iloc[:, 1:] == 0).sum(axis=0)
    valid_assays = (pos_counts >= min_samples) & (neg_counts >= min_samples)
    selected_columns = ['smiles', 'Normalized_SMILES'] + valid_assays[valid_assays].index.tolist()
    return df[selected_columns]

def plot_distributions(filtered_df: pd.DataFrame, pos_counts_assays: pd.Series, 
                      neg_counts_assays: pd.Series, valid_assays: pd.Series):
    """
    Create visualization plots for assay and drug distributions.
    
    Args:
        filtered_df (pd.DataFrame): Filtered DataFrame
        pos_counts_assays (pd.Series): Counts of positive samples per assay
        neg_counts_assays (pd.Series): Counts of negative samples per assay
        valid_assays (pd.Series): Boolean mask for valid assays
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot assay distributions
    pos_assays = pos_counts_assays[valid_assays].sort_values(ascending=False)
    neg_assays = neg_counts_assays[valid_assays].sort_values(ascending=False)
    
    ax1.plot(pos_assays.values, label='Positive', color='red')
    ax1.plot(neg_assays.values, label='Negative', color='green')
    ax1.set_yscale('log')
    ax1.set_title('Distribution of Positive/Negative Samples per Assay')
    ax1.set_xlabel('Assay Index (sorted by count)')
    ax1.set_ylabel('Count (log scale)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot drug distributions
    pos_drugs = (filtered_df.iloc[:, 2:] == 1).sum(axis=1).sort_values(ascending=False)
    neg_drugs = (filtered_df.iloc[:, 2:] == 0).sum(axis=1).sort_values(ascending=False)
    measurements = (filtered_df.iloc[:, 2:] != -1).sum(axis=1).sort_values(ascending=False)
    
    ax2.plot(measurements.values, label='total_measurements', color='blue')
    ax2.plot(pos_drugs.values, label='Positive', color='red')
    ax2.plot(neg_drugs.values, label='Negative', color='green')
    ax2.set_yscale('log')
    ax2.set_title('Distribution of Positive/Negative Results per Drug')
    ax2.set_xlabel('Drug Index (sorted by count)')
    ax2.set_ylabel('Count (log scale)')
    ax2.legend()
    ax2.grid(True)

def read_data_file(file_path: str, csv_sep: str = ',') -> pd.DataFrame:
    """
    Read data from various file formats (csv, parquet, tar.gz).
    For tar.gz files, specifically looks for 'chembl20.parquet'.
    
    Args:
        file_path (str): Path to the input file
        csv_sep (str): Separator for CSV files (default: ',')
        
    Returns:
        pd.DataFrame: Loaded DataFrame
        
    Raises:
        ValueError: If file format is not supported or chembl20.parquet not found
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Handle regular files
    if file_path.suffix == '.parquet':
        print(f"Reading parquet file: {file_path}")
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        print(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path, sep=csv_sep)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats are: .parquet, .csv, .tar.gz")
    
    print(f"Successfully loaded data with shape: {df.shape}")
    return df

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process ChEMBL20 data for ToxBERT pretraining')
    
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input file (supported formats: .parquet, .csv, .tar.gz)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save processed data (will be saved as parquet)'
    )
    
    parser.add_argument(
        '--smiles_column',
        type=str,
        default='smiles',
        help='Name of the column containing SMILES strings (default: smiles)'
    )
    
    parser.add_argument(
        '--min_samples',
        type=int,
        default=10,
        help='Minimum number of positive/negative samples required for assays (default: 10)'
    )
    
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Save distribution plots to output directory'
    )
    
    parser.add_argument(
        '--csv_sep',
        type=str,
        default=',',
        help='Separator for CSV files (default: ,)'
    )
    
    return parser.parse_args()

def main():
    """Main function to process ChEMBL20 data."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_path}")
    try:
        chembl20 = read_data_file(args.input_path, csv_sep=args.csv_sep)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Ensure required columns exist
    if args.smiles_column not in chembl20.columns:
        print(f"Error: SMILES column '{args.smiles_column}' not found in data. Available columns: {list(chembl20.columns)}")
        return
    
    # Rename SMILES column to standard name for processing
    chembl20 = chembl20.rename(columns={args.smiles_column: 'smiles'})
    
    # Normalize SMILES
    print(f"Original unique SMILES: {chembl20.smiles.nunique()}")
    normalized_smiles_list = normalize_smiles_parallel(chembl20.smiles.tolist())
    chembl20['Normalized_SMILES'] = normalized_smiles_list
    print(f"Normalized unique SMILES: {chembl20.Normalized_SMILES.nunique()}")
    
    # Remove duplicates
    chembl20 = chembl20.drop_duplicates(subset=["Normalized_SMILES"]).reset_index(drop=True)
    
    # Filter assays and create visualizations
    filtered_df = filter_assays(chembl20, min_samples=args.min_samples)
    pos_counts_assays = (chembl20.iloc[:, 1:] == 1).sum(axis=0)
    neg_counts_assays = (chembl20.iloc[:, 1:] == 0).sum(axis=0)
    valid_assays = (pos_counts_assays >= args.min_samples) & (neg_counts_assays >= args.min_samples)
    
    # Print statistics
    print(f"Original number of assays: {chembl20.shape[1]-2}")
    print(f"Number of assays with ≥{args.min_samples} pos and neg: {len(filtered_df.columns)-2}")
    print(f"Shape of filtered dataset: {filtered_df.shape}")
    
    # Create and optionally save visualizations
    plot_distributions(filtered_df, pos_counts_assays, neg_counts_assays, valid_assays)
    if args.save_plots:
        plot_path = output_dir / "distribution_plots.png"
        plt.savefig(plot_path)
        print(f"Saved distribution plots to {plot_path}")
    
    # Save processed data
    print(f"Saving processed data to {args.output_path}")
    filtered_df.to_parquet(args.output_path)

if __name__ == "__main__":
    main()