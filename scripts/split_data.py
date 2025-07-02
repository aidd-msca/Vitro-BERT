import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os
from collections import defaultdict

# RDKit imports for scaffold splitting
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def calculate_ratio(invitro, task):
    pos = (invitro[task] == 1).sum()
    neg = (invitro[task] == 0).sum()
    ratio = pos/neg
    return ratio

def generate_scaffold(smiles):
    """
    Generate Murcko scaffold for a SMILES string.
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        str: Scaffold SMILES string or original SMILES if scaffold generation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return smiles

def scaffold_split(data, test_size=0.05, smiles_column='Normalized_SMILES', random_state=42):
    """
    Split data based on molecular scaffolds.
    
    Args:
        data (pd.DataFrame): Input DataFrame with SMILES
        test_size (float): Proportion of data for validation set
        smiles_column (str): Name of column containing SMILES
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_data, val_data)
    """
    np.random.seed(random_state)
    
    # Generate scaffolds
    print("Generating molecular scaffolds...")
    scaffolds = data[smiles_column].apply(generate_scaffold)
    
    # Group molecules by scaffold
    scaffold_groups = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        scaffold_groups[scaffold].append(idx)
    
    # Sort scaffolds by group size (largest first)
    scaffold_groups = dict(sorted(scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True))
    
    # Split scaffolds into train and validation
    total_molecules = len(data)
    target_val_size = int(total_molecules * test_size)
    
    train_indices = []
    val_indices = []
    val_size = 0
    
    # Shuffle scaffold keys for randomness while maintaining scaffold integrity
    scaffold_keys = list(scaffold_groups.keys())
    np.random.shuffle(scaffold_keys)
    
    for scaffold in scaffold_keys:
        indices = scaffold_groups[scaffold]
        
        # If adding this scaffold to validation would exceed target size,
        # and we already have some validation data, add to training instead
        if val_size + len(indices) > target_val_size and val_size > 0:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)
            val_size += len(indices)
    
    # Create train and validation datasets
    train_data = data.iloc[train_indices].reset_index(drop=True)
    val_data = data.iloc[val_indices].reset_index(drop=True)
    
    print(f"Scaffold split results:")
    print(f"  Total scaffolds: {len(scaffold_groups)}")
    print(f"  Train scaffolds: {len(set(scaffolds.iloc[train_indices]))}")
    print(f"  Val scaffolds: {len(set(scaffolds.iloc[val_indices]))}")
    print(f"  Actual val ratio: {len(val_data) / len(data):.3f}")
    
    return train_data, val_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Split data into train and validation sets')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input pickle file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save output files')
    parser.add_argument('--split_type', type=str, default='Random',
                      choices=['Random', 'Stratified', 'Scaffold'],
                      help='Type of split to perform (Random, Stratified, or Scaffold)')
    parser.add_argument('--test_size', type=float, default=0.05,
                      help='Proportion of data to use for validation set')
    parser.add_argument('--smiles_column', type=str, default='Normalized_SMILES',
                      help='Name of column containing SMILES strings (default: Normalized_SMILES)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    invitro = pd.read_pickle(args.input_path)
    
    # Get all task columns (all columns except the first one)
    tasks = invitro.iloc[:,1:].columns.tolist()

    if args.split_type == "Stratified":
        # Create an instance of MultilabelStratifiedShuffleSplit
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
        # Get the split indices
        for train_index, val_index in splitter.split(invitro, invitro[tasks]):
            train_data = invitro.iloc[train_index].reset_index(drop=True)
            val_data = invitro.iloc[val_index].reset_index(drop=True)
            break

    elif args.split_type == "Random":
        train_data, val_data = train_test_split(invitro, test_size=args.test_size, random_state=42)
        # Reset indices
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
    
    elif args.split_type == "Scaffold":
        # Check if SMILES column exists
        if args.smiles_column not in invitro.columns:
            raise ValueError(f"SMILES column '{args.smiles_column}' not found in data. Available columns: {list(invitro.columns)}")
        
        train_data, val_data = scaffold_split(
            invitro, 
            test_size=args.test_size, 
            smiles_column=args.smiles_column,
            random_state=42
        )

    # Print statistics
    print(f"Total samples: {len(invitro)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Create lists to store the positive ratio for each task
    orig_ratios = []
    train_ratios = []
    val_ratios = []

    # For each task, get the ratio of positive cases
    for task in tqdm(tasks):
        orig_ratio = calculate_ratio(invitro, task)
        train_ratio = calculate_ratio(train_data, task)
        val_ratio = calculate_ratio(val_data, task)
        
        orig_ratios.append(orig_ratio)
        train_ratios.append(train_ratio)
        val_ratios.append(val_ratio)

    # Create a DataFrame to compare the ratio
    ratio_comparison = pd.DataFrame({
        'Task': tasks,
        'Original': orig_ratios,
        'Training': train_ratios,
        'Validation': val_ratios
    })

    # Add columns for absolute differences
    ratio_comparison['Train-Orig Diff'] = abs(ratio_comparison['Training'] - ratio_comparison['Original'])
    ratio_comparison['Val-Orig Diff'] = abs(ratio_comparison['Validation'] - ratio_comparison['Original'])
    ratio_comparison['Train-Val Diff'] = abs(ratio_comparison['Training'] - ratio_comparison['Validation'])
    
    # Save ratio comparison
    ratio_comparison.to_csv(os.path.join(args.output_dir, f"split_ratio_{args.split_type}.csv"), index=False)

    # Save split data
    train_data.to_pickle(os.path.join(args.output_dir, "Chembl20_filtered_for_MolBERT_train.pkl"))
    val_data.to_pickle(os.path.join(args.output_dir, "Chembl20_filtered_for_MolBERT_val.pkl"))

if __name__ == "__main__":
    main()
