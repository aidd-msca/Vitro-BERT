import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os

def calculate_ratio(invitro, task):
    pos = (invitro[task] == 1).sum()
    neg = (invitro[task] == 0).sum()
    ratio = pos/neg
    return ratio

def parse_arguments():
    parser = argparse.ArgumentParser(description='Split data into train and validation sets')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input pickle file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save output files')
    parser.add_argument('--split_type', type=str, default='Random',
                      choices=['Random', 'Stratified'],
                      help='Type of split to perform (Random or Stratified)')
    parser.add_argument('--test_size', type=float, default=0.05,
                      help='Proportion of data to use for validation set')
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

    if args.split_type == "Random":
        train_data, val_data = train_test_split(invitro, test_size=args.test_size, random_state=42)
        # Reset indices
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)

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
