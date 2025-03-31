import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def calculate_ratio(invitro, task):
    pos = (invitro[task] == 1).sum()
    neg = (invitro[task] == 0).sum()
    ratio = pos/neg
    return ratio

invitro = pd.read_pickle("/scratch/work/masooda1/ToxBERT/data/pretraining_data/Chembl20_filtered_for_MolBERT.pkl")
split_type = "Random"
test_size = 0.05
# Get all task columns (all columns except the first one)
tasks = invitro.iloc[:,1:].columns.tolist()

if split_type == "Stratified":
    # Create an instance of MultilabelStratifiedShuffleSplit
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    # Get the split indices
    for train_index, val_index in splitter.split(invitro, invitro[tasks]):
        train_data = invitro.iloc[train_index].reset_index(drop=True)
        val_data = invitro.iloc[val_index].reset_index(drop=True)
        break

if split_type == "Random":
    train_data, val_data = train_test_split(invitro, test_size=test_size, random_state=42)
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

# For each task, get the ratio of positive cases (assuming binary labels where 1 is positive)
for task in tqdm(tasks):
    # Using value_counts(normalize=True) to get proportions
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
ratio_comparison.to_csv(f"/scratch/work/masooda1/ToxBERT/data/pretraining_data/split_ratio_{split_type}.csv", index = False)

train_data.to_pickle(f"/scratch/work/masooda1/ToxBERT/data/pretraining_data/Chembl20_filtered_for_MolBERT_train.pkl")
val_data.to_pickle(f"/scratch/work/masooda1/ToxBERT/data/pretraining_data/Chembl20_filtered_for_MolBERT_val.pkl")
