from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import torch
from torch.utils.data import Dataset

import rdkit
print(rdkit.__version__)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from multiprocessing import Pool
from tqdm import tqdm


########################################################
# get_stratified_folds
#####################################################
def get_stratified_folds(X, y, ids, num_of_folds, config):

    if config["num_of_tasks"] > 1:
        mskf = MultilabelStratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=config["seed"])
    else:
        mskf = StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=config["seed"])
    fold_data = []
    for fold_idx, (train_index, val_index) in enumerate(mskf.split(X, np.nan_to_num(y, nan=0))):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        ids_train, ids_val = ids[train_index], ids[val_index]

        if config["num_of_tasks"] > 1:
            class_sum = min(np.nansum(y_val, axis = 0))
            if np.min(class_sum) < 1:
                raise ValueError("Error: No active compound in certain class")
        # Store the data for the current fold
        fold_data.append({
            'fold_idx': fold_idx,
            'train': {'X': X_train, 'y': y_train, 'ids': ids_train},
            'val': {'X': X_val, 'y': y_val,'ids': ids_val}
        })
    return fold_data

########################################################
# dataloader_for_numpy
#####################################################
class dataloader_for_numpy(Dataset):

    def __init__(self, X, y, x_type = 'SMILES'):
        
        if x_type == 'SMILES':
            self.x = X.tolist()
            self.n_samples = len(self.x)
        else:
            self.x = torch.tensor(X, dtype=torch.float32)
            self.n_samples = self.x.shape[0]

        self.y = torch.tensor(y, dtype=torch.float32)
        

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    

########################################################
# standardize_smiles
########################################################
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

