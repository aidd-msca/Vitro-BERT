from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import torch

import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import seaborn as sns

path_to_checkpoint = '/projects/home/mmasood1/TG GATE/MolBERT/molbert/molbert_100epochs/molbert_100epochs/checkpoints/last.ckpt'
f = MolBertFeaturizer(path_to_checkpoint)

data_dir = "/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/"
targets = pd.read_pickle(data_dir + 'invitro_1m_plus_300k_labels.pkl')
SMILES_information = pd.read_pickle(data_dir + 'invitro_1m_plus_300k_SMILES.pkl')
smiles_list = SMILES_information.Normalized_SMILES.tolist()

batch_size = 2000
batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]
print(len(smiles_list))

features_all, masks_all = [],[]
for batch_smiles in tqdm(batches):
    _, masks = f.transform(batch_smiles)
    torch.cuda.empty_cache()
    masks_all = masks_all + masks.tolist()

filtered_SMILES = SMILES_information[masks_all].reset_index(drop = True)
filtered_targets = targets[masks_all].reset_index(drop = True)

filtered_smiles = len(smiles_list) - len(masks_all)
print(f"We filtered {filtered_smiles} SMILES")
data_dir = "/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/"
filtered_targets.to_pickle(data_dir + 'invitro_1m_plus_300k_labels_filtered.pkl')
filtered_SMILES.to_pickle(data_dir + 'invitro_1m_plus_300k_SMILES_filtered.pkl')

print("Script Completed")