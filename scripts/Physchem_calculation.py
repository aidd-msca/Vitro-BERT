#!/usr/bin/env python
# coding: utf-8

# # Physchem calculation for preclinical and clinical set



# # Physchem invitro

# In[1]:


import pandas as pd
from molbert.datasets.smiles import BertSmilesDataset
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from tqdm import tqdm


# In[2]:


invitro_data = pd.read_pickle("/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/SMILES_len_th_128/invitro_1m_300k_ADME_SMILES_filtered.pkl")
config_dict = {
    "max_seq_length": 128,
    "train_file": "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/02_05_2024/clinical_pre_clinical_with_blood_marker_filtered.csv",
    "num_physchem_properties": 200,
    "permute": False,
    
}


# In[3]:


featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config_dict["max_seq_length"], permute = False)
#elements = featurizer.load_periodic_table()[0]
#featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(max_length=config_dict["max_seq_length"], 
#                                                                allowed_elements=tuple(elements),
#                                                                permute = False)
config_dict["vocab_size"] = featurizer.vocab_size
train_dataset = BertSmilesDataset(
            input_path= config_dict['train_file'],
            featurizer= featurizer,
            single_seq_len= config_dict["max_seq_length"],
            total_seq_len= config_dict["max_seq_length"],
            num_physchem= config_dict["num_physchem_properties"],
            permute= config_dict["permute"],
            inference_mode = False
        )
descriptors_names = train_dataset.physchem_featurizer.descriptors


# In[6]:


from multiprocessing import Pool
def compute_physchem(smile):
    physchem = train_dataset.calculate_physchem_props(smile)
    return physchem

def compute_physchem_parallel(smiles_list):
    with Pool() as pool:
        
        total = len(smiles_list)
        with tqdm(total=total, ncols=80, desc="Processing") as pbar:
            results = []
            for physchem in pool.imap(compute_physchem, smiles_list):
                results.append(physchem)
                pbar.update(1)
            results
    return results


# In[9]:


physchem_df = compute_physchem_parallel(invitro_data.Normalized_SMILES.tolist())
physchem_df = pd.DataFrame(physchem_df)
physchem_df.columns = descriptors_names
physchem_df.insert(0, "SMILES", invitro_data.Normalized_SMILES.tolist())
# In[8]:

print("Saving pickle")
physchem_df.to_pickle("/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/SMILES_len_th_128/PhysChem_invitro_1m_300k_ADME_SMILES_filtered.pkl")

print("creating histograms")
import pandas as pd
import matplotlib.pyplot as plt

df = physchem_df.drop("SMILES", axis = 1)
#df = df.loc[:,df.var() > 1e-6]

# Assuming df is your DataFrame
n_cols = 10  # Number of columns per row
n_rows = (df.shape[1] + n_cols - 1) // n_cols  # Calculate number of rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))

for i, col in enumerate(df.columns):
    ax = axes[i // n_cols, i % n_cols]
    ax.hist(df[col].dropna(), bins=20)
    ax.set_title(col)

plt.tight_layout()
plt.show()

fig.savefig("/projects/home/mmasood1/TG GATE/Fig/invitro_PreClinical_Clinical/invitro/invitro_physchem_distribution.png", 
            dpi = 300, bbox_inches = 'tight')
# In[ ]:
print("Script Completed")



