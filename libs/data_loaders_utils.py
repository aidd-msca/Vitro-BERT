import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import mmread
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import scipy.sparse as sp
import deepchem as dc
from deepchem.data import NumpyDataset
from rdkit import RDLogger  
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog('rdApp.*')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def standardize(smiles, config, remover=SaltRemover()):
    # follows the steps in
    # https://github.com/rdkit/rdkit/blob/master/Docs/Notebooks/MolStandardize.ipynb
    
    if config["StandardizeSmiles"]:
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        smiles = rdMolStandardize.StandardizeSmiles(smiles)

    mol = Chem.MolFromSmiles(smiles)
    # remove salts
    if config["SaltRemover"]:
        mol = remover.StripMol(mol, dontRemoveEverything=False) 

    if config["FragmentParent"]:
        mol = rdMolStandardize.FragmentParent(mol) 

    normalized_smiles = Chem.MolToSmiles(mol, 
                        isomericSmiles = config["isomericSmiles"],
                        kekuleSmiles = config["kekuleSmiles"],
                        canonical = config["canonical"],
                        allHsExplicit = False)
    if normalized_smiles == '':
        normalized_smiles = np.nan
    return normalized_smiles

def get_basic_marcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold

def scaffold_split_deepchem(data, config):
    data = data[["SMILES",config["selected_tasks"]]]
    response = data[config["selected_tasks"]].values.astype(float)

    # compute fingerprints
    featurizer = dc.feat.CircularFingerprint(size=1024, radius=6)
    X = featurizer.featurize(data.SMILES)

    # split data into train and test
    splitter = dc.splits.ScaffoldSplitter()
    dc_dataset = NumpyDataset(X = X, y = response, ids = data.SMILES)
    train_set, test_set = splitter.train_test_split(dataset = dc_dataset,
                                                    frac_train = config["train_frac"], 
                                                    seed = config["seed"])

    train_set = pd.DataFrame({"SMILES":train_set.ids,
                config["selected_tasks"]:train_set.y,
                "fold" : "Train"})

    test_set = pd.DataFrame({"SMILES":test_set.ids,
                config["selected_tasks"]:test_set.y,
                "fold": "Test"})
    
    return train_set, test_set
    
def make_train_test_fold(complete_data, config):
    # Scaffold computation
    if type(config["selected_tasks"]) == str:
        selected_tasks = [config["selected_tasks"]]
    else:
        selected_tasks = config["selected_tasks"]
    selected_col = ["SMILES"] + selected_tasks
    data = complete_data[selected_col]
    data.insert(2,'Scafold',data.SMILES.apply(get_basic_marcko_scaffold))
    computed_clusters = data['Scafold'].tolist()
    
    # Stratification would be based on 1-columns
    task_for_stratification = config["task_for_stratification"]
    repsonse_task_for_stratification = complete_data[task_for_stratification].fillna(0).values.astype(float)
    response = data[selected_tasks].fillna(0).values.astype(float)

    # Splitting
    data.insert(3,"fold", -1)
    gss = StratifiedGroupKFold(n_splits=5, random_state= None, shuffle = False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["seed"])

    ######################################################    
    if config["split_method"] == "StratifiedGroupKFold":
        # stratify based on DLIL_binary, + scafold
        for i, (train_index, test_index) in enumerate(gss.split(data.SMILES.values, repsonse_task_for_stratification, computed_clusters)):
            data.loc[test_index, 'fold'] = i

    if config["split_method"] == "StratifiedKFold":
        # stratify based on DLIL_binary, + scafold
        for i, (train_index, test_index) in enumerate(skf.split(data.SMILES.values, response)):
            data.loc[test_index, 'fold'] = i
    
    if config["split_method"] == "scaffold_split_deepchem":
        train_set, test_set = scaffold_split_deepchem(data, config)
    #######################################################

    if config["split_method"] != "scaffold_split_deepchem":
        # Train, test fold creation
        fold_list = [0,1,2,3,4]
        most_diverse_fold = data.groupby("fold")["Scafold"].nunique().idxmax()
        most_toxic_fold = data.groupby("fold")[selected_tasks].sum().sum(axis = 1).idxmax()
        
        if config["test_set_creteria"] == "most_diverse_fold":
            train_folds = list(filter(lambda x: x!= most_diverse_fold, fold_list))
            data.loc[data[data.fold == most_diverse_fold].index, "fold"] = "Test"

        if config["test_set_creteria"] == "most_toxic_fold":
            train_folds = list(filter(lambda x: x!= most_toxic_fold, fold_list))
            data.loc[data[data.fold == most_toxic_fold].index, "fold"] = "Test"

        data.loc[data[data.fold.isin(train_folds)].index, "fold"] = "Train"

        train_set = data[data["fold"] == "Train"].reset_index(drop = True)
        test_set = data[data["fold"] == "Test"].reset_index(drop = True)

    #print("train-test compounds", train_set.shape[0], test_set.shape[0])
    #print("Train_set pos ratio", train_set[config["selected_tasks"]].sum() / train_set.shape[0])
    #print("Test_set pos ratio", test_set[config["selected_tasks"]].sum() /  test_set.shape[0])
    return train_set, test_set
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
# scafoldsplit_train_test (version 2)
#####################################################
class train_test_scafold_split():
    def __init__(self, target_file, preclinical_tasks, clinical_tasks, FP_size, train_frac):
        self.target_file = target_file
        self.preclinical_tasks = preclinical_tasks
        self.clinical_tasks = clinical_tasks
        self.FP_size = FP_size
        self.train_frac = train_frac

    def convert_into_dc_dataset(self, data, selected_tasks):
        featurizer = dc.feat.CircularFingerprint(size=self.FP_size, radius=6)
        X = featurizer.featurize(data.SMILES)
        y = data.loc[:, selected_tasks].values
        dc_dataset = NumpyDataset(X=X, y=y, ids=data.SMILES)
        return dc_dataset

    def split(self, data, selected_tasks):
        dc_dataset = self.convert_into_dc_dataset(data, selected_tasks)

        # split data into train and test
        splitter = dc.splits.ScaffoldSplitter()

        train_set, test_set = splitter.train_test_split(dataset=dc_dataset, frac_train=self.train_frac, seed=42)
        return train_set, test_set

    def scaffoldsplit_train_test(self):
        data = pd.read_csv(self.target_file)
        preclinical_data = data[~data.loc[:, "DILI_binary"].isnull()].reset_index(drop=True)
        clinical_data = data[data.loc[:, "DILI_binary"].isnull()].reset_index(drop=True)

        preclinical_train_set, preclinical_test_set = self.split(preclinical_data, self.preclinical_tasks)

        clinical_train_set, clinical_test_set = self.split(clinical_data, self.clinical_tasks)

        # merge both
        train_ids = np.concatenate([preclinical_train_set.ids, clinical_train_set.ids], axis=0)
        test_ids = np.concatenate([preclinical_test_set.ids, clinical_test_set.ids], axis=0)
        train_set = data[data.SMILES.isin(train_ids)]
        test_set = data[data.SMILES.isin(test_ids)]

        selected_tasks = self.preclinical_tasks + self.clinical_tasks
        train_set = self.convert_into_dc_dataset(train_set, selected_tasks)
        test_set = self.convert_into_dc_dataset(test_set, selected_tasks)

        # compute fingerprints
        print('train_test_features', train_set.X.shape, test_set.X.shape)
        print('train_test_targets', train_set.y.shape, test_set.y.shape)
        return train_set, test_set
    
########################################################
# scafoldsplit_train_test
#####################################################
def scafoldsplit_train_test(target_file, selected_tasks, FP_size, train_frac):
    data = pd.read_csv(target_file)
    y = data.loc[:,selected_tasks].values
    # compute fingerprints
    featurizer = dc.feat.CircularFingerprint(size=FP_size, radius=6)
    X = featurizer.featurize(data.SMILES)

    # split data into train and test
    splitter = dc.splits.ScaffoldSplitter()
    dc_dataset = NumpyDataset(X = X, y = y, ids = data.SMILES)
    train_set, test_set = splitter.train_test_split(dataset = dc_dataset,
                                                    frac_train = train_frac, 
                                                    seed = 42)
    print('train_test_features',train_set.X.shape, test_set.X.shape)
    print('train_test_targets',train_set.y.shape, test_set.y.shape)
    return train_set, test_set
########################################################
# invitro 1 million 
#####################################################
class invitro_million(Dataset):

    def __init__(self):
        targets = sp.csr_matrix(mmread("/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/after_therholding/filtered_targets.mtx"))
        smiles = np.load("/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/smiles_invitro_full_set.npy", allow_pickle = True)
        
        self.x = smiles.tolist()
        self.y = targets
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], np.squeeze(self.y[index].toarray())

    def __len__(self):
        return self.n_samples

########################################################
# SIDER SOC 
#####################################################
class SIDER_SOC(Dataset):

    def __init__(self, data_path):

        feature_target = pd.read_csv(data_path)
        self.x = feature_target.SMILES.tolist()
        self.y = feature_target.loc[:,"Hepatobiliary disorders":"Injury, poisoning and procedural complications"].values        
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

########################################################
# Chemprop with Matrix Factorization
#####################################################
class MF_Dataset(Dataset):
    def __init__(self,data_path):

        feature_target = pd.read_csv(data_path)
 
        self.x = feature_target.SMILES.tolist()
        self.y = feature_target.loc[:,"Apoptosis":"hepatobiliary_disorders"].values
        self.drug_index = np.arange(0, len(self.x), 1)
        self.task_index = np.arange(0, self.y.shape[1], 1)

        self.n_samples = len(self.x)
        self.num_rows, self.num_cols = self.drug_index.shape[0], self.task_index.shape[0]

    def __len__(self):

        return self.n_samples

    def __getitem__(self, index):

        x = self.x[index]
        y = self.y[index,:]
        drug_index = self.drug_index[index]
        task_index = self.task_index
        
        return (
            torch.tensor(drug_index, dtype = torch.long),
            x,
            torch.tensor(task_index, dtype = torch.long),
            torch.tensor(y, dtype = torch.float) 
        )
########################################################
# Dataloader for Tox21
#####################################################

class Tox21Dataset(Dataset):

    def __init__(self, dataset, config):
        #selected_tasks = config['selected_tasks']
        self.x = torch.tensor(dataset.X, dtype=torch.float32)
        self.y = torch.tensor(dataset.y, dtype=torch.float32)
        self.w = torch.tensor(dataset.w, dtype=torch.float32)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.w[index]

    def __len__(self):
        return self.n_samples
########################################################
# Dataloader for Matrix factorization model
########################################################

class ADRDataset(Dataset):
    def __init__(self,feature_target, config):

        y = feature_target.values

        # Molecular fingerprints
        selected_rows = feature_target.index
        self.x = sp.load_npz(config['FP_file'])
        self.x = self.x[selected_rows,:].toarray().astype(np.float32)
        
        # get samplewise weights
        deepchem_dataset = dc.data.NumpyDataset(self.x, y,
                                        w = np.ones(y.shape),  
                                        ids = np.arange(y.shape[0]))
        transformer = dc.trans.BalancingTransformer(dataset = deepchem_dataset)
        deepchem_dataset = transformer.transform(deepchem_dataset)
        sample_weight = deepchem_dataset.w

       # Get the number of rows and columns in the matrix
        self.num_rows, self.num_cols = y.shape

        # Create a list of tuples containing (row_index, col_index, value)
        self.data = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.data.append((i, j, sample_weight[i,j], y[i, j]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #row_idx, col_idx, value = self.data[idx]
        # replace nan with -1
        drug_index,side_effect_index,w, ADR = self.data[idx]
        #ADR = np.nan_to_num(ADR, nan = -1.0)
        drug_attribute = self.x[drug_index,:]

        return (
            torch.tensor(drug_index, dtype = torch.int),
            torch.tensor(side_effect_index, dtype = torch.int),
             torch.tensor(w, dtype = torch.float),
            torch.tensor(ADR, dtype = torch.float),
            torch.tensor(drug_attribute, dtype=torch.float)
        )
########################################################
# get invitro_PreClinical_clinical fold
########################################################
def get_invitro_PreClinical_clinical_folds(fold, file):
    
    df = pd.read_csv(file)
    feature_target_train = df[df.BioSig_KFold != fold]
    feature_target_val = df[df.BioSig_KFold == fold]
        
    return feature_target_train, feature_target_val

########################################################
# dataloader invitro_PreClinical_clinical
########################################################
class dataloader_invitro_PreClinical_clinical_folds(Dataset):

    def __init__(self,feature_target, config):
        
        if config['data_modality'] == 'SIDER':
            feature_target = feature_target[~feature_target.hepatobiliary_disorders.isnull()]
            self.y = feature_target.loc[:,'10001551':'hepatobiliary_disorders']
        
        if config['data_modality'] == 'invitro_PreClinical_clinical':
            self.y = feature_target.loc[:,'104543_level_5.0':'hepatobiliary_disorders']

        if config['data_modality'] == 'PreClinical_clinical':
            self.y = feature_target.loc[:,'Hypertrophy, hepatocellular':'hepatobiliary_disorders']

        if config['data_modality'] == 'invitro':
            feature_target = feature_target[feature_target.loc[:,'104543_level_5.0':'cytoHepG224BE_level_5.5'].count(axis = 1) >0]
            self.y = feature_target.loc[:,'104543_level_5.0':'cytoHepG224BE_level_5.5']
            
        selected_rows = feature_target.index
        self.x = sp.load_npz(config['FP_file'])
        self.x = self.x[selected_rows,:].toarray().astype(np.float32)
        self.x = torch.tensor(self.x)
        self.y.fillna(-1, inplace = True)
        self.y = torch.from_numpy(self.y.values.astype(np.float32))
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index],self.y[index] #self.n[index]
    
    def __len__(self):
        return self.n_samples 

########################################################
# dataloader_invitro_to_clinical
######################################################## 
class dataloader_invitro_to_clinical(Dataset):

    def __init__(self,feature_target, config):

        # use invitro values as features
        self.x = feature_target.loc[:,'104543_level_5.0':"cytoHepG224BE_level_5.5"]
        self.y = feature_target.loc[:,config['selected_tasks']]
        
        self.x = torch.from_numpy(self.x.values.astype(np.float32))
        self.y = torch.from_numpy(self.y.values.astype(np.float32))
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index],self.y[index] #self.n[index]
    
    def __len__(self):
        return self.n_samples 

########################################################
# dataloader_FP_plus_invitro_to_clinical
########################################################
class dataloader_FP_plus_invitro_to_clinical(Dataset):

    def __init__(self,feature_target, config):

        # use invitro values as features
        invitro_features = feature_target.loc[:,'104543_level_5.0':"cytoHepG224BE_level_5.5"].values.astype(np.float32)
        
        # use FP as features
        selected_rows = feature_target.index
        FP = sp.load_npz(config['FP_file'])
        FP = FP[selected_rows,:].toarray().astype(np.float32)
        self.x = np.concatenate((invitro_features,FP), axis = 1)

        self.y = feature_target.loc[:,config['selected_tasks']]
        self.y = torch.from_numpy(self.y.values.astype(np.float32))
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index],self.y[index] #self.n[index]
    
    def __len__(self):
        return self.n_samples
########################################################
# get SIDER fold
########################################################

def get_SIDER_folds(fold, file):
    
    SIDER = pd.read_pickle(file)

    feature_target_train = SIDER[SIDER.BioSig_KFold != fold].reset_index(drop = True)
    feature_target_val = SIDER[SIDER.BioSig_KFold == fold].reset_index(drop = True)
        
    return feature_target_train, feature_target_val

########################################################
# SIDER Dataloader
########################################################
class dataloader_SIDER(Dataset):

    def __init__(self,feature_target, config):
        
        if config['data_modality'] == 'SIDER':
            feature_target = feature_target[feature_target.hepatobiliary_disorders!= -1]
            
        self.x = torch.from_numpy(feature_target.loc[:,0:4095].to_numpy().astype(np.float32))

        selected_col = config["selected_tasks"]
        self.y = feature_target.loc[:,selected_col].values
        self.y = torch.from_numpy(self.y.astype(np.float32))
        self.y = self.y.reshape(-1,config['num_of_tasks'])
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index],self.y[index] #self.n[index]
        
    def __len__(self):
        return self.n_samples  

########################################################
# get_fold_Clinical_pre_Clinical
########################################################

def get_fold_Clinical_pre_Clinical(fold):
    
    folder = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/"
    Clinical_pre_Clinical_binary_targets = pd.read_pickle(folder + "Clinical_pre_Clinical_binary_targets.pkl")

    feature_target_train = Clinical_pre_Clinical_binary_targets[Clinical_pre_Clinical_binary_targets.BioSig_KFold != fold].reset_index(drop = True)
    feature_target_val = Clinical_pre_Clinical_binary_targets[Clinical_pre_Clinical_binary_targets.BioSig_KFold == fold].reset_index(drop = True)
        
        
    return feature_target_train, feature_target_val


########################################################
# Data Loaders for binar classification
########################################################
class dataloader_Clinical_pre_Clinical(Dataset):

    def __init__(self,feature_target, config):
        
        feature_col = np.arange(0,4096).tolist()
        self.x = torch.from_numpy(feature_target[feature_col].to_numpy().astype(np.float32))

        selected_col = config["selected_tasks"]
        
        self.y = feature_target[selected_col].values
        self.y = torch.from_numpy(self.y.astype(np.float32))
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index],self.y[index] #self.n[index]
        
    def __len__(self):
        return self.n_samples  
    
########################################################
# Data Loaders NDT for pathologies - updated
########################################################
class dataloader_TG_JNJ_MiniTox(Dataset):

    def __init__(self,feature,target):
        
        self.feature, self.target = feature,target
        
        self.feature = self.feature.drop('COMPOUND_NAME', axis = 1)
        self.x = torch.from_numpy(self.feature.iloc[:,:4096].to_numpy().astype(np.float32))
        self.d = torch.from_numpy(self.feature.dose.values.astype(np.float32))
        
        # trasform time into logscale
        self.log_time = np.log(self.feature.time.values)
        self.t = torch.from_numpy(self.log_time.astype(np.float32))
        
        self.dose_pred_features = feature.iloc[:,-5:]
        self.dose_pred_features = self.dose_pred_features.fillna(0.5)

        scaler = MinMaxScaler()
        self.dose_pred_features = pd.DataFrame(scaler.fit_transform(self.dose_pred_features), columns=self.dose_pred_features.columns)
        #self.dose_pred_features = (self.dose_pred_features-min(self.dose_pred_features))/(max(self.dose_pred_features)-min(self.dose_pred_features))
        #self.dose_pred_features = (self.dose_pred_features-self.dose_pred_features.mean())/self.dose_pred_features.std()
        self.dose_pred_features = torch.from_numpy(self.dose_pred_features.to_numpy().astype(np.float32))

        self.y = torch.from_numpy(self.target.astype(np.float32))
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index], self.d[index], self.t[index],self.dose_pred_features[index], self.y[index] #self.n[index]
        
    def __len__(self):
        return self.n_samples    
        
########################################################
# Data Loaders NDT for pathologies - updated
########################################################
class dataloader_ndt_updated(Dataset):
   
    def __init__(self,feature,target):
        
        self.feature, self.target = feature,target
        
        self.feature = self.feature.drop('COMPOUND_NAME', axis = 1)
        self.x = torch.from_numpy(self.feature.iloc[:,:4096].to_numpy().astype(np.float32))
        self.d = torch.from_numpy(self.feature.dose.values.astype(np.float32))
        
        # trasform time into logscale
        self.log_time = np.log(self.feature.time.values)
        self.t = torch.from_numpy(self.log_time.astype(np.float32))
        
        self.y = torch.from_numpy(self.target.astype(np.float32))
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index], self.d[index], self.t[index], self.y[index] #self.n[index]
        
    def __len__(self):
        return self.n_samples    

########################################################
# Data Loaders for HTHC binary labels
########################################################
class dataloader_Binary_labels(Dataset):
   
    def __init__(self,feature,target):
        
        self.feature, self.target = feature,target
        
        self.feature = self.feature.drop('COMPOUND_NAME', axis = 1)
        self.x = self.feature.iloc[:,:4096].to_numpy().astype(np.float32)
        self.d = self.feature.dose.values.astype(np.float32)
        
        # trasform time into logscale
        self.log_time = np.log(self.feature.time.values)
        self.t = self.log_time.astype(np.float32)
        
        # Even if one animal is toxic, instance is toxic
        self.y_train_binary = np.nansum(self.target, axis = 1).astype(bool) * 1
        #self.y_train_binary = target.sum(axis = 1).astype(bool) * 1
        self.y = self.y_train_binary.astype(np.float32)
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index], self.d[index], self.t[index], self.y[index] #self.n[index]
        
    def __len__(self):
        return self.n_samples    
########################################################
# Data Loaders NDT for pathologies - updated
########################################################
class dataloader_with_unit_dt(Dataset):
   
    def __init__(self,feature,target):
        
        self.feature, self.target = feature,target
        
        self.feature = self.feature.drop('COMPOUND_NAME', axis = 1)
        self.x = torch.from_numpy(self.feature.iloc[:,:4096].to_numpy().astype(np.float32))
        self.d = torch.from_numpy(self.feature.dose.values.astype(np.float32))
        self.d = torch.ones(self.d.shape)
        
        # trasform time into logscale
        self.log_time = np.log(self.feature.time.values)
        self.t = torch.from_numpy(self.log_time.astype(np.float32))
        self.t = torch.ones(self.t.shape)
        
        self.y = torch.from_numpy(self.target.astype(np.float32))
        self.n_samples = self.x.shape[0]
        
    def __getitem__(self, index):

        return self.x[index], self.d[index], self.t[index], self.y[index] #self.n[index]
        
    def __len__(self):
        return self.n_samples  