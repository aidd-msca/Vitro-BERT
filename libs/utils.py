import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import wandb
import torch
import torchmetrics
from torch.distributions.beta import Beta
from scipy.stats import beta
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , recall_score, f1_score
#from netcal.metrics import ECE,ACE

from sklearn.model_selection import GridSearchCV, StratifiedKFold


from libs.data_loaders_utils import get_fold_Clinical_pre_Clinical, get_SIDER_folds, get_invitro_PreClinical_clinical_folds

import scipy.sparse as sp
from torch.distributions import MultivariateNormal
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
import re
# 1. drug_dose_time_finding_data
# 2. wandb_init_model
# 3. open_black_box
# 4. post_training_evaluation
# 5. convert labels into binary labels to compute auc
# 6. 
#7. 
################################################3
# compute metric from prediction
# For 8+12+30 tasks
################################################
import warnings
def check_nan(df):
    if df.isnull().any().any():
        nan_rows, nan_cols = np.where(pd.isnull(df))
        warning_msg = "Warning: NaN values found in DataFrame at rows: {}, columns: {}"
        warning_msg = warning_msg.format(df.loc[nan_rows, "Tasks"].values, df.columns[nan_cols])
        warnings.warn(warning_msg)
        
def compute_metric(y_true, y_prob):

    assert (y_prob >= 0).all() and (y_prob <= 1).all(), "Probability value should be between 0 and 1"
    assert y_true.shape[1] == 50, "The second dimension of y_true should be 50"
    assert y_prob.shape[1] == 50, "The second dimension of y_prob should be 50"
    assert isinstance(y_prob, np.ndarray), "y_prob should be a NumPy array"
    assert isinstance(y_true, np.ndarray), "y_true should be a NumPy array"

    pathological_tasks = ['Cytoplasmic alteration (Basophilic/glycogen depletion)',
                        'Cytoplasmic alteration (Eosinophilic)',
                        'Extramedullary Hematopoiesis',
                        'Hypertrophy, hepatocellular',
                        'Hypertrophy/Hyperplasia',
                        'Increased mitoses',
                        'Infiltration, Mononuclear',
                        'Necrosis',
                        'Pigmentation (pigment deposition)',
                        'Single Cell Necrosis',
                        'Vacuolation',
                        'DILI_binary']

    blood_tasks = ['ALP(IU/L)',
                    'AST(IU/L)',
                    'ALT(IU/L)',
                    'GTP(IU/L)',
                    'TC(mg/dL)',
                    'TG(mg/dL)',
                    'TBIL(mg/dL)',
                    'DBIL(mg/dL)']

    clinical_tasks = ['10001551', '10003481', '10004663', '10005364', '10005630', '10008612',
                    '10008629', '10008635', '10017693', '10019663', '10019670', '10019692',
                    '10019708', '10019717', '10019754', '10019837', '10019842', '10019851',
                    '10020578', '10023126', '10023129', '10024670', '10024690', '10054889',
                    '10059570', '10060795', '10062000', '10067125', 'SIDER_binary',
                    'hepatobiliary_disorders']

    metrics = compute_binary_classification_metrics_MT(y_true = y_true, 
                                                    y_pred_proba = y_prob,
                                                    missing = 'nan')

    metrics.insert(0, 'Tasks', pathological_tasks + blood_tasks + clinical_tasks)
    mean_preformances = {"pathology_mean": metrics[metrics.Tasks.isin(pathological_tasks)].iloc[:,1:].mean(),
                        "blood_mean": metrics[metrics.Tasks.isin(blood_tasks)].iloc[:,1:].mean(),
                        "clinical_mean": metrics[metrics.Tasks.isin(clinical_tasks)].iloc[:,1:].mean(),
                        "combined_all": metrics.iloc[:,1:].mean()}
    mean_preformances = pd.DataFrame(mean_preformances).T
    mean_preformances = mean_preformances.rename_axis('Tasks').reset_index()
    metrics = pd.concat([metrics, mean_preformances], ignore_index=True)
    check_nan(metrics)
    return metrics

################################################3
# Calibration metrics
################################################
def compute_ece(y_true, y_prob, n_bins=10, equal_intervals = True):
    # Calculate bin boundaries
    if equal_intervals == True: # ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    else:                       # ACE
        bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    
    # Calculate bin indices
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
    
    ece = 0
    total_samples = len(y_true)
    
    # Calculate ECE
    for bin_idx in range(n_bins):
        # Filter samples within the bin
        bin_mask = bin_indices == bin_idx
        bin_samples = np.sum(bin_mask)
        
        if bin_samples > 0:
            # Calculate accuracy and confidence for the bin
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_prob[bin_mask])
        
            # Update ECE
            ece += (bin_samples / total_samples) * np.abs(bin_accuracy - bin_confidence)
    
    return ece

#####################################################################################3
# Sklearn model hyp opt
################################################################################

def Repeated_cross_validation(Repeats, K, param_grid, 
                              train_X, train_y, test_X, test_y,
                              train_ids, test_ids,
                              estimator,
                              missing,
                              train_frac):
    
    # Initialize variables to store best hyperparameters and best validation score
    best_hyperparameters = None
    best_val_score = 0

    # mask missing
    if missing == 'nan':
        mask = ~np.isnan(train_y)
        train_y = train_y[mask]
        train_X = train_X[mask]
        train_ids = train_ids[mask]
        
        
    S_val_r = []
    for repeat,seed in enumerate(Repeats):
        cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        
        # Initialize Random Forest Classifier
        classifier = estimator(random_state=seed, class_weight = 'balanced', n_jobs = -1)
        
        # Perform GridSearchCV on training data
        grid_search = GridSearchCV(estimator=classifier, 
                                   param_grid=param_grid, 
                                   scoring='average_precision', 
                                   cv=cv, verbose=1, 
                                   n_jobs=-1
                                   )
        print("train data", train_X.shape, train_y.shape)
        grid_search.fit(train_X, train_y)

        # Get the best hyperparameters and the corresponding model
        best_params = grid_search.best_params_
        best_val_score_repeat = grid_search.best_score_
        S_val_r.append(best_val_score_repeat)

        if repeat == 0:
            best_estimator = grid_search.best_estimator_
            train_true, train_pred, val_true, val_pred = get_train_val_predictions(best_estimator, 
                                                                                   cv,
                                                                                    train_X, 
                                                                                    train_y, 
                                                                                    train_ids)
                                                            
        # Update best hyperparameters and best validation score if the current repeat has a better score
        if best_val_score_repeat > best_val_score:
            best_hyperparameters = best_params
            best_val_score = best_val_score_repeat

    # Train the final model using the best hyperparameters
    best_model = estimator(random_state=42, class_weight = 'balanced',**best_hyperparameters, n_jobs = -1)
    best_model.fit(train_X, train_y)

    if train_frac < 100:
        # Evaluate the best model on the test set
        if missing == 'nan':
            test_mask = ~np.isnan(test_y)
    
        test_predictions = best_model.predict_proba(test_X)[:, 1]
        print("test data", test_y[test_mask].shape, test_predictions[test_mask].shape)
        try:
            test_roc_auc = roc_auc_score(test_y[test_mask], test_predictions[test_mask])
            test_aupr = average_precision_score(test_y[test_mask], test_predictions[test_mask])
        except:
            test_roc_auc = np.nan
            test_aupr = np.nan
    else:
        test_predictions = None
        test_roc_auc = np.nan
        test_aupr = np.nan

    pred = (train_true,train_pred), (val_true,val_pred), test_predictions
    return grid_search.cv_results_, S_val_r, test_roc_auc, test_aupr , pred
#####################################################################################3
# get pred of Sklearn models
################################################################################
def get_train_val_predictions(best_estimator, 
                              cv,
                              train_X, 
                              train_y, 
                              train_ids):
    # get train and validation predictions
    train_true, train_pred = [],[] 
    val_true, val_pred = [],[] 
    train_smiles , val_smiles = [],[]

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_X, train_y)):
        
        # get train, val labels and SMILES
        x_train, y_train = train_X[train_idx], train_y[train_idx]
        x_val, y_val = train_X[val_idx], train_y[val_idx]
        val_smiles.extend(train_ids[val_idx].tolist())

        # get val predictions
        best_estimator.fit(x_train, y_train)
        val_pred_fold = best_estimator.predict_proba(x_val)[:, 1]
        val_pred.extend(val_pred_fold.tolist())
        val_true.extend(y_val.tolist())

        # get train_pred
        if fold == 4:
            train_smiles = train_ids[train_idx].tolist()
            train_pred = best_estimator.predict_proba(x_train)[:, 1]
    
    # convert into dataframe
    val_true = pd.DataFrame(val_true)
    val_pred = pd.DataFrame(val_pred)
    train_true = pd.DataFrame(y_train)
    train_pred = pd.DataFrame(train_pred)

    val_true.insert(0, 'SMILES', val_smiles)
    val_pred.insert(0, 'SMILES', val_smiles)
    train_true.insert(0, 'SMILES', train_smiles)
    train_pred.insert(0, 'SMILES', train_smiles)
    return train_true, train_pred, val_true, val_pred
#####################################################################################3
# Compute class weights
################################################################################
def compute_neg_to_positive_ratio(complete_data):
    '''
    Negatice_to_positive eatio to balance within each class
    '''
    active = (complete_data == 1).sum(axis = 0)
    inactive = (complete_data == 0).sum(axis = 0)
    target_weights = inactive / active
    targets = pd.DataFrame(target_weights).reset_index()
    targets.columns = ['Targets','weights']
    return targets

def compute_target_weights(complete_data):
    '''
    Compute target weights to balane preclinicala and clincial
    '''
    n_preclinical = complete_data[~complete_data.DILI_binary.isnull()].shape[0]
    n_clinical = complete_data[~complete_data.hepatobiliary_disorders.isnull()].shape[0]

    N = n_preclinical + n_clinical
    Np = complete_data.count(axis = 0)

    target_weights = N / Np
    target_weights = pd.DataFrame(target_weights).reset_index()
    target_weights.columns = ['Targets','weights']
    return target_weights

#####################################################################################3
# sort preclinical clincal
################################################################################
def sort_dataframe_as_per_toxic_values(y_true, y_pred):
    # Strategy: divide and concur 
    preclinical_data = y_true[~y_true.DILI_binary.isnull()].reset_index(drop = True)
    row_sums = preclinical_data[preclinical_data == 1].sum(axis = 1)
    col_sums = preclinical_data[preclinical_data == 1].sum(axis = 0)
    # Sort rows based on row sums, and column sum
    preclinical_data = preclinical_data.iloc[row_sums.sort_values(ascending=False).index]
    preclinical_data = preclinical_data[col_sums.sort_values(ascending=False).index]
    SMILES = preclinical_data.pop('SMILES')
    preclinical_data.insert(0,'SMILES',SMILES)

    # clinical data 
    clinical_data = y_true[y_true.DILI_binary.isnull()].reset_index(drop = True)
    row_sums = clinical_data[clinical_data == 1].sum(axis = 1)
    col_sums = clinical_data[clinical_data == 1].sum(axis = 0)
    # Sort rows based on row sums, and column sum
    clinical_data = clinical_data.iloc[row_sums.sort_values(ascending=False).index]
    clinical_data = clinical_data[col_sums.sort_values(ascending=False).index]
    SMILES = clinical_data.pop('SMILES')
    clinical_data.insert(0,'SMILES',SMILES)
    y_true = pd.concat([preclinical_data, clinical_data], axis = 0).reset_index(drop = True)

    # sort predictions
    y_pred = y_pred.set_index("SMILES")
    y_pred = y_pred.reindex(index = y_true['SMILES']).reset_index()
    y_pred = y_pred[y_true.columns]
    
    y_true.drop('SMILES', axis = 1, inplace = True)
    y_pred.drop('SMILES', axis = 1, inplace = True)
    return y_true, y_pred
#####################################################################################3
# get predictions SIDER SOC
################################################################################
def get_predictions_SIDER_SOC(dataloader, model): 
        device = torch.device('cuda')
        model = model.eval()
        model = model.to(device)  
        preds, targets = [], []
        for batch in dataloader:
            smiles, batch_targets = batch
            smiles = [[SMILES] for SMILES in smiles]

            batch_preds = model(smiles)
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)
            targets.extend(batch_targets.cpu().detach().tolist())
        return targets, preds

def compute_metrics_SIDER_SOC(targets, preds):   
    
    num_tasks = len(targets[0])
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):

        for j in range(len(preds)):
            if targets[j][i] != -1:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])
                
    roc_score, aupr_score, f1, mcc_score = [], [],[], []
    for i in range(num_tasks):
        try:
            roc_score.append(roc_auc_score(valid_targets[i], valid_preds[i]))
        except:
            roc_score.append(np.nan)
            print(f"class {i}")
        try:
            precision, recall, _ = precision_recall_curve(valid_targets[i], valid_preds[i])
            aupr_score.append(auc(recall, precision))
        except:
            aupr_score.append(np.nan)

        threshold = 0.5
        hard_preds = [1 if p > threshold else 0 for p in valid_preds[i]]
        mcc_score.append(matthews_corrcoef(valid_targets[i], hard_preds))
        f1.append(f1_score(valid_targets[i], hard_preds))

    return roc_score,aupr_score, mcc_score, f1

#####################################################################################3
# get predictions
################################################################################
def get_MF_model_predictions(model,args, val_dataloader):
        device = torch.device('cuda')
        model = model.eval()
        model = model.to(torch.device("cuda"))

        ytrue_tensor = torch.full((args.val_num_mols,args.num_tasks),-1.0, device=torch.device('cuda'))
        ypred_tensor = torch.full((args.val_num_mols,args.num_tasks),-1.0, device=torch.device('cuda'))
        for batch in val_dataloader:
                mol_indices, smiles, task_indices, targets = batch
                smiles = [[SMILES] for SMILES in smiles]
                mol_indices = mol_indices.to(device)
                task_indices = task_indices.to(device)
                targets = targets.to(device)
                
                # get pred
                preds, _ = model(mol_indices, smiles, task_indices)
                # save pred
                ytrue_tensor[mol_indices,:] = targets.to(torch.float)
                ypred_tensor[mol_indices,:] = preds
        return ytrue_tensor, ypred_tensor
        
#####################################################################################3
# Compute metrics
################################################################################

def compute_metrics_MF(ytrue_tensor, ypred_tensor):   
    
    targets = torch.nan_to_num(ytrue_tensor, nan = -1)
    targets = targets.cpu().detach().tolist()
    preds = ypred_tensor.cpu().detach().tolist()
    num_tasks = ytrue_tensor.shape[1]
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):

        for j in range(len(preds)):
            if targets[j][i] != -1:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])
                
    roc_score, aupr_score, f1, mcc_score = [], [],[], []
    for i in range(num_tasks):
        try:
            roc_score.append(roc_auc_score(valid_targets[i], valid_preds[i]))
        except:
            roc_score.append(np.nan)
            print(f"class {i}")
        try:
            precision, recall, _ = precision_recall_curve(valid_targets[i], valid_preds[i])
            aupr_score.append(auc(recall, precision))
        except:
            aupr_score.append(np.nan)

        threshold = 0.5
        hard_preds = [1 if p > threshold else 0 for p in valid_preds[i]]
        mcc_score.append(matthews_corrcoef(valid_targets[i], hard_preds))
        f1.append(f1_score(valid_targets[i], hard_preds))

    return roc_score,aupr_score, mcc_score, f1
#####################################################################################3
# get pretrained model
################################################################################
def pretrained_model(model, args):
    debug = info = print

    # Load model and args

    state = torch.load(args.pretrained_dir + 'model.pt', map_location=lambda storage, loc: storage)
    loaded_state_dict = state["state_dict"]

    # Remove last layer
    loaded_state_dict = {key: value for key, value in list(loaded_state_dict.items())[:-2]}

    model = model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r"(encoder\.encoder\.)([Wc])", loaded_param_name) and not args.reaction_solvent:
            param_name = loaded_param_name.replace("encoder.encoder", "encoder.encoder.0")
        elif re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)
    return model
#####################################################################################3
# get_tox21_model_pred
################################################################################
def get_tox21_model_pred(model, selected_dataloader, config):
    
    y_true_list = []
    y_pred_list = []
    val_w = []
    for batch in selected_dataloader:
        x, y, w = batch
        y_hat = model(x)
        y_prob = torch.sigmoid(y_hat)

        y_true_list.append(y.cpu())
        y_pred_list.append(y_prob.cpu())
        val_w.append(w.cpu())
    
    y = torch.cat(y_true_list, dim=0)
    y_prob = torch.cat(y_pred_list, dim=0)
    val_w = torch.cat(val_w, dim=0)

    selected_col = config["selected_tasks"]
    #drug_information = pd.DataFrame({'drug_index':metadata.index})

    y_true = pd.DataFrame(y.detach().numpy())
    y_prob = pd.DataFrame(y_prob.detach().numpy())
    val_w = pd.DataFrame(val_w.detach().numpy())

    y_true.columns = selected_col
    y_prob.columns = selected_col

    y_true['fold'],y_prob['fold'] = config["fold"], config["fold"]
    return y_true,y_prob, val_w

#####################################################################################3
# get_tox21_model_pred
################################################################################

def compute_classification_metric_Tox21(y_true, y_pred_proba, val_w, threshold=0.5):
    """
    Compute various metrics for binary classification.
    
    Parameters:
        y_true (array-like): Binary labels (0 or 1).
        y_pred_proba (array-like): Predictive probabilities for the positive class.
        threshold (float, optional): Threshold value for classification. Default is 0.5.
    
   Returns:
        pandas.DataFrame: DataFrame containing the computed metrics for each task (accuracy, ROC AUC, average precision, MCC, F1-score, random precision, gain in average precision).
    """

    num_tasks = val_w.shape[1]  # Get the number of tasks

    metrics_list = []

    for i in range(num_tasks):
        # Apply masking
        y_true_task = y_true[:, i]
        y_true_task = y_true_task[val_w[:,i] != 0]

        y_pred_proba_task = y_pred_proba[:, i]
        y_pred_proba_task = y_pred_proba_task[val_w[:,i] != 0]
        

        y_pred_task = (y_pred_proba_task >= threshold).astype(int)

        metrics_task = {}
        metrics_task['accuracy'] = accuracy_score(y_true_task, y_pred_task)
        try:
            metrics_task['roc_auc'] = roc_auc_score(y_true_task, y_pred_proba_task)
        except:
            metrics_task['roc_auc'] = np.nan
        try:
            metrics_task['average_precision'] = average_precision_score(y_true_task, y_pred_proba_task)
        except:
            metrics_task['average_precision'] = np.nan

        metrics_task['mcc'] = matthews_corrcoef(y_true_task, y_pred_task)
        metrics_task['f1_score'] = f1_score(y_true_task, y_pred_task)

        positive_ratio = np.mean(y_true_task)
        random_precision = positive_ratio  # Precision for random prediction
        metrics_task['gain_in_average_precision'] = (metrics_task['average_precision'] - random_precision) / (1 - random_precision)
        metrics_task['random_precision'] = random_precision
        metrics_task['random_precision']
        metrics_list.append(metrics_task)

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df
#####################################################################################3
# compute compute_weighted_metrics
######################################################################################
def compute_weighted_metrics(y_true, y_pred, w):
    num_tasks = y_true.shape[1]  # Get the number of tasks
    aupr_list, roc_auc_list = [], []
    AUPR = torchmetrics.AveragePrecision(task="binary")
    AUROC = torchmetrics.AUROC(task="binary")
    
    for i in range(num_tasks):
        y_task = y_true[:, i]
        y_task = y_task[w[:,i] != 0]

        ypred_task = y_pred[:, i]
        ypred_task = ypred_task[w[:,i] != 0]

        try:
            aupr = AUPR(ypred_task, y_task.type(torch.int))
        except:
            aupr = torch.tensor([0.0])
        aupr_list.append(aupr.item())

        try:
            roc_auc = AUROC(ypred_task, y_task.type(torch.int))
        except:
            roc_auc = torch.tensor([0.5])
        roc_auc_list.append(roc_auc.item())

    return torch.tensor(aupr_list).nanmean().item(), torch.tensor(roc_auc).nanmean().item()

#####################################################################################3
# invitr_emp_prior
######################################################################################

def invitr_emp_prior(feature_target, config):
    device = torch.device('cuda')
    feature_target = feature_target[feature_target.loc[:,'104543_level_5.0':'cytoHepG224BE_level_5.5'].count(axis = 1) >0]

    selected_rows = feature_target.index
    x = sp.load_npz(config['FP_file'])
    x = x[selected_rows,:].toarray().astype(np.float32)
    x = torch.tensor(x)

    prior_mean = torch.mean(x, dim=0)
    prior_cov = torch.mm((x - prior_mean).T, x - prior_mean) / (x.shape[0] - 1)

    #cov_matrix = self.prior_cov.to(device) * self.alpha + self.epsilon * torch.eye(weight_size, device=device)
    cov_matrix = prior_cov * config["alpha"] + config["epsilon"] * torch.eye(config["input_dim"])
    cov_dist = MultivariateNormal(torch.zeros(config["input_dim"],device=device), cov_matrix.to(device))
    return cov_dist

#####################################################################################3
# SIDER empirical prior
######################################################################################
def SIDER_emp_prior(feature_target, config):
    device = torch.device('cuda')
    feature_target = feature_target[~feature_target.hepatobiliary_disorders.isnull()]
    feature_target = feature_target.loc[:,'10001551':'hepatobiliary_disorders']

    selected_rows = feature_target.index
    x = sp.load_npz(config['FP_file'])
    x = x[selected_rows,:].toarray().astype(np.float32)
    x = torch.tensor(x)

    prior_mean = torch.mean(x, dim=0)
    prior_cov = torch.mm((x - prior_mean).T, x - prior_mean) / (x.shape[0] - 1)

    #cov_matrix = self.prior_cov.to(device) * self.alpha + self.epsilon * torch.eye(weight_size, device=device)
    cov_matrix = prior_cov * config["alpha"] + config["epsilon"] * torch.eye(config["input_dim"])
    cov_dist = MultivariateNormal(torch.zeros(config["input_dim"],device=device), cov_matrix.to(device))
    return cov_dist
#####################################################################################3
# Compute compute_binary_classification_metrics
######################################################################################

def class_weights_for_SIDER(config):
    
    #if config['data_modality'] == 'SIDER':
    #    feature_target_train, feature_target_val = get_SIDER_folds(fold = 0, file = config['target_file'])
    #if config['data_modality'] == 'invitro_PreClinical_clinical':
    feature_target_train, feature_target_val = get_invitro_PreClinical_clinical_folds(fold = 0, file = config['target_file'])
    all_data = pd.concat([feature_target_train, feature_target_val]).reset_index(drop = True)

    if config['data_modality'] == 'invitro':  
        Y  = all_data.loc[:,'104543_level_5.0':'cytoHepG224BE_level_5.5'].replace(-1, np.nan)

    else:
        selected_col = config['selected_tasks']
        Y = all_data[selected_col].replace(-1, np.nan)
    # Compute class weights
    
    class_weights = []
    try:
        for column in Y.columns:
            postive_class = (Y[column] == 1).sum()
            negative_class = (Y[column] == 0).sum()
            weights = negative_class / postive_class
            class_weights.append(weights.item())
    except:
        postive_class = (Y == 1).sum()
        negative_class = (Y == 0).sum()
        weights = negative_class / postive_class
        class_weights.append(weights.item())


    class_weights = torch.FloatTensor(class_weights)
    return class_weights

#####################################################################################3
# Compute compute_binary_classification_metrics: Multitask
######################################################################################
def compute_AUPR_by_ignoring_missing_values(y_true, y_pred_proba):

    y_true, y_pred_proba = y_true.cpu().detach().numpy(), y_pred_proba.cpu().detach().numpy()
    #print(y_true.shape, y_pred_proba.shape)
    num_tasks = y_true.shape[1]  # Get the number of tasks
    aupr_list  = []

    for i in range(num_tasks):
        y_true_task = y_true[:, i]
        y_pred_proba_task = y_pred_proba[:, i]

        # Apply masking
        mask = (y_true_task != -1)
        y_true_task = y_true_task[mask]
        y_pred_proba_task = y_pred_proba_task[mask]
        #try:
        aupr = average_precision_score(y_true_task, y_pred_proba_task)
        #except:
        #    aupr = torch.tensor([0.0])
        aupr_list.append(aupr)
    return np.nanmean(aupr_list)
#####################################################################################3
# Compute compute_binary_classification_metrics: Multitask
######################################################################################
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score

def prob_to_labels(pred, threshold):
	    return (pred >= threshold).astype('int')

def compute_binary_classification_metrics_MT(y_true, y_pred_proba, 
                                             missing):
    """
    Compute various metrics for binary classification.
    
    Parameters:
        y_true (array-like): Binary labels (0 or 1).
        y_pred_proba (array-like): Predictive probabilities for the positive class.
        threshold (float, optional): Threshold value for classification. Default is 0.5.
    
   Returns:
        pandas.DataFrame: DataFrame containing the computed metrics for each task (accuracy, ROC AUC, average precision, MCC, F1-score, random precision, gain in average precision).
    """
    try:
        num_tasks = y_true.shape[1]  # Get the number of tasks
    except:
        num_tasks = 1
    metrics_list = []

    for i in range(num_tasks):
        if num_tasks > 1:
            y_true_task = y_true[:, i]
            y_pred_proba_task = y_pred_proba[:, i]
        else:
            y_true_task = y_true
            y_pred_proba_task = y_pred_proba
            
        # Apply masking
        if missing == 'nan':
            mask = ~np.isnan(y_true_task)
        if missing == -1:
            mask = (y_true_task != -1)

        y_true_task = y_true_task[mask]
        y_pred_proba_task = y_pred_proba_task[mask]


        metrics_task = {}
        try:
            # ROC AUC
            fpr, tpr, th = roc_curve(y_true_task, y_pred_proba_task)
            metrics_task['roc_auc'] = auc(fpr, tpr)

            # Balanced accuracy
            balanced_accuracy = (tpr + (1 - fpr)) / 2
            metrics_task['balanced_acc'] = np.max(balanced_accuracy)
            
            # sensitivity, specificity
            optimal_threshold_index = np.argmax(balanced_accuracy)
            optimal_threshold = th[optimal_threshold_index]
            metrics_task['sensitivity'] = tpr[optimal_threshold_index]
            metrics_task['specificity'] = 1 - fpr[optimal_threshold_index]

        except:
            metrics_task['roc_auc'] = np.nan
            metrics_task['sensitivity']= np.nan
            metrics_task['specificity']= np.nan
        try:
            precision, recall, thresholds = precision_recall_curve(y_true_task, y_pred_proba_task)
            metrics_task['AUPR'] = auc(recall, precision)
            f1 = [f1_score(y_true_task, prob_to_labels(y_pred_proba_task, t)) for t in thresholds]
            metrics_task['f1_score'] = np.max(f1)

            metrics_task['average_precision'] = average_precision_score(y_true_task, y_pred_proba_task)
        except:
            metrics_task['AUPR'] = np.nan
            metrics_task['f1_score'] = np.nan
        
        try:
            # calibration metrics
            metrics_task["ECE"] = compute_ece(y_true_task, y_pred_proba_task, n_bins=10, equal_intervals = True)
            metrics_task["ACE"] = compute_ece(y_true_task, y_pred_proba_task, n_bins=10, equal_intervals = False)
        except:
            metrics_task['ECE'] = np.nan
            metrics_task['ACE'] = np.nan

        metrics_list.append(metrics_task)
    metrics_df = pd.DataFrame(metrics_list)
    col = ['balanced_acc', 'f1_score','specificity','sensitivity', 
           'roc_auc','AUPR', 'average_precision','ECE','ACE']
    
    return metrics_df[col]
#####################################################################################3
# Compute compute_binary_classification_metrics: Single_Task
######################################################################################

def compute_binary_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute various metrics for binary classification.
    
    Parameters:
        y_true (array-like): Binary labels (0 or 1).
        y_pred_proba (array-like): Predictive probabilities for the positive class.
        threshold (float, optional): Threshold value for classification. Default is 0.5.
    
    Returns:
        dict: Dictionary containing the computed metrics (accuracy, ROC AUC, average precision, MCC, F1-score, random precision, gain in average precision).
    """
    # Apply masking

    mask = (y_true != - 1)
    y_true = y_true[mask]
    y_pred_proba = y_pred_proba[mask]

    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['average_precision'] = auc(recall, precision)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    
    positive_ratio = np.mean(y_true)
    random_precision = positive_ratio  # Precision for random prediction
    
    # This formula is similar to cohens-kappa score, that is used to compare two classifiers
    # https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
    gain_in_average_precision = (metrics['average_precision'] - random_precision) / (1 - random_precision)
    metrics['random_precision'] = random_precision
    metrics['gain_in_average_precision'] = gain_in_average_precision
    
    return metrics

#####################################################################################3
# Get Model predictions: Single_Task
#####################################################################################3
def get_predictions_MF(model, selected_dataloader, config):
    
    selected_col = config["selected_tasks"]
    y_prob = []
    y_true = []

    with torch.no_grad():
        for drug_idx, side_effect_idx, rating, drug_attributes in selected_dataloader:
            predictions = model(drug_idx, side_effect_idx, drug_attributes)
            y_prob.extend(predictions.detach().tolist())  # Convert tensor to list
            y_true.extend(rating.detach().tolist())  # Convert tensor to list

    y_true = np.array(y_true).reshape(-1,len(selected_col))
    y_prob = np.array(y_prob).reshape(-1,len(selected_col))
    

    y_true = pd.DataFrame(y_true)
    y_prob = pd.DataFrame(y_prob)

    y_true.columns = selected_col
    y_prob.columns = selected_col

    y_true['fold'],y_prob['fold'] = config["fold"], config["fold"]
    return y_true,y_prob
#####################################################################################3
# Get Model predictions: Single_Task
######################################################################################
def get_model_predictions_MT(model, selected_dataloader, config, ids):
    
    y_true_list = []
    y_pred_list = []
    for batch in selected_dataloader:
        x, y = batch
        y_hat = model(x)
        y_true_list.append(y.cpu())
        y_pred_list.append(y_hat.cpu())
    
    y = torch.cat(y_true_list, dim=0)
    y_hat = torch.cat(y_pred_list, dim=0)

    if config["num_of_tasks"] > 1:
        y = pd.DataFrame(y.cpu().detach().numpy())
        y_hat = pd.DataFrame(y_hat.cpu().detach().numpy())
        y.columns = config['selected_tasks']
        y_hat.columns = config['selected_tasks']
    else:
        y = pd.DataFrame({config["selected_tasks"]: y.cpu().detach().numpy()})
        y_hat = pd.DataFrame({config["selected_tasks"]: y_hat.cpu().detach().numpy().reshape(-1)})

    y.insert(0,'SMILES', ids)
    y_hat.insert(0,'SMILES', ids)
    return y,y_hat

#####################################################################################3
# Get Model predictions: Multi_Task
######################################################################################
def get_model_predictions(model, selected_dataloader, metadata, config):
    
    x, y = next(iter(selected_dataloader))
    y_true = y.detach().numpy()
    y_hat = model(x)
    y_prob = torch.sigmoid(y_hat)
    y_prob = y_prob.detach().numpy()
    return y_true,y_prob

#####################################################################################3
# post_training_evaluation_TG_JNJ_MiniTox
######################################################################################
class post_training_evaluation_TG_JNJ_MiniTox(object):
    def __init__(self, config, title):
        self.tasks = config['num_of_tasks'] 
        self.title = title
        self.subjective_tox_thershold = config["subjective_tox_thershold"]
        self.eps = 1e-5
        
        self.task_name_list =['Apop','BDH','GlyD','Ephi','GlyA','EMH','Fibro','HTHC',
                                'HTHP','IM','InMC','NecZ','Pigmen','ScNec','Vacu']
        
    def ELL_EoM_EoV(self, model, given_dataloader,config):
        
        x, d , t, dpf, self.y = next(iter(given_dataloader))
        alphas_betas = model(x,d,t,dpf)
        self.alphas, self.betas = alphas_betas[:,0:self.tasks], alphas_betas[:,self.tasks:]
        
        mask = torch.isnan(self.y)
        y_without_nan = self.y.nan_to_num(0.5)
        y_true_clipped = y_without_nan.clip(self.eps, 1 - self.eps)
        
        dist =  Beta(torch.unsqueeze(self.alphas, dim = 2),torch.unsqueeze(self.betas, dim = 2))
        self.LL = dist.log_prob(y_true_clipped)
        self.LL = self.LL[~mask]
        ELL = self.LL.mean()
        
        # EoM calculation
        pred_mean, response_mean = Beta(self.alphas, self.betas).mean, self.y.nanmean(axis =2)
        mask = torch.isnan(response_mean)
        EoM = torch.abs(pred_mean[~mask] - response_mean[~mask]).mean()
        
        # EoV calculation
        pred_var = Beta(self.alphas, self.betas).variance.detach().numpy()
        response_var = np.nanvar(self.y.detach().numpy(), axis = 2)
        mask = np.isnan(response_var)

        EoV = np.abs(pred_var[~mask] - response_var[~mask]).mean()
        observed_binary_labels, pred_prob_1 = self.compute_binary_auc()

        if config["class_dependent_Log_likelihood"] == 1:
            self.class_dependent_Log_likelihood()
            
            
        return ELL,EoM,EoV, observed_binary_labels, pred_prob_1, self.alphas, self.betas
    
    def class_ELL(self, y, class_label):
        mask = y.eq(class_label)
        class_ELL = torch.masked_select(self.LL, mask).mean()
        return class_ELL
                
    def class_dependent_Log_likelihood(self):
        toxicity_class = [0,0.2,0.4,0.6,0.8,1]
        
         # to handle sparsity
        mask = torch.isnan(self.y)
        
        for class_label in toxicity_class:
            ELL_class = self.class_ELL(self.y[~mask],class_label)
            #wandb.log({f'{self.title}_ELL_fold_class_{class_label}': np.round(ELL_class.item(),2)})
    
    def compute_binary_auc(self):

        # get binary labels
        observed_binary_labels = self.get_binary_labels(self.y)
        pred_prob_1 = self.pred_tox_prob()
        return observed_binary_labels, pred_prob_1
    
    def get_binary_labels(self,lables):
        '''
        labels format: (ndt, findings, animals)
        '''
        binary_labels = [[] for i in range(lables.shape[0])]
        for finding in range(lables.shape[1]):
            for comb in range(lables.shape[0]):

                if lables[comb,finding,:].isnan().all() == True:
                    binary_labels[comb].append(np.nan)
                else:
                    y_finding = lables[comb,finding,:]
                    #total_tested_animals = torch.count_nonzero(~torch.isnan(y_finding))
                    tox_animals = torch.count_nonzero(y_finding[~torch.isnan(y_finding)])
                    #tox_percentage = tox_animals / total_tested_animals

                    #if tox_percentage >= (self.subjective_tox_thershold):
                    if tox_animals >= self.subjective_tox_thershold:
                        binary_labels[comb].append(1)
                    else:
                        binary_labels[comb].append(0)
        binary_labels = np.array(binary_labels)
        return binary_labels
    
    def pred_tox_prob(self):
        '''
        return shape: (ndt) * task
        '''
        a, b = self.alphas.detach().numpy(), self.betas.detach().numpy() 
        prob = 1 - beta.cdf(x = 0.1, a = a, b = b)
        return prob

def Get_fold_Tg_jnj_MiniTox_HTHC(fold):
    '''
    fetch prepcoessed data, 
    replace toxic catagories with numeric categories,
    return training and val folds
    '''

    features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/TG_JNJ_MiniTox_with_dose_pred.pkl')    
    features_target = features_target[features_target.finding == 'Hypertrophy, hepatocellular']
    train = features_target[features_target.kfold != fold].reset_index(drop = True).drop(['kfold'],axis =1)
    validation = features_target[features_target.kfold == fold].reset_index(drop = True).drop(['kfold'],axis =1)

    train_drug_dose_time_finding_combinations = train.iloc[:,np.r_[0,4097:train.shape[1]]]
    val_drug_dose_time_finding_combinations = validation.iloc[:,np.r_[0,4097:validation.shape[1]]]

    # training data
    x_cols = train.iloc[:,:4100].columns.values.tolist() + train.loc[:,"HDP_Halflife_hr":"HDP_r_vivo_CL_mL/min/kg"].columns.values.tolist()
    x_train = train[x_cols]
    x_train.drop_duplicates(inplace = True)

    y_train = train.iloc[:,4098:]
    num_findings = y_train.finding.nunique()
    num_combinatins = x_train.shape[0]
    max_num_animals = y_train.loc[:,'A1':'A78'].shape[1]

    y_train = y_train.loc[:,'A1':'A78']
    y_train = y_train.to_numpy().reshape(num_combinatins,num_findings,max_num_animals)

    # validation data
    x_cols = validation.iloc[:,:4100].columns.values.tolist() + validation.loc[:,"HDP_Halflife_hr":"HDP_r_vivo_CL_mL/min/kg"].columns.values.tolist()
    x_val = validation[x_cols]
    x_val.drop_duplicates(inplace = True)
    y_val = validation.iloc[:,4098:]

    num_findings = y_val.finding.nunique()
    num_combinatins = x_val.shape[0]
    max_num_animals = y_val.loc[:,'A1':'A78'].shape[1]

    y_val = y_val.loc[:,'A1':'A78']
    y_val = y_val.to_numpy().reshape(num_combinatins,num_findings,max_num_animals)
    return x_train, y_train, x_val, y_val, train_drug_dose_time_finding_combinations, val_drug_dose_time_finding_combinations
'''
    
for i in range(5):
    x_train, y_train, x_val, y_val, train_ndtp, val_ndtp = Get_fold_Tg_jnj_MiniTox_HTHC(fold = i)       
    # store dataframes files
    x_train.to_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/x_train'+'_fold_' + str(i) + '.pkl')
    x_val.to_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/x_val'+'_fold_' + str(i) + '.pkl')
    train_ndtp.to_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/train_ndtp'+'_fold_' + str(i) + '.pkl')
    val_ndtp.to_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/val_ndtp'+'_fold_' + str(i) + '.pkl')

    # store numpy arrays 
    np.save('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/y_train'+'_fold_' + str(i) +'.npy', y_train)
    np.save('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/y_val'+'_fold_' + str(i) +'.npy', y_val)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, train_ndtp.shape, val_ndtp.shape)

'''

def Get_fold_Tg_jnj_MiniTox(fold):
    '''
    fetch prepcoessed data, 
    replace toxic catagories with numeric categories,
    return training and val folds
    '''

    features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/TG_JNJ_MiniTox_with_dose_pred.pkl')    
    train = features_target[features_target.kfold != fold].reset_index(drop = True).drop(['kfold'],axis =1)
    validation = features_target[features_target.kfold == fold].reset_index(drop = True).drop(['kfold'],axis =1)

    train_drug_dose_time_finding_combinations = train.iloc[:,np.r_[0,4097:train.shape[1]]]
    val_drug_dose_time_finding_combinations = validation.iloc[:,np.r_[0,4097:validation.shape[1]]]

    # training data
    x_cols = train.iloc[:,:4100].columns.values.tolist() + train.loc[:,"HDP_Halflife_hr":"HDP_r_vivo_CL_mL/min/kg"].columns.values.tolist()
    x_train = train[x_cols]
    x_train.drop_duplicates(inplace = True)

    y_train = train.iloc[:,4098:]
    num_findings = y_train.finding.nunique()
    num_combinatins = x_train.shape[0]
    max_num_animals = y_train.loc[:,'A1':'A78'].shape[1]

    y_train = y_train.loc[:,'A1':'A78']
    y_train = y_train.to_numpy().reshape(num_combinatins,num_findings,max_num_animals)

    # validation data
    x_cols = validation.iloc[:,:4100].columns.values.tolist() + validation.loc[:,"HDP_Halflife_hr":"HDP_r_vivo_CL_mL/min/kg"].columns.values.tolist()
    x_val = validation[x_cols]
    x_val.drop_duplicates(inplace = True)
    y_val = validation.iloc[:,4098:]

    num_findings = y_val.finding.nunique()
    num_combinatins = x_val.shape[0]
    max_num_animals = y_val.loc[:,'A1':'A78'].shape[1]

    y_val = y_val.loc[:,'A1':'A78']
    y_val = y_val.to_numpy().reshape(num_combinatins,num_findings,max_num_animals)
    
    ''''
    for i in range(5):
        x_train, y_train, x_val, y_val, train_ndtp, val_ndtp = Get_fold_Tg_jnj_MiniTox(fold = i)       
        # store dataframes files
        x_train.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/x_train'+'_fold_' + str(i) + '.pkl')
        x_val.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/x_val'+'_fold_' + str(i) + '.pkl')
        train_ndtp.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/train_ndtp'+'_fold_' + str(i) + '.pkl')
        val_ndtp.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/val_ndtp'+'_fold_' + str(i) + '.pkl')

        # store numpy arrays 
        np.save('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/y_train'+'_fold_' + str(i) +'.npy', y_train)
        np.save('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/y_val'+'_fold_' + str(i) +'.npy', y_val)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, train_ndtp.shape, val_ndtp.shape)
    '''
    return x_train, y_train, x_val, y_val, train_drug_dose_time_finding_combinations, val_drug_dose_time_finding_combinations

def get_data_tggates_jnj_MiniTox(fold):
    x_train = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/x_train_fold_'+ str(fold)+'.pkl')
    x_val = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/x_val_fold_'+ str(fold)+'.pkl')
    train_ndtp = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/train_ndtp_fold_'+ str(fold)+'.pkl')
    val_ndtp = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/val_ndtp_fold_'+ str(fold)+'.pkl')

    y_train = np.load('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/y_train_fold_'+ str(fold)+'.npy')
    y_val= np.load('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj_MiniTox/y_val_fold_'+ str(fold)+'.npy')
    
    return x_train, y_train, x_val, y_val, train_ndtp, val_ndtp

def get_data_TG_jnj_MiniTox_HTHC(fold):
    x_train = pd.read_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/x_train_fold_'+ str(fold)+'.pkl')
    x_val = pd.read_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/x_val_fold_'+ str(fold)+'.pkl')
    train_ndtp = pd.read_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/train_ndtp_fold_'+ str(fold)+'.pkl')
    val_ndtp = pd.read_pickle('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/val_ndtp_fold_'+ str(fold)+'.pkl')

    y_train = np.load('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/y_train_fold_'+ str(fold)+'.npy')
    y_val= np.load('/extra/arslan_data_repository/pre_processed/TG_JNJ_MiniTox_HTHC/y_val_fold_'+ str(fold)+'.npy')
    
    return x_train, y_train, x_val, y_val, train_ndtp, val_ndtp

def get_data(fold):
    x_train = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/x_train_fold_'+ str(fold)+'.pkl')
    x_val = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/x_val_fold_'+ str(fold)+'.pkl')
    train_ndtp = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/train_ndtp_fold_'+ str(fold)+'.pkl')
    val_ndtp = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/val_ndtp_fold_'+ str(fold)+'.pkl')

    y_train = np.load('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/y_train_fold_'+ str(fold)+'.npy')
    y_val= np.load('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/y_val_fold_'+ str(fold)+'.npy')
    
    return x_train, y_train, x_val, y_val, train_ndtp, val_ndtp

def Get_fold_Tg_jnj(fold):
    '''
    fetch prepcoessed data, 
    replace toxic catagories with numeric categories,
    return training and val folds
    '''

    features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/tggate_jnj_folds.pkl')
    train = features_target[features_target.kfold != fold].reset_index(drop = True).drop(['kfold'],axis =1)
    validation = features_target[features_target.kfold == fold].reset_index(drop = True).drop(['kfold'],axis =1)

    train_drug_dose_time_finding_combinations = train.iloc[:,np.r_[0,4097:train.shape[1]]]
    val_drug_dose_time_finding_combinations = validation.iloc[:,np.r_[0,4097:validation.shape[1]]]
    del features_target

    # training data
    x_train = train.iloc[:,:4100]
    x_train.drop('mol_set', axis =1, inplace = True)
    x_train.drop_duplicates(inplace = True)
    y_train = train.iloc[:,4098:]

    num_findings = y_train.finding.nunique()
    num_combinatins = x_train.shape[0]
    max_num_animals = y_train.loc[:,'A1':'A78'].shape[1]

    y_train.drop(['dose','time','finding'], axis =1 , inplace = True)
    y_train = y_train.to_numpy().reshape(num_combinatins,num_findings,max_num_animals)

    # validation data
    x_val = validation.iloc[:,:4100]
    x_val.drop('mol_set', axis =1, inplace = True)
    x_val.drop_duplicates(inplace = True)
    y_val = validation.iloc[:,4098:]

    num_findings = y_val.finding.nunique()
    num_combinatins = x_val.shape[0]
    max_num_animals = y_val.loc[:,'A1':'A78'].shape[1]

    y_val.drop(['dose','time','finding'], axis =1 , inplace = True)
    y_val = y_val.to_numpy().reshape(num_combinatins,num_findings,max_num_animals)
    '''
    for i in range(5):
        x_train, y_train, x_val, y_val, train_ndtp, val_ndtp = Get_fold_Tg_jnj(fold = i)       
        # store dataframes files
        x_train.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/x_train'+'_fold_' + str(i) + '.pkl')
        x_val.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/x_val'+'_fold_' + str(i) + '.pkl')
        train_ndtp.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/train_ndtp'+'_fold_' + str(i) + '.pkl')
        val_ndtp.to_pickle('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/val_ndtp'+'_fold_' + str(i) + '.pkl')

        # store numpy arrays 
        np.save('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/y_train'+'_fold_' + str(i) +'.npy', y_train)
        np.save('/extra/arslan_data_repository/pre_processed/Mix_tggates_jnj/y_val'+'_fold_' + str(i) +'.npy', y_val)

        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, train_ndtp.shape, val_ndtp.shape)
    '''
    
    return x_train, y_train, x_val, y_val, train_drug_dose_time_finding_combinations, val_drug_dose_time_finding_combinations

def drug_dose_time_finding_data(fold):
    '''
    fetch prepcoessed data, 
    replace toxic catagories with numeric categories,
    return training and val folds
    '''
    # train data
    features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Time_Dose_findings.pkl')
    #features_target.drop('Unnamed: 0', axis = 1, inplace = True)
    finding_list = features_target.finding.unique().tolist()
    train = features_target[features_target.kfold != fold].reset_index(drop = True).drop(['kfold'],axis =1)
    validation = features_target[features_target.kfold == fold].reset_index(drop = True).drop(['kfold'],axis =1)

    # get drug-dose-time pairs by removing the redundent rows
    x_train = train.iloc[:,:4099]
    x_train.drop_duplicates(inplace = True)

    # reshape target into shape findings * animals
    y_train = train.iloc[:,4096:]
    y_train.drop(['drug','dose','time','finding'], axis =1 , inplace = True)
    y_train = y_train.to_numpy().reshape(x_train.shape[0],14,5)
    y_train[y_train ==1] = 0.2
    y_train[y_train ==2] = 0.4
    y_train[y_train ==3] = 0.6
    y_train[y_train ==4] = 0.8
    y_train[y_train ==5] = 1

    x_val = validation.iloc[:,:4099]
    x_val.drop_duplicates(inplace = True)

    y_val = validation.iloc[:,4096:]
    y_val.drop(['drug','dose','time','finding'], axis =1 , inplace = True)
    y_val = y_val.to_numpy().reshape(x_val.shape[0],14,5)
    y_val[y_val ==1] = 0.2
    y_val[y_val ==2] = 0.4
    y_val[y_val ==3] = 0.6
    y_val[y_val ==4] = 0.8
    y_val[y_val ==5] = 1
    return x_train, y_train, x_val, y_val

def drug_dose_time_finding_data_v2(fold = None, selected_findings_name = None):
    '''
    fetch prepcoessed data, 
    replace toxic catagories with numeric categories,
    return training and val folds
    '''
    
    ############# HTHC data ######################
    if fold == 'JnJ_HTHC_data_single_task':
        features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/JnJ_updated/JnJ_HTHC_data_single_task.pkl')
        test_drug_dose_time_finding_combinations = features_target.iloc[:,np.r_[0,4097:features_target.shape[1]]]

        x_test = features_target.iloc[:,:4099]
        x_test.drop_duplicates(inplace = True)

        y_test = features_target.iloc[:,4097:]
        num_findings = y_test.finding.nunique()
        num_combinatins = x_test.shape[0]

        y_test.drop(['dose','time','finding'], axis =1 , inplace = True)
        y_test = y_test.to_numpy().reshape(num_combinatins,num_findings,y_test.shape[1])
    ########### JnJ1 ########################
    elif fold == 'test':
        # test data 
        features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/JnJ1_Preprocess.pkl')
        test_drug_dose_time_finding_combinations = features_target.iloc[:,np.r_[0,4097:features_target.shape[1]]]
        
        # select relevent findingsplot_p
        # taken from JnJ_preprocessing file
        selected_Findings = [0,5,17,18,21,23,29,33,35,36]
        test_drug_dose_time_finding_combinations = test_drug_dose_time_finding_combinations[test_drug_dose_time_finding_combinations.finding.isin(selected_Findings)]
     
        # drop Non-Toxci class
        features_target = features_target[features_target.finding != 31]
        x_test = features_target.iloc[:,:4099]
        x_test.drop_duplicates(inplace = True)

        y_test = features_target.iloc[:,4097:]
        num_findings = y_test.finding.nunique()
        num_combinatins = x_test.shape[0]

        y_test.drop(['dose','time','finding'], axis =1 , inplace = True)
        y_test = y_test.to_numpy().reshape(num_combinatins,num_findings,y_test.shape[1])
        
    elif fold == 'jnj_complete':
        # test data 
        features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/JnJ_complete_Preprocess.pkl')
        test_drug_dose_time_finding_combinations = features_target.iloc[:,np.r_[0,4097:features_target.shape[1]]]
        
        # select relevent findingsplot_p
        # taken from JnJ_preprocessing file
        
        selected_Findings = [5,17,21,29,33,35,36]
        test_drug_dose_time_finding_combinations = test_drug_dose_time_finding_combinations[test_drug_dose_time_finding_combinations.finding.isin(selected_Findings)]

     
        # drop Non-Toxci class
        features_target = features_target[features_target.finding.isin(selected_Findings)]
        x_test = features_target.iloc[:,:4099]
        x_test.drop_duplicates(inplace = True)

        y_test = features_target.iloc[:,4097:]
        num_findings = y_test.finding.nunique()
        num_combinatins = x_test.shape[0]

        y_test.drop(['dose','time','finding'], axis =1 , inplace = True)
        y_test = y_test.to_numpy().reshape(num_combinatins,num_findings,y_test.shape[1])
    else:
        # train data
        features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Preprocess_Step3.pkl')
        selected_findings_dict ={0:'Apop',
                        3:'BDH',
                        4:'GlyD',
                        5:'Ephi',
                        6:'GlyA',
                        10:'EMH',
                        11:'Fibro',
                        17:'HTHC',  
                        18:'HTHP',
                        21:'IM',
                        23:'InMC',
                        29:'NecZ',
                        33:'Pigmen',
                        35:'ScNec',
                        36:'Vacu'}
        # select relevent findingsplot_p

        selected_Findings = [finding_number for finding_number, finding_name in selected_findings_dict.items() if finding_name in selected_findings_name]
        features_target = features_target[features_target.finding.isin(selected_Findings)]

        train = features_target[features_target.kfold != fold].reset_index(drop = True).drop(['kfold'],axis =1)
        validation = features_target[features_target.kfold == fold].reset_index(drop = True).drop(['kfold'],axis =1)
        
        train_drug_dose_time_finding_combinations = train.iloc[:,np.r_[0,4097:train.shape[1]]]
        val_drug_dose_time_finding_combinations = validation.iloc[:,np.r_[0,4097:validation.shape[1]]]
     
        # training data
        x_train = train.iloc[:,:4099]
        x_train.drop_duplicates(inplace = True)
        y_train = train.iloc[:,4097:]
        num_findings = y_train.finding.nunique()
        num_combinatins = x_train.shape[0]
        y_train.drop(['dose','time','finding'], axis =1 , inplace = True)
        y_train = y_train.to_numpy().reshape(num_combinatins,num_findings,5)

        # validation data
        x_val = validation.iloc[:,:4099]
        x_val.drop_duplicates(inplace = True)
        y_val = validation.iloc[:,4097:]
        num_findings = y_val.finding.nunique()
        num_combinatins = x_val.shape[0]
        y_val.drop(['dose','time','finding'], axis =1 , inplace = True)
        y_val = y_val.to_numpy().reshape(num_combinatins,num_findings,5)
    
    if (fold == 'test') or (fold == 'jnj_complete') or (fold == 'JnJ_HTHC_data_single_task'):
        return x_test, y_test, test_drug_dose_time_finding_combinations

    else:
        return x_train, y_train, x_val, y_val, train_drug_dose_time_finding_combinations, val_drug_dose_time_finding_combinations

###################################################################
# fetch pathological and bloodmarker data folds
###################################################################
def pathoology_BM_folds(fold):
    '''
    fetch prepcoessed data, 
    replace toxic catagories with numeric categories,
    return training and val folds
    '''
    features_target = pd.read_pickle('/extra/arslan_data_repository/pre_processed/Time_Dose_findings.pkl')
    #features_target.drop('Unnamed: 0', axis = 1, inplace = True)
    finding_list = features_target.finding.unique().tolist()
    train = features_target[features_target.kfold != fold].reset_index(drop = True).drop(['kfold'],axis =1)
    validation = features_target[features_target.kfold == fold].reset_index(drop = True).drop(['kfold'],axis =1)

    # get drug-dose-time pairs by removing the redundent rows
    x_train = train.iloc[:,:4099]
    x_train.drop_duplicates(inplace = True)

    # reshape target into shape findings * animals
    y_train = train.iloc[:,4096:]
    y_train.drop(['drug','dose','time','finding'], axis =1 , inplace = True)
    y_train = y_train.to_numpy().reshape(x_train.shape[0],14,5)
    y_train[y_train ==1] = 0.2
    y_train[y_train ==2] = 0.4
    y_train[y_train ==3] = 0.6
    y_train[y_train ==4] = 0.8
    y_train[y_train ==5] = 1

    x_val = validation.iloc[:,:4099]
    x_val.drop_duplicates(inplace = True)

    y_val = validation.iloc[:,4096:]
    y_val.drop(['drug','dose','time','finding'], axis =1 , inplace = True)
    y_val = y_val.to_numpy().reshape(x_val.shape[0],14,5)
    y_val[y_val ==1] = 0.2
    y_val[y_val ==2] = 0.4
    y_val[y_val ==3] = 0.6
    y_val[y_val ==4] = 0.8
    y_val[y_val ==5] = 1

    # blood markers data
    BM_Data = pd.read_pickle('/home/amasood1/TG GATE/Results/Blood_Marker_preprocess_v1.0.pkl')
    y_train_bm = BM_Data[BM_Data.kfold != fold].reset_index(drop = True).drop(['kfold'],axis =1)
    y_val_bm = BM_Data[BM_Data.kfold == fold].reset_index(drop = True).drop(['kfold'],axis =1)
    train_ndt_combinations = y_train_bm.drug.nunique() * y_train_bm.dose.nunique() * y_train_bm.time.nunique()
    val_ndt_combinations = y_val_bm.drug.nunique() * y_val_bm.dose.nunique() * y_val_bm.time.nunique()
    num_BM = y_train_bm.BM.nunique()
    y_train_bm.drop(['drug','dose','time','BM'], axis =1 , inplace = True)
    y_val_bm.drop(['drug','dose','time','BM'], axis =1 , inplace = True)
    y_train_bm = y_train_bm.to_numpy().reshape(train_ndt_combinations,num_BM,5)
    y_val_bm = y_val_bm.to_numpy().reshape(val_ndt_combinations,num_BM,5)

    # concatenate pathological and blood marker targets
    y_train = np.concatenate((y_train, y_train_bm), axis = 1)
    y_val = np.concatenate((y_val, y_val_bm), axis = 1)
    return x_train, y_train, x_val, y_val


###################################################################
#
###################################################################

def wandb_init_model(model, config, train_dataloader,val_dataloader, model_type):
    if val_dataloader == None:
        limit_val_batches = 0.0
    else:
        limit_val_batches = 1.0
    # Init our model
    if model_type == 'chemprop':
        run = wandb.init(
                        project= config.project_name,
                        dir = '/projects/home/mmasood1/Model_weights',
                        entity="arslan_masood", 
                        reinit = True, 
                        config = None,
                        name = config.model_name,
                        settings=wandb.Settings(start_method="fork"))
        
        default_root_dir = config.model_weights_dir
        use_pretrained_model = config.pretrained_model
        use_EarlyStopping = config.EarlyStopping
        max_epochs = config.epochs
        accelerator =config.accelerator
        return_trainer = config.return_trainer
        print(max_epochs)
    else:
        
        run = wandb.init(
                        project= config["project_name"],
                        dir = '/projects/home/mmasood1/Model_weights',
                        entity="arslan_masood", 
                        reinit = True, 
                        config = config,
                        name = config["model_name"],
                        settings=wandb.Settings(start_method="fork"))
        
        default_root_dir = config["model_weights_dir"]
        use_pretrained_model = config["pretrained_model"]
        use_EarlyStopping = config["EarlyStopping"]
        max_epochs = config["epochs"]
        accelerator =config["accelerator"]
        return_trainer = config["return_trainer"]

    if use_pretrained_model:
        model = pretrained_model(model,config)
    else:
        model = model(config)
    wandb_logger = WandbLogger(project= config["project_name"],
                        dir = '/projects/home/mmasood1/Model_weights',
                        entity="arslan_masood", 
                        reinit = True, 
                        config = config,
                        name = config["model_name"],
                        settings=wandb.Settings(start_method="fork"))
    #wandb_logger.watch(model, log="all",log_freq=1)
    
    if use_EarlyStopping == True:
        callback = [EarlyStopping(
                                monitor='train_BCE_loss',
                                min_delta=1e-5,
                                patience=10,
                                verbose=False,
                                mode='min'
                                )]
    else:
        callback = []

    checkpoint_callback = ModelCheckpoint(
    monitor=None,  # Metric to monitor for saving the best model
    mode='min',          # Minimize the monitored metric
    dirpath= default_root_dir,  # Directory to store checkpoints
    filename='model-{epoch:02d}-{val_BCE_non_weighted:.2f}',  # Checkpoint filename format
    #filename=config['chkp_file_name'],  # Checkpoint filename format
    save_top_k=1,
    save_last = None)
    callback.append(checkpoint_callback)


    trainer = Trainer(
        callbacks=callback,
        max_epochs= int(max_epochs),
        accelerator= accelerator, 
        devices= config['gpu'],
        #limit_val_batches = limit_val_batches,
        #precision=16,
        enable_progress_bar = True,
        #profiler="simple",
        enable_model_summary=True,
        logger=wandb_logger,
        default_root_dir=default_root_dir)

    # model fitting 
    trainer.fit(model, 
                train_dataloaders=train_dataloader,
                val_dataloaders =val_dataloader,
                )
    if return_trainer:
        return model, run, trainer
    else:
        return model, run

###################################################################
#
###################################################################
def open_black_box(model, config):
    
    def filter(name):
        isWeight = "weight" in name
        return isWeight

    plt.ioff()
    model_weights = plot_individual_weight(model, ncols=6, filter=filter)
    FI_plot,top_feature =  feature_importance(model)

    wandb.log({'model_weights': wandb.Image(model_weights),
               'FI_plot': wandb.Image(FI_plot)})

    return top_feature

class post_training_evaluation_Beta_CV_updated(object):
    def __init__(self, config, title):
          
        self.tasks = config['num_of_tasks']
        self.title = title
        self.subjective_tox_thershold = config["subjective_tox_thershold"]
        self.eps = 1e-5
        if self.title == 'test':
            self.task_name_list =['Apop','Ephi','HTHC','HTHP','IM','InMC','NecZ','Pigmen','ScNec','Vacu']
            
        if self.title == 'train':
            self.task_name_list  = {0:'Apop',
                        3:'BDH',
                        4:'GlyD',
                        5:'Ephi',
                        6:'GlyA',
                        10:'EMH',
                        11:'Fibro',
                        17:'HTHC',  
                        18:'HTHP',
                        21:'IM',
                        23:'InMC',
                        29:'NecZ',
                        33:'Pigmen',
                        35:'ScNec',
                        36:'Vacu'}
        
    def ELL_EoM_EoV(self, model, given_dataloader,config):
        x, d , t, self.y = next(iter(given_dataloader))
        self.y_true_clipped = self.y.clip(self.eps, 1 - self.eps)
        
        alphas_betas = model(x,d,t)
        self.alphas, self.betas = alphas_betas[:,0:self.tasks], alphas_betas[:,self.tasks:]
        self.LL = Beta(torch.unsqueeze(self.alphas, dim = 2),torch.unsqueeze(self.betas, dim = 2)).log_prob(self.y_true_clipped)
        ELL = self.LL.mean()
        
        self.beta_dist = Beta(self.alphas, self.betas)
        
        EoM = (torch.abs(self.beta_dist.mean - self.y.mean(axis =2))).mean()
        EoV = (torch.abs(self.beta_dist.variance - torch.var(self.y, axis = 2))).mean()
        observed_binary_labels, pred_prob_1 = self.compute_binary_auc()

        if config["class_dependent_Log_likelihood"] == 1:
            self.class_dependent_Log_likelihood()
            
            
        if config["plot_pdfs"] == 1:
            self.plot_overall_pred()
            self.plot_across_task()
            self.plot_across_dose_time()
        

        if config["alpha_beta_scatter_plot"] == 1:
            alpha_beta_scatter_plot = alpha_beta_scatter(self.alphas,self.betas)
            wandb.log({'alpha_beta_scatter_plot': wandb.Image(alpha_beta_scatter_plot)})

        if config["MSE_distribution_heatmaps"] == 1:
            # drug, dose, time, findings
            diff = MSE.view(-1,3,8,14)
            MSE_dt = diff.mean(axis = [0,3])
            MSE_dtp = diff.mean(axis = [0])
            heat_map_MSE_DT = plot_MSE_heatmap(MSE_dt,scale = 'MSE_dt', title = 'Marginalized across Drugs and Findings')
            heat_map_MSE_DTP = plot_MSE_heatmap(MSE_dtp,scale = 'MS_dtp', title = 'Marginalized across Drugs')
            wandb.log({'{}_heat_map_MSE_DT'.format(self.title): wandb.Image(heat_map_MSE_DT),
                       '{}_heat_map_MSE_DTP'.format(self.title): wandb.Image(heat_map_MSE_DTP)})
            
        return ELL,EoM,EoV, observed_binary_labels, pred_prob_1, self.alphas, self.betas
    
    def class_ELL(self, class_label):
        mask = self.y.eq(class_label)
        class_ELL = torch.masked_select(self.LL, mask).mean()
        return class_ELL
                
    def class_dependent_Log_likelihood(self):
        toxicity_class = [0,0.2,0.4,0.6,0.8,1]
        
        for class_label in toxicity_class:
            ELL_class = self.class_ELL(class_label)
            wandb.log({f'{self.title}_ELL_fold_class_{class_label}': np.round(ELL_class.item(),2)})
    
    def compute_binary_auc(self):

        # get binary labels
        observed_binary_labels = self.get_binary_labels(self.y)
        pred_prob_1 = self.pred_tox_prob()
        return observed_binary_labels, pred_prob_1
    
    def confusion_matrix_at_threshold(self, observed_binary_labels,pred_prob_1, th, ax):
        y_true = observed_binary_labels.ravel()
        y_pred = np.where(pred_prob_1.ravel() > th, 1, 0).ravel()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred, average='binary', pos_label= 1)
        recall_sensitivity = recall_score(y_true, y_pred, average='binary', pos_label= 1)
        specificity = recall_score(y_true, y_pred, average='binary', pos_label= 0)

        cm_highlights = np.matrix([[tp, fp, precision], [fn, tn, np.nan], [recall_sensitivity, specificity, np.nan]])
        cm_highlights = pd.DataFrame(cm_highlights, 
                                     columns = ['Observed \n present', 'Observed \n absent',''],
                                     index=['Predicted \n present', 'Predicted \n absent', ''])

        annotations = np.matrix([['TP \n'+ str(tp), 'FP \n'+ str(fp), 'precision \n TP/(TP+FP) \n'+ str(np.round(precision,2))], 
                                 ['FN \n'+ str(fn), 'TN \n'+ str(tn), np.nan], 
                                 ['recall \n sensitivity \n TP/(TP+FN)\n'+ str(np.round(recall_sensitivity,2)), 
                                  'specificity \n TN/(FP+TN)\n'+ str(np.round(specificity,2)), np.nan]])

        cmap = ListedColormap(['gray']).copy()
        cmap.set_bad('white')      # color of mask on heatmap
        cmap.set_under('white')    # color of mask on cbar

        sns.heatmap(cm_highlights, 
                         vmin = 10,
                         annot = annotations, 
                         fmt = '', 
                         cmap = cmap,
                         ax = ax, 
                         cbar  = False,
                         linewidths = 10,
                         linecolor = 'white')

        sns.heatmap(cm_highlights, 
                         vmin = 10,
                         fmt = '', 
                         cmap = cmap,
                         mask=cm_highlights < 1,
                         ax = ax,
                         cbar=False,
                         linewidths = 10,
                         linecolor = 'white')
        ax.tick_params(left=False, bottom=False, labeltop=True, labelbottom=False)
        ax.add_patch(Rectangle(xy = (0, 0), width = 1, height = 3, fill=False, edgecolor='blue', lw=5, ls = '--', alpha = 0.5))
        ax.add_patch(Rectangle(xy = (1, 0), width = 1, height = 3, fill=False, edgecolor='red', lw=5, ls = '--', alpha = 0.5))
        ax.add_patch(Rectangle(xy = (0, 0), width = 3, height = 1, fill=False, edgecolor='green', lw=5, ls = '--', alpha = 0.5))
        ax.set_title(label = 'Thershold {0:.2f}'.format(th), x = 0.35, y = -0.1)
    
    def plot_aucs(self,observed_binary_labels,pred_prob_1):
        
        fpr, tpr, precision, recall, roc_auc, average_precision = dict(),dict(),dict(), dict(), dict(), dict()
        
        num_of_tasks = observed_binary_labels.shape[1]
        for finding in range(num_of_tasks):

            fpr[finding], tpr[finding], _ = roc_curve(observed_binary_labels[:,finding], pred_prob_1[:,finding], pos_label=1)
            precision[finding], recall[finding], _ = precision_recall_curve(observed_binary_labels[:,finding], pred_prob_1[:,finding], pos_label=1)
            
            roc_auc[finding] = auc(fpr[finding], tpr[finding])
            average_precision[finding] = average_precision_score(observed_binary_labels[:,finding], pred_prob_1[:,finding], pos_label=1)
    
        fpr["micro"], tpr["micro"], thresholds_roc = roc_curve(observed_binary_labels.ravel(), pred_prob_1.ravel(), pos_label=1)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], thresholds_aupr = precision_recall_curve(observed_binary_labels.ravel(), pred_prob_1.ravel(), pos_label=1)
        average_precision["micro"] = average_precision_score(observed_binary_labels, pred_prob_1, average="micro")

        wandb.log({
            '{}_micro_roc_score'.format(self.title):  roc_auc["micro"],
            '{}_micro_avg_precision_score'.format(self.title):  average_precision["micro"]
        })

        ########## plot AUC-ROC ################
        plt.ioff()
        fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(16,8), constrained_layout = True)
        axs = ax.ravel()
        for i in range(num_of_tasks):
            sns.lineplot(fpr[i],tpr[i], lw=2, ci = None, alpha = 0.4, ax = axs[0],
                         label="{0} \n [{1:.2f},{2:.2f}]".format(self.task_name_list[i], roc_auc[i], average_precision[i]))

        sns.lineplot(fpr["micro"],tpr["micro"],lw=2, ax = axs[0], color = 'blue', 
                          label= "Micro_avg \n [{0:.2f},{1:.2f}]".format(roc_auc["micro"], average_precision["micro"]))
        
        # best value
        gmeans = np.sqrt(tpr["micro"] * (1-fpr["micro"]))
        ix = np.argmax(gmeans)
        th_roc = thresholds_roc[ix]
        sns.scatterplot(fpr["micro"][ix].reshape(-1), 
                        tpr["micro"][ix].reshape(-1),
                        s = 500, 
                        marker = '*',
                        ax = axs[0])

        sns.lineplot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", ax = axs[0])
        axs[0].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("FPR (1 - specificity)"), ylabel = ("Sensitivity"))
        axs[0].set_title("AUC-ROC (micro avg.)  %0.2f" % (roc_auc["micro"]))
        axs[0].legend().set_visible(False)
        sns.lineplot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", ax = axs[0])
        #axs[0].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("False Positive Rate"), ylabel = ("True Positive Rate"))
        axs[0].set_title("AUC-ROC (micro avg.)  %0.2f" % (roc_auc["micro"]))
        axs[0].legend().set_visible(False)
        ########## Plot AU-PR ###################
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color="gray",ls = '--', alpha=0.2)
            axs[1].annotate("f1={0:0.1f}".format(f_score), xy=(0.87, y[45] + 0.02))

        for i in range(num_of_tasks):
            sns.lineplot(recall[i],precision[i], lw=2, ci = None, alpha = 0.4, ax = axs[1])
        
        sns.lineplot(recall["micro"],precision["micro"], lw=2, color="blue", ax = axs[1])
        # best value
        fscore = (2 * precision["micro"] * recall["micro"]) / (precision["micro"] + recall["micro"])
        ix = np.argmax(np.nan_to_num(fscore, 0))
        th_aupr = thresholds_aupr[ix]
        sns.scatterplot(recall["micro"][ix].reshape(-1), 
                        precision["micro"][ix].reshape(-1),
                        s = 500, 
                        marker = '*',
                        ax = axs[1])
        # set the legend and the axes
        axs[1].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("Recall (sensitivity)"), ylabel = ("Precision"))
        axs[0].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("FPR (1 - specificity)"), ylabel = ("Sensitivity"))
        #axs[1].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("Recall"), ylabel = ("Precision"))
        axs[1].set_title("AU-PR (micro avg.) %0.2f" % (average_precision["micro"]))
        
        # plot confusion matrix at best location
        self.confusion_matrix_at_threshold(observed_binary_labels,pred_prob_1, th_roc, axs[2])
        self.confusion_matrix_at_threshold(observed_binary_labels,pred_prob_1, th_aupr, axs[3])
        handles, labels = axs[0].get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        fig.legend(handles, labels, ncol = int(num_of_tasks/2),
                   title = 'Task [AUC-ROC, AU-PR]',title_fontsize = 15,
                      bbox_to_anchor=(0.95, 0))
        wandb.log({'{}_AUC_ROC_AUPR'.format(self.title): wandb.Image(fig)})
        plt.close()
        plt.show()

        
        
    def get_binary_labels(self,lables):
        '''
        labels format: (ndt, findings, animals)
        '''
        binary_labels = [[] for i in range(lables.shape[0])]
        
        for finding in range(lables.shape[1]):
            for comb in range(lables.shape[0]):
                
                y_finding = lables[comb,finding,:]
                #total_tested_animals = torch.count_nonzero(~torch.isnan(y_finding))
                tox_animals = torch.count_nonzero(y_finding[~torch.isnan(y_finding)])
                #tox_percentage = tox_animals / total_tested_animals
                
                #if tox_percentage >= (self.subjective_tox_thershold):
                if tox_animals >= (self.subjective_tox_thershold):
                    binary_labels[comb].append(1)
                else:
                    binary_labels[comb].append(0)
        binary_labels = np.array(binary_labels)
        return binary_labels
    
    def pred_tox_prob(self):
        '''
        return shape: (ndt) * task
        '''
        a, b = self.alphas.detach().numpy(), self.betas.detach().numpy() 
        prob = 1 - beta.cdf(x = 0.1, a = a, b = b)
        return prob
    
    def get_pdfs_of_most_toxic_drug(self):
        
        alphas, betas = self.alphas.view(-1,3,8,14).detach().numpy(), self.betas.view(-1,3,8,14).detach().numpy()
        y_true = self.y.view(-1,3,8,14,5)
        max_tox_drug = torch.argmax(y_true.count_nonzero(dim = (1,2,3,4))).item()
        alphas, betas = alphas[max_tox_drug,:,:,:], betas[max_tox_drug,:,:,:]

        x = np.linspace(0.001, 0.99, 100)
        pdfs = pd.DataFrame()
        for dose in range(3):
            for time in range(8):
                for finding in range(14):
                    dist = beta(alphas[dose,time,finding], betas[dose,time,finding])
                    dist = dist.pdf(x)
                    #print(dist.shape)
                    pdfs = pdfs.append({'Tox_class' : pd.Series(x.tolist()),
                                        'dose' : pd.Series([dose] * 100),
                                        'time' : pd.Series([time] * 100),
                                        'finding' : pd.Series([finding] * 100),
                                        'pdfs' : pd.Series(dist.tolist())},
                                       ignore_index=True)
        pdfs = pdfs.explode(['Tox_class','dose','time','finding','pdfs'])  

        time_categories = { 0:'3h', 1:'6h', 2:'9h', 3:'24h', 4:'4day', 5:'8day', 6:'15day', 7:'29day'}
        pdfs.time = pdfs.time.map(time_categories)

        dose_categories = {0:'low', 1:'Middle', 2:'High'}
        pdfs.dose = pdfs.dose.map(dose_categories)
        return pdfs
    
    def insetsection_hist_kde(self,start,end,beta,x,hist_val):
   
        beta_pdf = beta[start:end]
        hist = np.repeat(hist_val,100)
        inters_x = np.minimum(beta_pdf, hist)
        area_inters_x = np.trapz(inters_x, x[start:end])
        return inters_x.tolist(), area_inters_x.tolist()
    
    def plot_overall_pred(self):
        plt.ioff()
        # plot indivdiula pdfs
        eps = 1e-5
        x1 = np.linspace(0,0.1,100)
        x2 = np.linspace(0.1,0.3,100)
        x3 = np.linspace(0.3,0.5,100)
        x4 = np.linspace(0.5,0.7,100)
        x5 = np.linspace(0.7,0.9,100)
        x6 = np.linspace(0.9,1,100)
        x = np.concatenate([x1, x2, x3, x4, x5, x6]).clip(eps, 1-eps)

        a = self.alphas.detach().numpy().reshape(-1)
        b = self.betas.detach().numpy().reshape(-1)
        obs = self.y.reshape(-1).detach().numpy()

        fig= plt.figure(figsize = (15,7))
        ax = fig.add_subplot(1, 1, 1)
        pdfs = []
        for i in range(a.shape[0]):
            pdf = beta.pdf(x, a[i], b[i])
            plt.plot(x,pdf, alpha = 0.1, lw = 0.05, color = 'orange', label = '_nolegend_')
            pdfs.append(pdf)

        # averaging all the pdfs
        pdf2 = np.stack(pdfs, axis =0).mean(axis = 0)
        plt.plot(x, pdf2, color='black',ls = '--', lw = 1, alpha = 1, label = '_nolegend_')

        # calculate histogram
        hist, bin_edges = np.histogram(obs,bins = [0,0.1,0.3,0.5,0.7,0.9,1], density = True)
        #hist = (hist *  np.diff(bin_edges))
        plt.bar(bin_edges[:-1],hist,width=np.diff(bin_edges), fill = False, align='edge')

        # calculate intersection bw av_beta and histogram
        edge_1 = [0,100,200,300,400,500,600,700,800,900]
        edge_2 = [100,200,300,400,500,600,700,800,900,1000]
        inters_x = []
        area_inters_x = 0
        for start,end,h in zip(edge_1, edge_2, hist):
            intersection, area = self.insetsection_hist_kde(start = start,end = end, beta = pdf2,x = x,hist_val = h)
            inters_x.extend(intersection)
            area_inters_x += area 

        plt.plot(x, inters_x, color='blue')
        plt.fill_between(x, inters_x, 0, facecolor='none', edgecolor='blue', hatch='xx', label = '_nolegend_')
        #plt.yscale('log')
        plt.ylim(0,20)

        # custom legends
        beta_i = mlines.Line2D([], [], color='orange',  ls='-', label=r'$\mathrm{Beta}_{i}$')
        E_Beta = mlines.Line2D([], [], color='black',  ls='--', label=r'$\mathbb{E}[\mathrm{Beta}_{i}]$')
        int_sec = matplotlib.patches.Patch(facecolor='#DCDCDC', edgecolor='blue', hatch='xx', label='intersection \n {0:.2f} %'.format(area_inters_x * 100) )
        plt.legend(handles = [beta_i,E_Beta, int_sec], title='Distributions', loc = 'upper right')
        plt.tight_layout()
        plt.ylabel('Density')
        plt.xlabel('Toxicity Severity')
        plt.title('Emprirical distribution vs Marginalized beta distribution')
        plt.tight_layout()
        plt.close()
        wandb.log({"Prediction_overall_{}".format(self.title): wandb.Image(ax)})
        wandb.log({f'{self.title}_OC': np.round(area_inters_x,2)})
        

    def plot_across_task(self):
        plt.ioff()
       # plot indivdiula pdfs
        eps = 1e-5
        x1 = np.linspace(0,0.1,100)
        x2 = np.linspace(0.1,0.3,100)
        x3 = np.linspace(0.3,0.5,100)
        x4 = np.linspace(0.5,0.7,100)
        x5 = np.linspace(0.7,0.9,100)
        x6 = np.linspace(0.9,1,100)
        x = np.concatenate([x1, x2, x3, x4, x5, x6]).clip(eps, 1-eps)

        fig, ax = plt.subplots(nrows=6, ncols=3,figsize=(30,20), sharey = True ,constrained_layout=True)

        ax.flat[-1].set_visible(False)
        axs = ax.ravel()
        for task in range(15):
            obs = self.y[:,task,:].detach().numpy()
            a_mean = self.alphas[:,task].detach().numpy().mean()
            b_mean = self.betas[:,task].detach().numpy().mean()

            a = self.alphas[:,task].detach().numpy().reshape(-1)
            b = self.betas[:,task].detach().numpy().reshape(-1)

            pdfs = []
            for i in range(a.shape[0]):
                pdf = beta.pdf(x, a[i], b[i])
                axs[task].plot(x,pdf, alpha = 0.1, lw = 0.05, color = 'orange', label = '_nolegend_')
                pdfs.append(pdf)

            # averaging all the pdfs
            pdf2 = np.stack(pdfs, axis =0).mean(axis = 0)
            axs[task].plot(x, pdf2, color='black',ls = '--', lw = 1, alpha = 1, label = '_nolegend_')

            # calculate histogram
            hist, bin_edges = np.histogram(obs,bins = [0,0.1,0.3,0.5,0.7,0.9,1], density = True)
            #hist = (hist *  np.diff(bin_edges))
            axs[task].bar(bin_edges[:-1],hist,width=np.diff(bin_edges), fill = False, align='edge')

            # calculate intersection bw av_beta and histogram
            edge_1 = [0,100,200,300,400,500,600,700,800,900]
            edge_2 = [100,200,300,400,500,600,700,800,900,1000]
            inters_x = []
            area_inters_x = 0
            for start,end,h in zip(edge_1, edge_2, hist):
                intersection, area = self.insetsection_hist_kde(start = start,end = end, beta = pdf2,x = x,hist_val = h)
                inters_x.extend(intersection)
                area_inters_x += area 

            axs[task].plot(x, inters_x, color='blue')
            axs[task].fill_between(x, inters_x, 0, facecolor='none', edgecolor='blue', hatch='xx', label = '_nolegend_')
            axs[task].set_xlabel('Toxicity Classes')
            axs[task].set_ylabel('Density')

            beta_i = mlines.Line2D([], [], color='orange',  ls='-', label=r'$\mathrm{Beta}_{i}$')
            E_Beta = mlines.Line2D([], [], color='black',  ls='--', label='E_Beta({0:.2f}, {1:.2f})'.format(np.round(a_mean,2),np.round(b_mean,2)))
            int_sec = matplotlib.patches.Patch(facecolor='#DCDCDC', edgecolor='blue', hatch='xx', label='intersection \n {0:.2f} %'.format(area_inters_x * 100) )
            axs[task].legend(handles = [beta_i,E_Beta, int_sec], title=f'Task : {self.task_name_list[task]}', loc = 'upper right')

        plt.ylim(-0.1, 20)
        # alpha distribution
        g = sns.histplot(self.alphas.detach().numpy().mean(axis = 0), kde = True, stat = 'count', ax = axs[task + 1])
        labels = [str(v) if v else '' for v in g.containers[0].datavalues]
        g.bar_label(g.containers[0], labels=labels)
        axs[task + 1].legend('', '', title= 'alpha distribution \n across Tasks', bbox_to_anchor=(1,1))

        # beta distribution
        g = sns.histplot(self.betas.detach().numpy().mean(axis = 0), kde = True, stat = 'count', ax = axs[task + 2], label = 'Beta distribution \n across Tasks')
        labels = [str(v) if v else '' for v in g.containers[0].datavalues]
        g.bar_label(g.containers[0], labels=labels)
        axs[task + 2].legend('', '', title= 'Beta distribution \n across Tasks', bbox_to_anchor=(1,1))
        plt.suptitle('Observed vs Predicted (Across tasks)', fontsize = 22)
        plt.close()
        wandb.log({"Prediction_across_tasks_{}".format(self.title): wandb.Image(fig)})
        
    def plot_across_dose_time(self):
        x = np.linspace(0,1,100).clip(self.eps, 1-self.eps)
        plt.ioff()
        eps = 1e-5
        x1 = np.linspace(0,0.1,100)
        x2 = np.linspace(0.1,0.3,100)
        x3 = np.linspace(0.3,0.5,100)
        x4 = np.linspace(0.5,0.7,100)
        x5 = np.linspace(0.7,0.9,100)
        x6 = np.linspace(0.9,1,100)
        x = np.concatenate([x1, x2, x3, x4, x5, x6]).clip(eps, 1-eps)

        fig, ax = plt.subplots(nrows=8, ncols=3,figsize=(20,35), sharey = True, constrained_layout=True)
        axs = ax.ravel()
        i = -1

        for time in range(8):
            for dose in range(3):
                i += 1
                obs = self.y.view(-1,3,8,15,5)
                a = self.alphas.view(-1,3,8,15).mean(axis = (0,3))
                b = self.betas.view(-1,3,8,15).mean(axis = (0,3))

                a = a[dose,time].detach().numpy().reshape(-1)
                b = b[dose,time].detach().numpy().reshape(-1)
                obs = obs[:,dose,time,:].detach().numpy()

                pdfs = []
                a_mean = a.mean()
                b_mean = b.mean()

                for param in range(a.shape[0]):
                    pdf = beta.pdf(x, a[param], b[param])
                    axs[i].plot(x,pdf, alpha = 0.1, lw = 0.05, color = 'orange', label = '_nolegend_')
                    pdfs.append(pdf)

                # averaging all the pdfs
                pdf2 = np.stack(pdfs, axis =0).mean(axis = 0)
                axs[i].plot(x, pdf2, color='black',ls = '--', lw = 1, alpha = 1, label = '_nolegend_')

                # calculate histogram
                hist, bin_edges = np.histogram(obs,bins = [0,0.1,0.3,0.5,0.7,0.9,1], density = True)
                #hist = (hist *  np.diff(bin_edges))
                axs[i].bar(bin_edges[:-1],hist,width=np.diff(bin_edges), fill = False, align='edge')

                # calculate intersection bw av_beta and histogram
                edge_1 = [0,100,200,300,400,500,600,700,800,900]
                edge_2 = [100,200,300,400,500,600,700,800,900,1000]
                inters_x = []
                area_inters_x = 0
                for start,end,h in zip(edge_1, edge_2, hist):
                    intersection, area = self.insetsection_hist_kde(start = start,end = end, beta = pdf2,x = x,hist_val = h)
                    inters_x.extend(intersection)
                    area_inters_x += area 

                axs[i].plot(x, inters_x, color='blue')
                axs[i].fill_between(x, inters_x, 0, facecolor='none', edgecolor='blue', hatch='xx', label = '_nolegend_')
                axs[i].set_xlabel('Toxicity Classes')
                axs[i].set_ylabel('Density')

                beta_i = mlines.Line2D([], [], color='orange',  ls='-', label=r'$\mathrm{Beta}_{i}$')
                E_Beta = mlines.Line2D([], [], color='black',  ls='--', label='E_Beta({0:.2f}, {1:.2f})'.format(np.round(a_mean,2),np.round(b_mean,2)))
                int_sec = matplotlib.patches.Patch(facecolor='#DCDCDC', edgecolor='blue', hatch='xx', label='intersection \n {0:.2f} %'.format(area_inters_x * 100) )
                axs[i].legend(handles = [beta_i,E_Beta, int_sec], title=f'time {time + 1}, dose {dose + 1}', loc = 'upper right')


        plt.ylim(-0.1, 20)
        plt.suptitle('Observed vs Predicted (Across Time-Dose)', fontsize = 22)
        plt.close()
        wandb.log({"Prediction_across_dose_time_{}".format(self.title): wandb.Image(fig)})
        
class evaluation_JnJ1(object):
    def __init__(self, config, title):
        
        self.tasks = config['num_of_tasks']
        self.title = title,
        self.subjective_tox_thershold = config["subjective_tox_thershold"]
        self.eps = 1e-5
        self.task_name_list =['Apop','Ephi','HTHC','HTHP','IM','InMC','NecZ','Pigmen','ScNec','Vacu']
        
    def ELL_EoM_EoV(self, model, given_dataloader,config):
        
        
        x, d , t, self.y = next(iter(given_dataloader))

        alphas_betas = model(x,d,t)
        self.alphas, self.betas = alphas_betas[:,0:self.tasks], alphas_betas[:,self.tasks:]

        # drop irrelevent task
        ind = torch.nonzero(self.y.nansum(axis = [0,2]))
        self.alphas, self.betas, self.y = self.alphas[:,ind], self.betas[:,ind], self.y[:,ind,:].squeeze()

        # use mask to handle missing animals
        mask =torch.isnan(self.y)
        y_nan_repalced_with_zero =  torch.nan_to_num(self.y)
        y_nan_repalced_with_zero_clipped = y_nan_repalced_with_zero.clip(self.eps, 1 - self.eps)

        # Loglikelihood
        beta_dist = Beta(self.alphas,self.betas)
        self.LL = beta_dist.log_prob(torch.nan_to_num(y_nan_repalced_with_zero_clipped))
        # apply masking to consider LL only for 'observed animals'
        LL_observed_animal = self.LL[~mask]
        ELL = LL_observed_animal.mean()
        
        # dont need mask, nanmean handles the missing values
        EoM = (torch.abs(beta_dist.mean.squeeze() - self.y.nanmean(axis =2))).mean()

        # python does not have function torch.nanvar()
        EoV = (torch.abs(beta_dist.variance.squeeze() - torch.from_numpy(np.nanvar(self.y.numpy(), axis = 2)))).mean()
            
        observed_binary_labels, pred_prob_1 = self.compute_binary_auc()
        
        if config["class_dependent_Log_likelihood"] == 1:
            self.class_dependent_Log_likelihood()
            
        return ELL,EoM,EoV, observed_binary_labels, pred_prob_1, self.alphas, self.betas, 
        
    def class_ELL(self, class_label):
        mask = self.y.eq(class_label)
        class_ELL = torch.masked_select(self.LL, mask).mean()
        return class_ELL
                
    def class_dependent_Log_likelihood(self):
        toxicity_class = [0,0.2,0.4,0.6,0.8,1]
        
        for class_label in toxicity_class:
            ELL_class = self.class_ELL(class_label)
            wandb.log({f'test_ELL_fold_class_{class_label}': np.round(ELL_class.item(),2)})
    
    def compute_binary_auc(self):

        # get binary labels
        observed_binary_labels = self.get_binary_labels(self.y)
        pred_prob_1 = self.pred_tox_prob(self.alphas.squeeze(), self.betas.squeeze())
        return observed_binary_labels, pred_prob_1
        
    def get_binary_labels(self,lables):
        '''
        labels format: (ndt, findings, animals)
        '''
        binary_labels = [[] for i in range(lables.shape[0])]
        
        for finding in range(lables.shape[1]):
            for comb in range(lables.shape[0]):
                
                y_finding = lables[comb,finding,:]
                #total_tested_animals = torch.count_nonzero(~torch.isnan(y_finding))
                tox_animals = torch.count_nonzero(y_finding[~torch.isnan(y_finding)])
                #tox_percentage = tox_animals / total_tested_animals
                
                #if tox_percentage >= (self.subjective_tox_thershold):
                if tox_animals >= (self.subjective_tox_thershold):
                    binary_labels[comb].append(1)
                else:
                    binary_labels[comb].append(0)
        binary_labels = np.array(binary_labels)
        return binary_labels
    
    def pred_tox_prob(self, alphas, betas):
        '''
        return shape: (ndt) * task
        '''
        a, b = alphas.detach().numpy(), betas.detach().numpy() 
        prob = 1 - beta.cdf(x = 0.1, a = a, b = b)
        return prob
    
    def plot_aucs(self,observed_binary_labels,pred_prob_1, config):
        test_evaluation = post_training_evaluation_Beta_CV_updated(config,'test')
        test_evaluation.plot_aucs(observed_binary_labels,pred_prob_1)
        
    def plot_overall_pred(self):
        plt.ioff()
        # plot indivdiula pdfs
        eps = 1e-5
        x1 = np.linspace(0,0.1,100)
        x2 = np.linspace(0.1,0.3,100)
        x3 = np.linspace(0.3,0.5,100)
        x4 = np.linspace(0.5,0.7,100)
        x5 = np.linspace(0.7,0.9,100)
        x6 = np.linspace(0.9,1,100)
        x = np.concatenate([x1, x2, x3, x4, x5, x6]).clip(eps, 1-eps)

        a = self.alphas.detach().numpy().reshape(-1)
        b = self.betas.detach().numpy().reshape(-1)
        obs = self.y.reshape(-1).detach().numpy()

        fig= plt.figure(figsize = (15,7))
        ax = fig.add_subplot(1, 1, 1)
        pdfs = []
        for i in range(a.shape[0]):
            pdf = beta.pdf(x, a[i], b[i])
            plt.plot(x,pdf, alpha = 0.1, lw = 0.05, color = 'orange', label = '_nolegend_')
            pdfs.append(pdf)

        # averaging all the pdfs
        pdf2 = np.stack(pdfs, axis =0).mean(axis = 0)
        plt.plot(x, pdf2, color='black',ls = '--', lw = 1, alpha = 1, label = '_nolegend_')

        # calculate histogram
        hist, bin_edges = np.histogram(obs,bins = [0,0.1,0.3,0.5,0.7,0.9,1], density = True)
        #hist = (hist *  np.diff(bin_edges))
        plt.bar(bin_edges[:-1],hist,width=np.diff(bin_edges), fill = False, align='edge')

        # calculate intersection bw av_beta and histogram
        edge_1 = [0,100,200,300,400,500,600,700,800,900]
        edge_2 = [100,200,300,400,500,600,700,800,900,1000]
        inters_x = []
        area_inters_x = 0
        for start,end,h in zip(edge_1, edge_2, hist):
            intersection, area = self.insetsection_hist_kde(start = start,end = end, beta = pdf2,x = x,hist_val = h)
            inters_x.extend(intersection)
            area_inters_x += area 

        plt.plot(x, inters_x, color='blue')
        plt.fill_between(x, inters_x, 0, facecolor='none', edgecolor='blue', hatch='xx', label = '_nolegend_')
        #plt.yscale('log')
        plt.ylim(0,20)

        # custom legends
        beta_i = mlines.Line2D([], [], color='orange',  ls='-', label=r'$\mathrm{Beta}_{i}$')
        E_Beta = mlines.Line2D([], [], color='black',  ls='--', label=r'$\mathbb{E}[\mathrm{Beta}_{i}]$')
        int_sec = matplotlib.patches.Patch(facecolor='#DCDCDC', edgecolor='blue', hatch='xx', label='intersection \n {0:.2f} %'.format(area_inters_x * 100) )
        plt.legend(handles = [beta_i,E_Beta, int_sec], title='Distributions', loc = 'upper right')
        plt.tight_layout()
        plt.ylabel('Density')
        plt.xlabel('Toxicity Severity')
        plt.title('Emprirical distribution vs Marginalized beta distribution')
        plt.tight_layout()
        plt.close()
        wandb.log({"Prediction_overall_{}".format(self.title): wandb.Image(ax)})
        wandb.log({f'{self.title}_OC': np.round(area_inters_x,2)})
    
    def insetsection_hist_kde(self,start,end,beta,x,hist_val):
   
        beta_pdf = beta[start:end]
        hist = np.repeat(hist_val,100)
        inters_x = np.minimum(beta_pdf, hist)
        area_inters_x = np.trapz(inters_x, x[start:end])
        return inters_x.tolist(), area_inters_x.tolist()
    
    def plot_across_task(self):
        plt.ioff()
       # plot indivdiula pdfs
        eps = 1e-5
        x1 = np.linspace(0,0.1,100)
        x2 = np.linspace(0.1,0.3,100)
        x3 = np.linspace(0.3,0.5,100)
        x4 = np.linspace(0.5,0.7,100)
        x5 = np.linspace(0.7,0.9,100)
        x6 = np.linspace(0.9,1,100)
        x = np.concatenate([x1, x2, x3, x4, x5, x6]).clip(eps, 1-eps)

        fig, ax = plt.subplots(nrows=4, ncols=3,figsize=(30,20), sharey = True ,constrained_layout=True)
        axs = ax.ravel()
        for task in range(10):
            obs = self.y[:,task,:].detach().numpy()
            a_mean = self.alphas[:,task].detach().numpy().mean()
            b_mean = self.betas[:,task].detach().numpy().mean()

            a = self.alphas[:,task].detach().numpy().reshape(-1)
            b = self.betas[:,task].detach().numpy().reshape(-1)

            pdfs = []
            for i in range(a.shape[0]):
                pdf = beta.pdf(x, a[i], b[i])
                axs[task].plot(x,pdf, alpha = 0.1, lw = 0.05, color = 'orange', label = '_nolegend_')
                pdfs.append(pdf)

            # averaging all the pdfs
            pdf2 = np.stack(pdfs, axis =0).mean(axis = 0)
            axs[task].plot(x, pdf2, color='black',ls = '--', lw = 1, alpha = 1, label = '_nolegend_')

            # calculate histogram
            hist, bin_edges = np.histogram(obs,bins = [0,0.1,0.3,0.5,0.7,0.9,1], density = True)
            #hist = (hist *  np.diff(bin_edges))
            axs[task].bar(bin_edges[:-1],hist,width=np.diff(bin_edges), fill = False, align='edge')

            # calculate intersection bw av_beta and histogram
            edge_1 = [0,100,200,300,400,500,600,700,800,900]
            edge_2 = [100,200,300,400,500,600,700,800,900,1000]
            inters_x = []
            area_inters_x = 0
            for start,end,h in zip(edge_1, edge_2, hist):
                intersection, area = self.insetsection_hist_kde(start = start,end = end, beta = pdf2,x = x,hist_val = h)
                inters_x.extend(intersection)
                area_inters_x += area 

            axs[task].plot(x, inters_x, color='blue')
            axs[task].fill_between(x, inters_x, 0, facecolor='none', edgecolor='blue', hatch='xx', label = '_nolegend_')
            axs[task].set_xlabel('Toxicity Classes')
            axs[task].set_ylabel('Density')

            beta_i = mlines.Line2D([], [], color='orange',  ls='-', label=r'$\mathrm{Beta}_{i}$')
            E_Beta = mlines.Line2D([], [], color='black',  ls='--', label='E_Beta({0:.2f}, {1:.2f})'.format(np.round(a_mean,2),np.round(b_mean,2)))
            int_sec = matplotlib.patches.Patch(facecolor='#DCDCDC', edgecolor='blue', hatch='xx', label='intersection \n {0:.2f} %'.format(area_inters_x * 100) )
            axs[task].legend(handles = [beta_i,E_Beta, int_sec], title=f'Task : {self.task_name_list[task]}', loc = 'upper right')

        plt.ylim(-0.1, 20)
        # alpha distribution
        g = sns.histplot(self.alphas.detach().numpy().mean(axis = 0), kde = True, stat = 'count', ax = axs[task + 1])
        labels = [str(v) if v else '' for v in g.containers[0].datavalues]
        g.bar_label(g.containers[0], labels=labels)
        axs[task + 1].legend('', '', title= 'alpha distribution \n across Tasks', bbox_to_anchor=(1,1))

        # beta distribution
        g = sns.histplot(self.betas.detach().numpy().mean(axis = 0), kde = True, stat = 'count', ax = axs[task + 2], label = 'Beta distribution \n across Tasks')
        labels = [str(v) if v else '' for v in g.containers[0].datavalues]
        g.bar_label(g.containers[0], labels=labels)
        axs[task + 2].legend('', '', title= 'Beta distribution \n across Tasks', bbox_to_anchor=(1,1))
        plt.suptitle('Observed vs Predicted (Across tasks)', fontsize = 22)
        plt.close()
        wandb.log({"Prediction_across_tasks_{}".format(self.title): wandb.Image(fig)})
        
    def plot_across_dose_time(self):
        x = np.linspace(0,1,100).clip(self.eps, 1-self.eps)
        plt.ioff()
        eps = 1e-5
        x1 = np.linspace(0,0.1,100)
        x2 = np.linspace(0.1,0.3,100)
        x3 = np.linspace(0.3,0.5,100)
        x4 = np.linspace(0.5,0.7,100)
        x5 = np.linspace(0.7,0.9,100)
        x6 = np.linspace(0.9,1,100)
        x = np.concatenate([x1, x2, x3, x4, x5, x6]).clip(eps, 1-eps)

        fig, ax = plt.subplots(nrows=8, ncols=3,figsize=(20,35), sharey = True, constrained_layout=True)
        axs = ax.ravel()
        i = -1

        for time in range(8):
            for dose in range(3):
                i += 1
                obs = self.y.view(-1,3,8,15,5)
                a = self.alphas.view(-1,3,8,15).mean(axis = (0,3))
                b = self.betas.view(-1,3,8,15).mean(axis = (0,3))

                a = a[dose,time].detach().numpy().reshape(-1)
                b = b[dose,time].detach().numpy().reshape(-1)
                obs = obs[:,dose,time,:].detach().numpy()

                pdfs = []
                a_mean = a.mean()
                b_mean = b.mean()

                for param in range(a.shape[0]):
                    pdf = beta.pdf(x, a[param], b[param])
                    axs[i].plot(x,pdf, alpha = 0.1, lw = 0.05, color = 'orange', label = '_nolegend_')
                    pdfs.append(pdf)

                # averaging all the pdfs
                pdf2 = np.stack(pdfs, axis =0).mean(axis = 0)
                axs[i].plot(x, pdf2, color='black',ls = '--', lw = 1, alpha = 1, label = '_nolegend_')

                # calculate histogram
                hist, bin_edges = np.histogram(obs,bins = [0,0.1,0.3,0.5,0.7,0.9,1], density = True)
                #hist = (hist *  np.diff(bin_edges))
                axs[i].bar(bin_edges[:-1],hist,width=np.diff(bin_edges), fill = False, align='edge')

                # calculate intersection bw av_beta and histogram
                edge_1 = [0,100,200,300,400,500,600,700,800,900]
                edge_2 = [100,200,300,400,500,600,700,800,900,1000]
                inters_x = []
                area_inters_x = 0
                for start,end,h in zip(edge_1, edge_2, hist):
                    intersection, area = self.insetsection_hist_kde(start = start,end = end, beta = pdf2,x = x,hist_val = h)
                    inters_x.extend(intersection)
                    area_inters_x += area 

                axs[i].plot(x, inters_x, color='blue')
                axs[i].fill_between(x, inters_x, 0, facecolor='none', edgecolor='blue', hatch='xx', label = '_nolegend_')
                axs[i].set_xlabel('Toxicity Classes')
                axs[i].set_ylabel('Density')

                beta_i = mlines.Line2D([], [], color='orange',  ls='-', label=r'$\mathrm{Beta}_{i}$')
                E_Beta = mlines.Line2D([], [], color='black',  ls='--', label='E_Beta({0:.2f}, {1:.2f})'.format(np.round(a_mean,2),np.round(b_mean,2)))
                int_sec = matplotlib.patches.Patch(facecolor='#DCDCDC', edgecolor='blue', hatch='xx', label='intersection \n {0:.2f} %'.format(area_inters_x * 100) )
                axs[i].legend(handles = [beta_i,E_Beta, int_sec], title=f'time {time + 1}, dose {dose + 1}', loc = 'upper right')


        plt.ylim(-0.1, 20)
        plt.suptitle('Observed vs Predicted (Across Time-Dose)', fontsize = 22)
        plt.close()
        wandb.log({"Prediction_across_dose_time_{}".format(self.title): wandb.Image(fig)})
#################################################################################################3     
# Post training evaluation
#################################################################################################3
def evaluate_model(model,title, evaluation_function, 
                   config,given_dataloader, meta_data, 
                   subjective_tox_thershold):
    
    evaluation = evaluation_function(config,title)
    ELL, EoM, EoV,true_binary_label, y_hat_prob, alphas, betas = evaluation.ELL_EoM_EoV(model,given_dataloader,config)
    
    # concatenate perdictions with metadata
    meta_data['alphas'] = alphas.reshape(-1).detach().numpy()
    meta_data['betas'] = betas.reshape(-1).detach().numpy()
    meta_data['y_binary_labels'] = true_binary_label.reshape(-1)
    meta_data['y_hat_prob'] = y_hat_prob.reshape(-1)
    
    # Here, we are using defination, if two out of 5 animals are toxic
    meta_data['y_hat_label'] = meta_data.y_hat_prob > 2/5
    meta_data['y_hat_label'] = meta_data['y_hat_label'].astype(int)

    labels = np.array(['TN',   # y_test, y_pred = 0,0
                       'FP',  # y_test, y_pred = 0,1
                       'FN',  # y_test, y_pred = 1,0
                       'TP'    # y_test, y_pred = 1,1
                      ])
    #meta_data['Confusion'] = labels[meta_data['y_binary_labels'] * 2 + meta_data['y_hat_label']]
    return ELL, EoM, EoV, meta_data


class post_training_evaluation_Mix_jnj_TG(object):
    def __init__(self, config, title):
        self.tasks = config['num_of_tasks'] 
        self.title = title
        self.subjective_tox_thershold = config["subjective_tox_thershold"]
        self.eps = 1e-5
        
        self.task_name_list =['Apop','BDH','GlyD','Ephi','GlyA','EMH','Fibro','HTHC',
                                'HTHP','IM','InMC','NecZ','Pigmen','ScNec','Vacu']
        
    def ELL_EoM_EoV(self, model, given_dataloader,config):
        
        x, d , t, self.y = next(iter(given_dataloader))
        alphas_betas = model(x,d,t)
        self.alphas, self.betas = alphas_betas[:,0:self.tasks], alphas_betas[:,self.tasks:]
        
        mask = torch.isnan(self.y)
        y_without_nan = self.y.nan_to_num(0.5)
        y_true_clipped = y_without_nan.clip(self.eps, 1 - self.eps)
        
        dist =  Beta(torch.unsqueeze(self.alphas, dim = 2),torch.unsqueeze(self.betas, dim = 2))
        self.LL = dist.log_prob(y_true_clipped)
        self.LL = self.LL[~mask]
        ELL = self.LL.mean()
        
        # EoM calculation
        pred_mean, response_mean = Beta(self.alphas, self.betas).mean, self.y.nanmean(axis =2)
        mask = torch.isnan(response_mean)
        EoM = torch.abs(pred_mean[~mask] - response_mean[~mask]).mean()
        
        # EoV calculation
        pred_var = Beta(self.alphas, self.betas).variance.detach().numpy()
        response_var = np.nanvar(self.y.detach().numpy(), axis = 2)
        mask = np.isnan(response_var)

        EoV = np.abs(pred_var[~mask] - response_var[~mask]).mean()
        observed_binary_labels, pred_prob_1 = self.compute_binary_auc()

        if config["class_dependent_Log_likelihood"] == 1:
            self.class_dependent_Log_likelihood()
            
            
        return ELL,EoM,EoV, observed_binary_labels, pred_prob_1, self.alphas, self.betas
    
    def class_ELL(self, y, class_label):
        mask = y.eq(class_label)
        class_ELL = torch.masked_select(self.LL, mask).mean()
        return class_ELL
                
    def class_dependent_Log_likelihood(self):
        toxicity_class = [0,0.2,0.4,0.6,0.8,1]
        
         # to handle sparsity
        mask = torch.isnan(self.y)
        
        for class_label in toxicity_class:
            ELL_class = self.class_ELL(self.y[~mask],class_label)
            #wandb.log({f'{self.title}_ELL_fold_class_{class_label}': np.round(ELL_class.item(),2)})
    
    def compute_binary_auc(self):

        # get binary labels
        observed_binary_labels = self.get_binary_labels(self.y)
        pred_prob_1 = self.pred_tox_prob()
        return observed_binary_labels, pred_prob_1
    
    def confusion_matrix_at_threshold(self, observed_binary_labels,pred_prob_1, th, ax):
        y_true = observed_binary_labels.ravel()
        y_pred = np.where(pred_prob_1.ravel() > th, 1, 0).ravel()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred, average='binary', pos_label= 1)
        recall_sensitivity = recall_score(y_true, y_pred, average='binary', pos_label= 1)
        specificity = recall_score(y_true, y_pred, average='binary', pos_label= 0)

        cm_highlights = np.matrix([[tp, fp, precision], [fn, tn, np.nan], [recall_sensitivity, specificity, np.nan]])
        cm_highlights = pd.DataFrame(cm_highlights, 
                                     columns = ['Observed \n present', 'Observed \n absent',''],
                                     index=['Predicted \n present', 'Predicted \n absent', ''])

        annotations = np.matrix([['TP \n'+ str(tp), 'FP \n'+ str(fp), 'precision \n TP/(TP+FP) \n'+ str(np.round(precision,2))], 
                                 ['FN \n'+ str(fn), 'TN \n'+ str(tn), np.nan], 
                                 ['recall \n sensitivity \n TP/(TP+FN)\n'+ str(np.round(recall_sensitivity,2)), 
                                  'specificity \n TN/(FP+TN)\n'+ str(np.round(specificity,2)), np.nan]])

        cmap = ListedColormap(['gray']).copy()
        cmap.set_bad('white')      # color of mask on heatmap
        cmap.set_under('white')    # color of mask on cbar

        sns.heatmap(cm_highlights, 
                         vmin = 10,
                         annot = annotations, 
                         fmt = '', 
                         cmap = cmap,
                         ax = ax, 
                         cbar  = False,
                         linewidths = 10,
                         linecolor = 'white')

        sns.heatmap(cm_highlights, 
                         vmin = 10,
                         fmt = '', 
                         cmap = cmap,
                         mask=cm_highlights < 1,
                         ax = ax,
                         cbar=False,
                         linewidths = 10,
                         linecolor = 'white')
        ax.tick_params(left=False, bottom=False, labeltop=True, labelbottom=False)
        ax.add_patch(Rectangle(xy = (0, 0), width = 1, height = 3, fill=False, edgecolor='blue', lw=5, ls = '--', alpha = 0.5))
        ax.add_patch(Rectangle(xy = (1, 0), width = 1, height = 3, fill=False, edgecolor='red', lw=5, ls = '--', alpha = 0.5))
        ax.add_patch(Rectangle(xy = (0, 0), width = 3, height = 1, fill=False, edgecolor='green', lw=5, ls = '--', alpha = 0.5))
        ax.set_title(label = 'Thershold {0:.2f}'.format(th), x = 0.35, y = -0.1)
    
    def plot_aucs(self,observed_binary_labels,pred_prob_1):
        
        fpr, tpr, precision, recall, roc_auc, average_precision = dict(),dict(),dict(), dict(), dict(), dict()
        for finding in range(num_of_tasks):

            mask = np.isnan(observed_binary_labels[:,finding])
            observed = observed_binary_labels[:,finding][~mask]
            pred = pred_prob_1[:,finding][~mask]

            fpr[finding], tpr[finding], _ = roc_curve(observed, pred, pos_label=1)
            precision[finding], recall[finding], _ = precision_recall_curve(observed, pred, pos_label=1)

            roc_auc[finding] = auc(fpr[finding], tpr[finding])
            average_precision[finding] = average_precision_score(observed, pred, pos_label=1)

        mask = np.isnan(observed_binary_labels.ravel())
        observed = observed_binary_labels.ravel()[~mask]
        pred = pred_prob_1.ravel()[~mask]   

        fpr["micro"], tpr["micro"], thresholds_roc = roc_curve(observed, pred, pos_label=1)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], thresholds_aupr = precision_recall_curve(observed, pred, pos_label=1)
        average_precision["micro"] = average_precision_score(observed, pred, average="micro")

        wandb.log({
            '{}_micro_roc_score'.format(self.title):  roc_auc["micro"],
            '{}_micro_avg_precision_score'.format(self.title):  average_precision["micro"]
        })

        ########## plot AUC-ROC ################
        plt.ioff()
        fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(16,8), constrained_layout = True)
        axs = ax.ravel()
        for i in range(num_of_tasks):
            sns.lineplot(fpr[i],tpr[i], lw=2, ci = None, alpha = 0.4, ax = axs[0],
                         label="{0} \n [{1:.2f},{2:.2f}]".format(self.task_name_list[i], roc_auc[i], average_precision[i]))

        sns.lineplot(fpr["micro"],tpr["micro"],lw=2, ax = axs[0], color = 'blue', 
                          label= "Micro_avg \n [{0:.2f},{1:.2f}]".format(roc_auc["micro"], average_precision["micro"]))
        
        # best value
        gmeans = np.sqrt(tpr["micro"] * (1-fpr["micro"]))
        ix = np.argmax(gmeans)
        th_roc = thresholds_roc[ix]
        sns.scatterplot(fpr["micro"][ix].reshape(-1), 
                        tpr["micro"][ix].reshape(-1),
                        s = 500, 
                        marker = '*',
                        ax = axs[0])

        sns.lineplot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", ax = axs[0])
        axs[0].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("FPR (1 - specificity)"), ylabel = ("Sensitivity"))
        axs[0].set_title("AUC-ROC (micro avg.)  %0.2f" % (roc_auc["micro"]))
        axs[0].legend().set_visible(False)
        sns.lineplot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", ax = axs[0])
        #axs[0].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("False Positive Rate"), ylabel = ("True Positive Rate"))
        axs[0].set_title("AUC-ROC (micro avg.)  %0.2f" % (roc_auc["micro"]))
        axs[0].legend().set_visible(False)
        ########## Plot AU-PR ###################
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color="gray",ls = '--', alpha=0.2)
            axs[1].annotate("f1={0:0.1f}".format(f_score), xy=(0.87, y[45] + 0.02))

        for i in range(num_of_tasks):
            sns.lineplot(recall[i],precision[i], lw=2, ci = None, alpha = 0.4, ax = axs[1])
        
        sns.lineplot(recall["micro"],precision["micro"], lw=2, color="blue", ax = axs[1])
        # best value
        fscore = (2 * precision["micro"] * recall["micro"]) / (precision["micro"] + recall["micro"])
        ix = np.argmax(np.nan_to_num(fscore, 0))
        th_aupr = thresholds_aupr[ix]
        sns.scatterplot(recall["micro"][ix].reshape(-1), 
                        precision["micro"][ix].reshape(-1),
                        s = 500, 
                        marker = '*',
                        ax = axs[1])
        # set the legend and the axes
        axs[1].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("Recall (sensitivity)"), ylabel = ("Precision"))
        axs[0].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("FPR (1 - specificity)"), ylabel = ("Sensitivity"))
        #axs[1].set(xlim = ([0.0, 1.0]), ylim = ([0.0, 1.05]), xlabel = ("Recall"), ylabel = ("Precision"))
        axs[1].set_title("AU-PR (micro avg.) %0.2f" % (average_precision["micro"]))
        
        # plot confusion matrix at best location
        self.confusion_matrix_at_threshold(observed_binary_labels,pred_prob_1, th_roc, axs[2])
        self.confusion_matrix_at_threshold(observed_binary_labels,pred_prob_1, th_aupr, axs[3])
        handles, labels = axs[0].get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        fig.legend(handles, labels, ncol = int(num_of_tasks/2),
                   title = 'Task [AUC-ROC, AU-PR]',title_fontsize = 15,
                      bbox_to_anchor=(0.95, 0))
        wandb.log({'{}_AUC_ROC_AUPR'.format(self.title): wandb.Image(fig)})
        plt.close()
        plt.show()

        
        
    def get_binary_labels(self,lables):
        '''
        labels format: (ndt, findings, animals)
        '''
        binary_labels = [[] for i in range(lables.shape[0])]
        for finding in range(lables.shape[1]):
            for comb in range(lables.shape[0]):

                if lables[comb,finding,:].isnan().all() == True:
                    binary_labels[comb].append(np.nan)
                else:
                    y_finding = lables[comb,finding,:]
                    #total_tested_animals = torch.count_nonzero(~torch.isnan(y_finding))
                    tox_animals = torch.count_nonzero(y_finding[~torch.isnan(y_finding)])
                    #tox_percentage = tox_animals / total_tested_animals

                    #if tox_percentage >= (self.subjective_tox_thershold):
                    if tox_animals >= self.subjective_tox_thershold:
                        binary_labels[comb].append(1)
                    else:
                        binary_labels[comb].append(0)
        binary_labels = np.array(binary_labels)
        return binary_labels
    
    def pred_tox_prob(self):
        '''
        return shape: (ndt) * task
        '''
        a, b = self.alphas.detach().numpy(), self.betas.detach().numpy() 
        prob = 1 - beta.cdf(x = 0.1, a = a, b = b)
        return prob
    
def TG_JNJ_f1_ROC_PR(metadata):
    
    TG_val = metadata[metadata.mol_set == 'TG_GATES']
    mask = np.isnan(TG_val.y_binary_labels.values)
    TG_f1_score = f1_score(TG_val.y_binary_labels.values[~mask], TG_val.y_hat_label.values[~mask])
    TG_AP_score = average_precision_score(TG_val.y_binary_labels[~mask], TG_val.y_hat_prob[~mask])
    TG_roc_auc = roc_auc_score(TG_val.y_binary_labels[~mask], TG_val.y_hat_prob[~mask])
    
    jnj_val = metadata[metadata.mol_set == 'jnj_set']
    mask = np.isnan(jnj_val.y_binary_labels.values)
    jnj_f1_score = f1_score(jnj_val.y_binary_labels.values[~mask], jnj_val.y_hat_label.values[~mask])
    jnj_AP_score = average_precision_score(jnj_val.y_binary_labels[~mask], jnj_val.y_hat_prob[~mask])
    jnj_roc_auc = roc_auc_score(jnj_val.y_binary_labels[~mask], jnj_val.y_hat_prob[~mask])
    
    return TG_f1_score,TG_roc_auc,TG_AP_score,jnj_f1_score,jnj_roc_auc,jnj_AP_score

def TG_JNJ_MiniTox_f1_ROC_PR(metadata):
    
    TG_val = metadata[metadata.mol_set == 'TG_GATES']
    mask = np.isnan(TG_val.y_binary_labels.values)
    TG_f1_score = f1_score(TG_val.y_binary_labels.values[~mask], TG_val.y_hat_label.values[~mask])
    TG_AP_score = average_precision_score(TG_val.y_binary_labels[~mask], TG_val.y_hat_prob[~mask])
    TG_roc_auc = roc_auc_score(TG_val.y_binary_labels[~mask], TG_val.y_hat_prob[~mask])
    
    jnj_val = metadata[metadata.mol_set == 'jnj_set']
    mask = np.isnan(jnj_val.y_binary_labels.values)
    jnj_f1_score = f1_score(jnj_val.y_binary_labels.values[~mask], jnj_val.y_hat_label.values[~mask])
    jnj_AP_score = average_precision_score(jnj_val.y_binary_labels[~mask], jnj_val.y_hat_prob[~mask])
    jnj_roc_auc = roc_auc_score(jnj_val.y_binary_labels[~mask], jnj_val.y_hat_prob[~mask])

    MiniTox_val = metadata[metadata.mol_set == 'MiniTox']
    mask = np.isnan(MiniTox_val.y_binary_labels.values)
    MiniTox_f1_score = f1_score(MiniTox_val.y_binary_labels.values[~mask], MiniTox_val.y_hat_label.values[~mask])
    MiniTox_AP_score = average_precision_score(MiniTox_val.y_binary_labels[~mask], MiniTox_val.y_hat_prob[~mask])
    MiniTox_roc_auc = roc_auc_score(MiniTox_val.y_binary_labels[~mask], MiniTox_val.y_hat_prob[~mask])
    
    return TG_f1_score,TG_roc_auc,TG_AP_score,jnj_f1_score,jnj_roc_auc,jnj_AP_score, MiniTox_f1_score,MiniTox_AP_score, MiniTox_roc_auc

def roc_pr_auc(model, selected_dataloader, metadata):
    x, d , t, y = next(iter(selected_dataloader))
    y_hat = model(x,d,t)
    y_hat_probs = torch.sigmoid(y_hat)

    metadata['y_binary'] = y.cpu().numpy()
    metadata['y_hat_probs'] = y_hat_probs.detach().cpu().numpy()

    y_hat_binary = np.where(y_hat_probs.detach().cpu().numpy() > 0.5, 1, 0)
    roc_auc = roc_auc_score(y.cpu().numpy(), y_hat_probs.detach().cpu().numpy())
    avg_pr_auc = average_precision_score(y.cpu().numpy(), y_hat_probs.detach().cpu().numpy())
    f1 = f1_score(y_hat_binary, y.cpu().numpy())
    return roc_auc, avg_pr_auc, f1, metadata


def AUCs_Clinical_Preclinical(y, y_prob):
        def compute_average_by_droping_nan(data_array):
            data_array = data_array[~np.isnan(data_array)].sum() / data_array.shape[0]
            data_array = np.around(data_array,2)
            return data_array
        
        roc_auc = torchmetrics.AUROC(task="binary")
        AP_score = torchmetrics.AveragePrecision(task="binary")
        Acc_Score = torchmetrics.classification.BinaryAccuracy(task="binary")
        Methew_Score = torchmetrics.classification.BinaryMatthewsCorrCoef()
        mask = (y != -1)
        # ROC_AUC and PR_AUC
        AUROC_Tasks, APScore_task, Accuracy_Score_Task, MCC_Score_Tasks = [], [], [], []

        for task in range(y.shape[1]):
            y_task_valid = y[:,task][mask[:,task]].round().to(torch.int64)
            pred_task_valid = y_prob[:,task][mask[:,task]]
            AUROC_Tasks.append(roc_auc(pred_task_valid, y_task_valid).item())
            APScore_task.append(AP_score(pred_task_valid, y_task_valid).item())
            Accuracy_Score_Task.append(Acc_Score(pred_task_valid,y_task_valid))
            MCC_Score_Tasks.append(Methew_Score(pred_task_valid,y_task_valid))
            
        AUROC_Tasks = np.array(AUROC_Tasks)
        APScore_task = np.array(APScore_task)
        Accuracy_Score_Task = np.array(Accuracy_Score_Task)
        # Modalilty wide calculations
        if y.shape[1] == 2:
          
            ROC_AUC_preclinical_global = np.around(AUROC_Tasks[0],2)
            ROC_AUC_clinical_global = np.around(AUROC_Tasks[1],2)
            
            APScore_preclinical_global = np.around(APScore_task[0],2)
            APScore_clinical_global = np.around(APScore_task[1],2)

            Acc_preclinical_global = np.around(Accuracy_Score_Task[0],2)
            Acc_clinical_global = np.around(Accuracy_Score_Task[1],2)

            MCC_preclinical_global = np.around(MCC_Score_Tasks[0],2)
            MCC_clinical_global = np.around(MCC_Score_Tasks[1],2)

            return np.around(AUROC_Tasks.mean(),2), np.around(APScore_task.mean(),2), ROC_AUC_preclinical_global,ROC_AUC_clinical_global,APScore_preclinical_global,APScore_clinical_global, Acc_preclinical_global, Acc_clinical_global, MCC_preclinical_global, MCC_clinical_global
        if y.shape[1] == 69:
            ROC_AUC_preclinical = np.around(AUROC_Tasks[:18].mean(),2)
            ROC_AUC_clinical = np.around(AUROC_Tasks[18:67].mean(),2)
            ROC_AUC_preclinical_global = np.around(AUROC_Tasks[67],2)
            ROC_AUC_clinical_global = np.around(AUROC_Tasks[68],2)

            APScore_preclinical = compute_average_by_droping_nan(APScore_task[:18])
            APScore_clinical = compute_average_by_droping_nan(APScore_task[18:67])
            APScore_All_tasks = compute_average_by_droping_nan(APScore_task)
            
            APScore_preclinical_global = np.around(APScore_task[67],2)
            APScore_clinical_global = np.around(APScore_task[68],2)
            return [np.around(AUROC_Tasks.mean(),2),APScore_All_tasks,ROC_AUC_preclinical,ROC_AUC_clinical,ROC_AUC_preclinical_global,ROC_AUC_clinical_global, APScore_preclinical, APScore_clinical, APScore_preclinical_global, APScore_clinical_global]

def AUCs_evaluation(model, selected_dataloader, metadata, config):

    selected_col = config["selected_tasks"]
    
    x, y = next(iter(selected_dataloader))
    y_hat = model(x)
    y_prob = torch.sigmoid(y_hat)
    predictions = pd.DataFrame(y_prob.detach().numpy())
    predictions.columns = selected_col
    drug_information = metadata[['Drug ID','set_type']]
    metadata = pd.concat([drug_information,predictions], axis = 1)
    AUCs = AUCs_Clinical_Preclinical(y, y_prob)
    
    return AUCs, metadata

def AUCs_evaluation_validation(metadata_binarylabels, metadata_predictions, config):

    selected_col = config["selected_tasks"]
    y = torch.from_numpy(metadata_binarylabels[selected_col].values)
    y_prob = torch.from_numpy(metadata_predictions[selected_col].values)
  
    AUCs = AUCs_Clinical_Preclinical(y, y_prob)
    
    return AUCs

def class_weights_for_complete_data(config):
    feature_target_train, feature_target_val = get_fold_Clinical_pre_Clinical(fold = 0)
    all_data = pd.concat([feature_target_train, feature_target_val]).reset_index(drop = True)
    selected_col = config['selected_tasks']

    Y = all_data[selected_col].replace(-1, np.nan)
    # Compute class weights
    class_weights = []
    for column in Y.columns:
        postive_class = (Y[column] == 1).sum()
        negative_class = (Y[column] == 0).sum()
        weights = negative_class / postive_class
        class_weights.append(weights.item())

    class_weights = torch.FloatTensor(class_weights)
    return class_weights

def get_valmetadata_with_multiple_seeds(selected_runs, folder):

    val_metadata = pd.DataFrame()
    for model_name, fold in zip(selected_runs.name.values, selected_runs.fold.values):
        model_name = 'val_metadata_pred_all_'+ model_name
        metadata = pd.read_pickle(f'{folder}/{model_name}.pkl')
        seed = model_name.split('_')[4].split('s')[-1]
        metadata['fold'] = fold
        metadata['seed'] = int(seed)
        val_metadata = pd.concat([val_metadata,metadata], ignore_index=True)

    return val_metadata

