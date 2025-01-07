import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging
logger = logging.getLogger(__name__)
import gc


# In[2]:


import os, yaml
from argparse import Namespace

import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import average_precision_score, f1_score

from tqdm import tqdm

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torch.nn.modules.loss import CrossEntropyLoss
from pytorch_lightning import seed_everything



import wandb
os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")


# In[3]:


from molbert.models.smiles import SmilesMolbertModel
from molbert.datasets.dataloading import CombinedMolbertDataLoader_cyclic, CombinedMolbertDataLoader_max, get_dataloaders
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer


# In[4]:


from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


# In[5]:


# config_dict
model_weights_dir = '/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/BERT_masking_physchem_invitro_invivo_pretraining_FL_v2/'
pretrained_model_path = '/projects/home/mmasood1/TG GATE/MolBERT/molbert/molbert_100epochs/molbert_100epochs/checkpoints/last.ckpt'
data_dir = '/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/SMILES_len_th_128/'
invitro_pos_weights = "/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/pos_weights.csv"
invivo_pos_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/18_01_2023/Filtered_data_for_BERT/BERT_filtered_preclinical_clinical_pos_weight.csv"
metadata_dir = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/BERT_finetune/MF/"
model_dir = os.path.dirname(os.path.dirname(pretrained_model_path))
hparams_path = os.path.join(model_dir, 'hparams.yaml')
# load config
with open(hparams_path) as yaml_file:
    config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

config_dict['project_name'] = "BERT_invitro_ADME_pretraining"
config_dict['model_name'] = "BERT_masking_physchem_invitro_invivo_pretraining_FL"

config_dict['model_weights_dir'] = model_weights_dir
config_dict['pretrained_model_path'] = pretrained_model_path
config_dict["metadata_dir"] = metadata_dir
config_dict['invitro_pos_weights'] = invitro_pos_weights
config_dict['invivo_pos_weights'] = invivo_pos_weights
config_dict['data_dir'] = data_dir
config_dict['invitro_train'] = data_dir + "train_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['invitro_val'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['invitro_test'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"

config_dict['mode'] = 'classification'
config_dict['alpha'] = 1.0
config_dict['beta'] = 0.0
config_dict['gamma'] = 2.0
config_dict['loss_type'] = 'Focal_loss' # 'BCE', 'Focal_loss'


config_dict['max_epochs'] = 30
config_dict['unfreeze_epoch'] = 210
config_dict["l2_lambda"] = 0.0
config_dict['embedding_size'] = 50
config_dict["num_physchem_properties"] = 200

config_dict['optim'] = 'AdamW'#SGD

config_dict['lr'] = 1e-05
config_dict["BERT_lr"] = 3e-5
config_dict["invitro_batch_size"] = 192
config_dict["invivo_batch_size"] = 32
config_dict["compute_classification"] = False
config_dict["seed"] = 42
config_dict['missing'] = 'nan'
config_dict['compute_metric_after_n_epochs'] = 5
config_dict['return_trainer'] = True
config_dict['EarlyStopping'] = False

config_dict["accelerator"] = "gpu"
config_dict["device"] = torch.device("cuda")
config_dict["precision"] = 32


data = pd.read_pickle(config_dict['invitro_train'])
data.drop(['SMILES'], axis = 1, inplace = True)
target_names = data.columns.tolist()

config_dict["output_size"] = len(target_names)
config_dict["num_invitro_tasks"] = len(target_names)
config_dict["num_of_tasks"] = len(target_names)


config_dict["invitro_columns"] = target_names
config_dict['num_mols'] = data.shape[0]
config_dict['max_seq_length'] = 128
config_dict['bert_output_dim'] = 768
config_dict['invitro_head_hidden_layer'] = 2048

############## invivo ###########################
config_dict["invivo_train"] = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/Data_for_BERT_finetuning/complete_training_set.csv"
config_dict["invivo_val"] = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/Data_for_BERT_finetuning/complete_test_set.csv"

data = pd.read_csv(config_dict['invivo_train'])
data.drop(['SMILES','Scafold','fold'], axis = 1, inplace = True)
invivo_target_names = data.columns.tolist()
config_dict["num_invivo_tasks"] = len(invivo_target_names)
config_dict["invivo_columns"] = invivo_target_names
###############################################
config_dict["permute"] = False

config_dict['pretrained_model'] = True
config_dict['freeze_level'] = False
config_dict["gpu"] = -1
config_dict["distributed_backend"] = "dp"
config_dict["pretrained_crash_model"] = "/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/BERT_masking_physchem_invitro_invivo_pretraining_FL_v2/epoch=2-step=0.ckpt"

featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config_dict["max_seq_length"], permute = False)
#elements = featurizer.load_periodic_table()[0]
#featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(max_length=config_dict["max_seq_length"], 
#                                                                allowed_elements=tuple(elements),
#                                                                permute = False)
config_dict["vocab_size"] = featurizer.vocab_size

# train dataloaders
invitro_train_dataloader_clean, invitro_train_dataloader_corrupted = get_dataloaders(
                                                                    datafile = config_dict['invitro_train'],
                                                                    featurizer = featurizer,
                                                                    targets = "invitro", 
                                                                    batch_size = config_dict["invitro_batch_size"], 
                                                                    num_workers = 8, shuffle = False,
                                                                    config_dict = config_dict)

invivo_train_dataloader_clean, invivo_train_dataloader_corrupted = get_dataloaders(
                                                                    datafile = config_dict['invivo_train'], 
                                                                    featurizer = featurizer,
                                                                    targets = "invivo", 
                                                                    batch_size = config_dict["invivo_batch_size"], 
                                                                    num_workers = 8, shuffle = False,
                                                                    config_dict = config_dict)
train_dataloader = CombinedMolbertDataLoader_cyclic(
                                             invitro_train_dataloader_clean, 
                                             invivo_train_dataloader_clean,
                                             invitro_train_dataloader_corrupted,
                                             invivo_train_dataloader_corrupted
                                             )
config_dict["num_batches"] = len(train_dataloader)

# val dataloaders
invitro_val_dataloader_clean, invitro_val_dataloader_corrupted = get_dataloaders(
                                                                    datafile = config_dict['invitro_val'],
                                                                    featurizer = featurizer,
                                                                    targets = "invitro", 
                                                                    batch_size = config_dict["invitro_batch_size"], 
                                                                    num_workers = 12, shuffle = False,
                                                                    config_dict = config_dict)

invivo_val_dataloader_clean, invivo_val_dataloader_corrupted = get_dataloaders(
                                                                    datafile = config_dict['invivo_val'], 
                                                                    featurizer = featurizer,
                                                                    targets = "invivo", 
                                                                    batch_size = config_dict["invivo_batch_size"], 
                                                                    num_workers = 12, shuffle = False,
                                                                    config_dict = config_dict)
val_dataloader = CombinedMolbertDataLoader_max(
                                             invitro_val_dataloader_clean, 
                                             invivo_val_dataloader_clean,
                                             invitro_val_dataloader_corrupted,
                                             invivo_val_dataloader_corrupted
                                             )

##########################################
config_dict["invivo_scale"] = config_dict["invivo_scale"] = 1 / (config_dict["num_batches"] * config_dict["invivo_batch_size"] / data.shape[0])

# In[7]:

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


class FocalLoss(nn.Module):
    def __init__(self, gamma, pos_weight):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.w_p = pos_weight


    def forward(self,y_pred, y_true):
        """
        Focal Loss function for binary classification.

        Arguments:
        y_true -- true binary labels (0 or 1), torch.Tensor
        y_pred -- predicted probabilities for the positive class, torch.Tensor

        Returns:
        Focal Loss
        """
        # Compute class weight
        p = torch.sigmoid(y_pred)

        # Ensure pos_weight is on the same device as y_pred
        w_p = self.w_p.to(y_pred.device)

        # Compute focal loss for positive and negative examples
        focal_loss_pos = - w_p * (1 - p) ** self.gamma * y_true * torch.log(p.clamp(min=1e-8))
        focal_loss_pos_neg = - p ** self.gamma * (1 - y_true) * torch.log((1 - p).clamp(min=1e-8))

        return focal_loss_pos + focal_loss_pos_neg
    
class MolbertModel(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()

        self.training_step_invitro_labels, self.training_step_invitro_pred = [],[]
        self.training_step_invivo_labels, self.training_step_invivo_pred = [],[]

        self.val_step_invitro_labels, self.val_step_invitro_pred = [],[]
        self.val_step_invivo_labels, self.val_step_invivo_pred = [],[]

        self.hparams = args
        self.non_weighted_creterian, self.invitro_weighted_creterien, self.invitro_FL = self.get_creterian(args, targets = "invitro")
        self.non_weighted_creterian, self.invivo_weighted_creterien, self.invivo_FL = self.get_creterian(args, targets = "invivo")


        # get model, load pretrained weights, and freeze encoder        
        model = SmilesMolbertModel(self.hparams)
        if self.hparams.pretrained_model:
            checkpoint = torch.load(self.hparams.pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'], strict = False)
        
        if self.hparams.freeze_level:
            # Freeze model
            MolbertModel.freeze_network(model, self.hparams.freeze_level)

        self.encoder = model.model.bert
        self.Masked_LM_task = model.model.tasks[0]
        self.Physchem_task = model.model.tasks[1]
        self.invitro_task = model.model.tasks[2]
        self.invivo_task = model.model.tasks[3]

        #3checkpoint = torch.load(self.hparams.pretrained_crash_model, map_location=lambda storage, loc: storage)
        #self.load_state_dict(checkpoint['state_dict'], strict = True)

    def forward(self, batch_inputs):
        #input_ids =  batch_inputs["input_ids"]
        #token_type_ids = batch_inputs["token_type_ids"]
        #attention_mask = batch_inputs["attention_mask"]

        # msking should be here 
        # masked_embedding = encoder (masked_tokens)
        # clean_embedding = encoder (cleaned_tokens)
        sequence_output, pooled_output = self.encoder(**batch_inputs)
        Masked_token_pred = self.Masked_LM_task(sequence_output, pooled_output)
        Physchem_pred = self.Physchem_task(sequence_output, pooled_output)
        invitro_pred = self.invitro_task(sequence_output, pooled_output)
        invivo_pred = self.invivo_task(sequence_output, pooled_output)

        return Masked_token_pred, Physchem_pred, invitro_pred, invivo_pred
    
    def get_creterian(self, config, targets):
        if targets == "invitro":
            pos_weights_file = config["invitro_pos_weights"]
            selected_tasks = config["invitro_columns"]
            num_of_tasks = len(selected_tasks)
            if self.hparams.beta > 0:
                class_weights_file = config["invitro_class_weights"]


        if targets == "invivo":
            pos_weights_file = config["invivo_pos_weights"]
            selected_tasks = config["invivo_columns"]
            num_of_tasks = len(selected_tasks)
            if self.hparams.beta > 0:
                class_weights_file = config["invivo_class_weights"]


        # pos weights
        if self.hparams.alpha > 0:
            pos_weights = pd.read_csv(pos_weights_file)
            if self.hparams.num_of_tasks == 1:
                pos_weights = pos_weights.set_index("Targets").reindex([selected_tasks]).weights.values
            else:
                pos_weights = pos_weights.set_index("Targets").reindex(selected_tasks).weights.values
            pos_weights = (config["alpha"] * pos_weights) + (1 - config["alpha"])*1
            pos_weights = torch.tensor(pos_weights, device = self.device)
        else:
            pos_weights = torch.tensor([1.0]* num_of_tasks, device = self.device)

        alpha_null = torch.isnan(pos_weights).any()
        assert not alpha_null, "There are null values in the pos_weight tensor"

        # class weights
        if self.hparams.beta > 0:
            if num_of_tasks > 1:
                class_weights = pd.read_csv(class_weights_file)
                class_weights = class_weights.set_index("Targets").reindex(selected_tasks).weights.values
                class_weights = (config["beta"] * class_weights) + (1 - config["beta"])*1
                class_weights = torch.tensor(class_weights, device = self.device)
            else:
                class_weights = torch.tensor([1.0], device = self.device)

            beta_null = torch.isnan(class_weights).any()
            assert not beta_null, "There are null values in the class_weight tensor"

            # train_weighted loss, validation no weights
            weighted_creterien =  nn.BCEWithLogitsLoss(reduction="none", 
                                                            pos_weight= pos_weights,
                                                            weight= class_weights)
        else:
            weighted_creterien =  nn.BCEWithLogitsLoss(reduction="none", 
                                                            pos_weight= pos_weights)
        
        FL = FocalLoss(gamma=config['gamma'], pos_weight= pos_weights)
        non_weighted_creterian =  nn.BCEWithLogitsLoss(reduction="none")

        return non_weighted_creterian, weighted_creterien, FL
    
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        
        # Apply only on weights, exclude bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def add_weight_decay(self, skip_list=()):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
            
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': self.hparams.l2_lambda}]
    
    def configure_optimizers(self):
        optimizer_grouped_parameters = self.add_weight_decay(skip_list=())

        if self.hparams.optim == 'SGD':
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, 
                                             lr=self.hparams.learning_rate)
        if self.hparams.optim == 'Adam':
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, 
                                             lr=self.hparams.learning_rate)
        if self.hparams.optim == 'AdamW':    
            optimizer = AdamW(optimizer_grouped_parameters, 
                                lr=self.hparams.learning_rate, 
                                eps=self.hparams.adam_epsilon)
        
        scheduler = self._initialise_lr_scheduler(optimizer)

        return [optimizer], [scheduler]
    
    def _initialise_lr_scheduler(self, optimizer):

        
        num_training_steps = self.hparams.num_batches // self.hparams.accumulate_grad_batches * self.hparams.max_epochs
        warmup_steps = int(num_training_steps * self.hparams.warmup_proportion)

        if self.hparams.learning_rate_scheduler == 'linear_with_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        elif self.hparams.learning_rate_scheduler == 'cosine_with_hard_restarts_warmup':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps, num_cycles=1
            )
        elif self.hparams.learning_rate_scheduler == 'cosine_schedule_with_warmup':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        elif self.hparams.learning_rate_scheduler == 'constant_schedule_with_warmup':
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

        elif self.hparams.learning_rate_scheduler == 'cosine_annealing_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, warmup_steps)
        elif self.hparams.learning_rate_scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.hparams.learning_rate_scheduler == 'constant':
            scheduler = StepLR(optimizer, 10, gamma=1.0)
        else:
            raise ValueError(
                f'learning_rate_scheduler needs to be one of '
                f'linear_with_warmup, cosine_with_hard_restarts_warmup, cosine_schedule_with_warmup, '
                f'constant_schedule_with_warmup, cosine_annealing_warm_restarts, reduce_on_plateau, '
                f'step_lr. '
                f'Given: {self.hparams.learning_rate_scheduler}'
            )

        logger.info(
            f'SCHEDULER: {self.hparams.learning_rate_scheduler} '
            f'num_batches={self.hparams.num_batches} '
            f'num_training_steps={num_training_steps} '
            f'warmup_steps={warmup_steps}'
        )

        return {'scheduler': scheduler, 'monitor': 'valid_loss', 'interval': 'step', 'frequency': 1}
    
    def _compute_loss(self, y, y_hat, targets):
        if self.hparams.num_of_tasks == 1:
            y = y.unsqueeze(1)
        # compute losses, wiht masking
        if self.hparams.missing == 'nan':
            nan_mask = torch.isnan(y)
            y[nan_mask] = -1
            #y = torch.nan_to_num(y, nan = -1), for newer version
        
        # masks
        valid_label_mask = (y != -1).float()
        pos_label_mask = (y == 1)
        negative_label_mask = (y == 0)

        if targets == "invitro":
            if self.hparams.loss_type == "BCE":
                weighted_loss = self.invitro_weighted_creterien(y_hat, y) * valid_label_mask
            if self.hparams.loss_type == "Focal_loss":
                weighted_loss = self.invitro_FL(y_hat, y)* valid_label_mask

        if targets == "invivo":
            if self.hparams.loss_type == "BCE":
                weighted_loss = self.invivo_weighted_creterien(y_hat, y) * valid_label_mask
            if self.hparams.loss_type == "Focal_loss":
                weighted_loss = self.invivo_FL(y_hat, y)* valid_label_mask

        Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask
        
        # Non_weighted_loss, positive negative loss
        pos_loss = Non_weighted_loss * pos_label_mask
        neg_loss = Non_weighted_loss * negative_label_mask
        pos_loss = pos_loss.sum() / pos_label_mask.sum()
        neg_loss = neg_loss.sum() / negative_label_mask.sum()
    
        # compute mean loss
        Non_weighted_loss = Non_weighted_loss.sum() / valid_label_mask.sum()
        weighted_loss = weighted_loss.sum() / valid_label_mask.sum()

        return weighted_loss, Non_weighted_loss, pos_loss, neg_loss
    
    def MaskedLM_loss(self, batch_labels, batch_predictions):

        loss_fn = CrossEntropyLoss(ignore_index=-1)
        vocab_size = self.hparams.vocab_size
        loss = loss_fn(batch_predictions.view(-1, vocab_size), 
                batch_labels['lm_label_ids'].view(-1))  
        return loss  
    
    def Physchem_loss(self, batch_labels, batch_predictions):

        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_predictions, batch_labels["physchem_props"])
        return loss  
        
    def extract_corrupted_cleaned_data(self, batch):
            
        device = self.hparams.device    
        invitro_clean, invivo_clean, invitro_masked, invivo_masked = batch[0], batch[1], batch[2], batch[3]
        (invitro_clean_inputs, _), _ = invitro_clean
        (invivo_clean_inputs, _), _ = invivo_clean

        (invitro_masked_inputs, invitro_masked_labels), _ = invitro_masked
        (invivo_masked_inputs, invivo_masked_labels), _ = invivo_masked

        if invivo_masked_inputs is not None:
            # two sets of inputs (each with invitro and invivo molecules)
            batch_inputs_clean = {k: torch.cat([invitro_clean_inputs[k], invivo_clean_inputs[k]], dim=0).to(device) for k in invitro_clean_inputs.keys()}
            batch_inputs_corrupted = {k: torch.cat([invitro_masked_inputs[k], invivo_masked_inputs[k]], dim=0).to(device) for k in invitro_masked_inputs.keys()}

            # concatenate batch outputs (only consider masked labels)
            invitro_labels = invitro_masked_labels["invitro"].squeeze().to(device)
            invivo_labels = invivo_masked_labels["invitro"].squeeze().to(device)

            missing_invitro_labels = torch.full((invivo_labels.shape[0], 1234), -1, device = device)
            missing_invivo_labels = torch.full((invitro_labels.shape[0], 50), -1, device = device)

            batch_labels = {"lm_label_ids":torch.cat([invitro_masked_labels["lm_label_ids"], invivo_masked_labels["lm_label_ids"]], dim=0).to(device),
                            "unmasked_lm_label_ids":torch.cat([invitro_masked_labels["unmasked_lm_label_ids"], invivo_masked_labels["unmasked_lm_label_ids"]], dim=0).to(device),
                            "physchem_props":torch.cat([invitro_masked_labels["physchem_props"], invivo_masked_labels["physchem_props"]], dim=0).to(device),
                            "invitro":torch.cat([invitro_labels, missing_invitro_labels], dim=0).to(device),
                            "invivo":torch.cat([missing_invivo_labels, invivo_labels], dim=0).to(device),
                            }
        else:
            batch_inputs_clean = {k: invitro_clean_inputs[k].to(device) for k in invitro_clean_inputs.keys()}
            batch_inputs_corrupted = {k: invitro_masked_inputs[k].to(device) for k in invitro_masked_inputs.keys()}

            invitro_labels = invitro_masked_labels["invitro"].squeeze().to(device)
            missing_invivo_labels = torch.full((invitro_labels.shape[0], 50), -1, device = device)

            batch_labels = {k: invitro_masked_labels[k].to(device) for k in invitro_masked_labels.keys()}
            batch_labels["invitro"] = invitro_labels
            batch_labels["invivo"] = missing_invivo_labels
        
        return batch_inputs_clean, batch_inputs_corrupted, batch_labels

    def training_step(self, batch, batch_idx):
        batch_inputs_clean, batch_inputs_corrupted, batch_labels = self.extract_corrupted_cleaned_data(batch)
        
        # compute forward pass with clean sequence
        _, Physchem_pred, invitro_pred, invivo_pred = self.forward(batch_inputs_clean)

        # compute forward pass with corrupted sequence
        Masked_token_pred, _, _, _ = self.forward(batch_inputs_corrupted)

        # classification loss, masking loss + MSE loss
        masking_loss = self.MaskedLM_loss(batch_labels, Masked_token_pred) 
        physchem_loss = self.Physchem_loss(batch_labels, Physchem_pred)
        invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(batch_labels["invitro"], invitro_pred, targets = "invitro") 
        invivo_weighted_loss, invivo_Non_weighted_loss, invivo_pos_loss, invivo_neg_loss = self._compute_loss(batch_labels["invivo"], invivo_pred, targets = "invivo") 
        
        # total loss
        total_loss = masking_loss + physchem_loss + invitro_weighted_loss + self.hparams.invivo_scale * invivo_weighted_loss
        
        if self.hparams.compute_classification == True:
            # save predictions for accuracy calculations
            self.training_step_invitro_labels.append(batch_labels["invitro"].long().detach().cpu())
            self.training_step_invitro_pred.append(torch.sigmoid(invitro_pred.detach().cpu()))

            self.training_step_invivo_labels.append(batch_labels["invivo"].long().detach().cpu())
            self.training_step_invivo_pred.append(torch.sigmoid(invivo_pred.detach().cpu()))
        
        return {
                "loss": total_loss,
                "masking_loss": masking_loss,
                "physchem_loss": physchem_loss,
                "invitro_weighted_loss": invitro_weighted_loss,
                "invitro_Non_weighted_loss": invitro_Non_weighted_loss,
                "invitro_pos_loss": invitro_pos_loss,
                "invitro_neg_loss": invitro_neg_loss,
                "invivo_weighted_loss": invivo_weighted_loss,
                "invivo_Non_weighted_loss": invivo_Non_weighted_loss,
                "invivo_pos_loss": invivo_pos_loss,
                "invivo_neg_loss": invivo_neg_loss
            }
    
    def training_step_end(self, outputs):
        # Define the step prefix
        step_prefix = "train_"
        
        # Calculate mean losses
        losses = {key: outputs[key].mean() for key in outputs.keys()}
        
        # Log the losses with WandB
        log_dict = {f'{step_prefix}{key}_step': value.item() for key, value in losses.items()}
        log_dict["global_step"] = self.trainer.global_step
        wandb.log(log_dict)

        ################################################################
        # save checkpoint in between
        ################################################################
        interval_batches = int(0.1 * self.hparams.num_batches)
        epoch, global_step = self.trainer.current_epoch, self.trainer.global_step + 1
        if (global_step % interval_batches == 0) and (epoch == 1):
            # Log the current epoch and step for clarity
            print(f"Saving checkpoint at epoch {epoch}, step {global_step}")
            filename = f"epoch_{epoch}_step_{global_step}.ckpt"
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
            self.trainer.save_checkpoint(ckpt_path)
        return losses
    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            batch_inputs_clean, batch_inputs_corrupted, batch_labels = self.extract_corrupted_cleaned_data(batch)
            
            # compute forward pass with clean sequence
            _, Physchem_pred, invitro_pred, invivo_pred = self.forward(batch_inputs_clean)

            # compute forward pass with corrupted sequence
            Masked_token_pred, _, _, _ = self.forward(batch_inputs_corrupted)

            # classification loss, masking loss + MSE loss
            masking_loss = self.MaskedLM_loss(batch_labels, Masked_token_pred) 
            physchem_loss = self.Physchem_loss(batch_labels, Physchem_pred)
            invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(batch_labels["invitro"], invitro_pred,targets = "invitro") 
            invivo_weighted_loss, invivo_Non_weighted_loss, invivo_pos_loss, invivo_neg_loss = self._compute_loss(batch_labels["invivo"], invivo_pred,targets = "invivo") 
            
            # total loss
            total_loss = masking_loss + physchem_loss + invitro_weighted_loss + self.hparams.invivo_scale * invivo_weighted_loss
            
            if self.hparams.compute_classification == True:
                # save predictions for accuracy calculations
                self.val_step_invitro_labels.append(batch_labels["invitro"].long().detach().cpu())
                self.val_step_invitro_pred.append(torch.sigmoid(invitro_pred.detach().cpu()))

                self.val_step_invivo_labels.append(batch_labels["invivo"].long().detach().cpu())
                self.val_step_invivo_pred.append(torch.sigmoid(invivo_pred.detach().cpu())) 
        
        return {
                "loss": total_loss,
                "masking_loss": masking_loss,
                "physchem_loss": physchem_loss,
                "invitro_weighted_loss": invitro_weighted_loss,
                "invitro_Non_weighted_loss": invitro_Non_weighted_loss,
                "invitro_pos_loss": invitro_pos_loss,
                "invitro_neg_loss": invitro_neg_loss,
                "invivo_weighted_loss": invivo_weighted_loss,
                "invivo_Non_weighted_loss": invivo_Non_weighted_loss,
                "invivo_pos_loss": invivo_pos_loss,
                "invivo_neg_loss": invivo_neg_loss
            }
    
    def validation_step_end(self, outputs):
        # Define the step prefix
        step_prefix = "val_"

       # Calculate mean losses
        losses = {key: outputs[key].mean() for key in outputs.keys()}
        
        # Filter out NaN values, invio nan after 10 steps
        filtered_losses = {key: value for key, value in losses.items() if not torch.isnan(value)}
        # Log the losses with WandB
        log_dict = {f'{step_prefix}{key}_step': value.item() for key, value in filtered_losses.items()}
        log_dict["global_step"] = self.trainer.global_step
        wandb.log(log_dict)
        return losses
    
    def on_epoch_start(self):
        # freeze, unfreeze network
        if self.current_epoch == 0:
            self.freeze_network()
            print(f"freezing the network, trainable parameters = {self.count_parameters(self)}")
            epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            print(f"Saving checkpoint before first epoch/step")
            filename = f"epoch_{epoch}_step_{global_step}.ckpt"
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
            self.trainer.save_checkpoint(ckpt_path)
        else:
            self.unfreeze_model()
            print(f"unfreezing the network, trainable parameters = {self.count_parameters(self)}")

    
    def training_epoch_end(self, outputs):

        step_prefix = "train_"
        
        # Calculate mean losses
        losses = {key: torch.stack([x[key] for x in outputs]).mean().item() for key in outputs[0].keys()}
        # Log the losses with WandB
        log_dict = {f'{step_prefix}{key}_epoch': loss for key, loss in losses.items()}
        log_dict.update({
            "current_epoch": self.trainer.current_epoch + 1,
            "global_step": self.trainer.global_step
        })

        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_dict.update({'learning_rate': lr})

        if self.hparams.compute_classification == True:
            # Collect predictions and true labels for the complete training set
            invitro_labels = torch.cat(self.training_step_invitro_labels, dim=0)
            invitro_pred = torch.cat(self.training_step_invitro_pred, dim=0)

            invivo_labels = torch.cat(self.training_step_invivo_labels, dim=0)
            invivo_pred = torch.cat(self.training_step_invivo_pred, dim=0)
            print("training epoch")
            print("invitro_labels", invitro_labels.shape)
            print("invitro_pred", invitro_pred.shape)
            print("invivo_labels", invivo_labels.shape)
            print("invivo_pred", invivo_pred.shape)

            invitro_score_list =  self.compute_metrics(invitro_labels, invitro_pred, targets_type = "invitro")
            invivo_score_list =  self.compute_metrics(invivo_labels, invivo_pred, targets_type = "invivo")

            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
                
            for i, score in enumerate(invitro_score_list):
                log_dict.update({f'train_invitro_{metric[i]}':score.item()})

            for i, score in enumerate(invivo_score_list):
                log_dict.update({f'train_invivo_{metric[i]}':score.item()})
        
        
            # Clear the lists to free memory for the next epoch
            self.training_step_invitro_labels.clear()
            self.training_step_invitro_pred.clear()
            self.training_step_invivo_labels.clear()
            self.training_step_invivo_pred.clear()
            del invitro_labels,invitro_pred, invivo_labels, invivo_pred

        wandb.log(log_dict)
        return losses
        
    def validation_epoch_end(self, outputs):

        step_prefix = "val_"
        
         # Calculate mean losses
        # Calculate mean losses
        losses = {key: torch.stack([x[key] for x in outputs]).mean().item() for key in outputs[0].keys()}

        # Log the losses with WandB
        log_dict = {f'{step_prefix}{key}_epoch': loss for key, loss in losses.items()}
        log_dict.update({
            "current_epoch": self.trainer.current_epoch + 1,
            "global_step": self.trainer.global_step
        })

        if self.hparams.compute_classification == True:
            # Collect predictions and true labels for the complete training set
            invitro_labels = torch.cat(self.val_step_invitro_labels, dim=0)
            invitro_pred = torch.cat(self.val_step_invitro_pred, dim=0)

            invivo_labels = torch.cat(self.val_step_invivo_labels, dim=0)
            invivo_pred = torch.cat(self.val_step_invivo_pred, dim=0)

            invitro_score_list =  self.compute_metrics(invitro_labels, invitro_pred, targets_type = "invitro")
            invivo_score_list =  self.compute_metrics(invivo_labels, invivo_pred, targets_type = "invivo")

            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
                
            for i, score in enumerate(invitro_score_list):
                log_dict.update({f'val_invitro_{metric[i]}':score.item()})

            for i, score in enumerate(invivo_score_list):
                log_dict.update({f'val_invivo_{metric[i]}':score.item()})
        
        
        
            # Clear the lists to free memory for the next epoch
            self.val_step_invitro_labels.clear()
            self.val_step_invitro_pred.clear()
            self.val_step_invivo_labels.clear()
            self.val_step_invivo_pred.clear()
            del invitro_labels,invitro_pred, invivo_labels, invivo_pred

        wandb.log(log_dict)
        return losses
    
    
    def on_epoch_end(self):

        weight_norm = self.l2_regularization()
        tensorboard_logs = {'weight_norm': weight_norm}
        wandb.log(tensorboard_logs)

       # Then clean the cache
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            with torch.cuda.device(f'cuda:{gpu_id}'):
                torch.cuda.empty_cache()
        # then collect the garbage
        gc.collect()
        print("!!!!!!!!! ALL CLEAR !!!!!!!!!!!!!!!!!")

    def compute_metrics(self, y_true, y_pred, targets_type): 
        self.eval()

        targets =  y_true.cpu().detach().tolist()
        preds = y_pred.cpu().detach().tolist()

        if targets_type == "invitro":
            num_of_tasks = len(self.hparams.invitro_columns)
        if targets_type == "invivo":
            num_of_tasks = len(self.hparams.invivo_columns)

        targets = np.array(targets, dtype=np.int8).reshape(-1,num_of_tasks)
        preds = np.array(preds, dtype=np.float16).reshape(-1,num_of_tasks)

        #if self.hparams.missing == 'nan':
        #    mask = ~np.isnan(targets)
        
        mask = (targets != -1)

        roc_score, blc_acc, sensitivity, specificity, AUPR, f1, average_precision = [],[],[],[],[],[],[]
        ECE_score, ACE_score = [],[]

        for i in range(num_of_tasks):
            
            try:
                # get valid targets, and convert logits to prob
                valid_targets = targets[:,i][mask[:,i]]
                valid_preds = expit(preds[:,i][mask[:,i]])
                ECE= compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = True)
                ACE = compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = False)
                ECE_score.append(ECE)
                ACE_score.append(ACE)
            except:
                ECE_score.append(np.nan)
                ACE_score.append(np.nan)

            try:
                # ROC_AUC
                fpr, tpr, th = roc_curve(valid_targets, valid_preds)
                roc_score.append(auc(fpr, tpr))

                # Balanced accuracy
                balanced_accuracy = (tpr + (1 - fpr)) / 2
                blc_acc.append(np.max(balanced_accuracy))

                # sensitivity, specificity
                optimal_threshold_index = np.argmax(balanced_accuracy)
                optimal_threshold = th[optimal_threshold_index]
                sensitivity.append(tpr[optimal_threshold_index])
                specificity.append(1 - fpr[optimal_threshold_index])

                # AUPR, F1
                precision, recall, thresholds = precision_recall_curve(valid_targets, valid_preds)
                AUPR.append(auc(recall, precision))
                f1_sc = f1_score(valid_targets, self.prob_to_labels(valid_preds, optimal_threshold))
                f1.append(f1_sc)
                average_precision.append(average_precision_score(valid_targets, valid_preds))
                
            except:
                roc_score.append(np.nan)
                AUPR.append(np.nan)
                average_precision.append(np.nan)
                #print('Performance metric is null')
                
        self.train()
        return np.nanmean(roc_score), np.nanmean(blc_acc), np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(AUPR), np.nanmean(f1), np.nanmean(average_precision),np.nanmean(ECE_score),np.nanmean(ACE_score)

    
    def prob_to_labels(self, pred, threshold):
	    return (pred >= threshold).astype('int')

    def unfreeze_model(self):
        for param in self.parameters():
            param.requires_grad = True

    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def freeze_network(self):
        # List of tasks to freeze
        tasks_to_freeze = ['encoder', 'Masked_LM_task', 'Physchem_task', 'invitro_task']
        
        # Iterate over the tasks and freeze their parameters
        for task_name in tasks_to_freeze:
            task = getattr(self, task_name)
            for param in task.parameters():
                param.requires_grad = False

        # Count the total number of trainable parameters
        total_trainable_params = self.count_parameters(self)

        # Count the number of trainable parameters in the invivo_task
        invivo_trainable_params = self.count_parameters(self.invivo_task)

        # Assert that they are equal
        assert total_trainable_params == invivo_trainable_params, (
            f"Total trainable parameters ({total_trainable_params}) do not match invivo task parameters ({invivo_trainable_params})"
        )


    def compute_interval_wise_means(self, outputs, interval_size):
        num_intervals = int(len(outputs) // interval_size)
        interval_wise_means = []

        for i in range(num_intervals):
            start_idx = i * interval_size
            end_idx = (i + 1) * interval_size

            interval_outputs = outputs[start_idx:end_idx]

            # Initialize a dictionary to accumulate sums
            interval_sums = {key: torch.tensor(0.0) for key in interval_outputs[0].keys()}

            # Sum the values for each key in the interval
            for output in interval_outputs:
                for key, value in output.items():
                    interval_sums[key] += value

            # Calculate the mean for each key
            interval_means = {key: (value / interval_size).item() for key, value in interval_sums.items()}
            
            interval_wise_means.append(interval_means)

        return interval_wise_means

from pytorch_lightning.callbacks import ModelCheckpoint

def wandb_init_model(model, 
                     config, 
                     train_dataloader,
                     val_dataloader, 
                     model_type):
    
    default_root_dir = config["model_weights_dir"]
    max_epochs = config["max_epochs"]
    return_trainer = config["return_trainer"]

    wandb.init(
            project= config["project_name"],
            dir = '/projects/home/mmasood1/Model_weights',
            entity="arslan_masood", 
            reinit = True, 
            config = config,
            name = config["model_name"],
            settings=wandb.Settings(start_method="fork"))
    
    # logger
    model = model(config)
    wandb_logger = WandbLogger( 
                        name = config["model_name"],
                        save_dir = '/projects/home/mmasood1/Model_weights',
                        project= config["project_name"],
                        entity="arslan_masood", 
                        log_model=False,
                        )
    
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
                                        filepath= default_root_dir + '{epoch}-{step}',
                                        save_top_k=-1,
                                        verbose = True)

    # trainer
    trainer = Trainer(
        max_epochs= int(max_epochs),
        distributed_backend= config["distributed_backend"],
        gpus = config["gpu"],
        logger = wandb_logger,
        precision = config_dict["precision"],
        default_root_dir=default_root_dir,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint = config["pretrained_crash_model"],
        #val_check_interval = 0.1, 
        fast_dev_run = False,
        #limit_train_batches = 0.1,
        #limit_val_batches = int(5)

        )

    # model fitting 
    trainer.fit(model, 
                train_dataloader = train_dataloader,
                val_dataloaders = val_dataloader,
                )
    if return_trainer:
        return model, trainer
    else:
        return model


# In[9]:


#config_dict["model_name"] = rf's{config_dict["seed"]}_alpha_{config_dict["alpha"]}_gamma_{config_dict["gamma"]}_{config_dict["loss_type"]}_λ{config_dict["l2_lambda"]}_{config_dict["optim"]}_Emb{config_dict["bert_output_dim"]}_Task{config_dict["num_of_tasks"]}_{config_dict["learning_rate_scheduler"]}'
seed_everything(config_dict["seed"])
trained_model, trainer = wandb_init_model(model = MolbertModel, 
                                                                train_dataloader = train_dataloader,
                                                                val_dataloader = val_dataloader,
                                                                config = config_dict, 
                                                                model_type = 'MLP')
wandb.finish()


#         
# def get_model_predictions_MT(model, selected_dataloader, config):
#     model = model.cpu() 
# 
#     y_true_list = []
#     y_pred_list = []
# 
#     for batch in tqdm(selected_dataloader):
#         with torch.no_grad():
#             (batch_inputs, batch_labels), _ = batch
#             y_invitro = batch_labels["invitro"].squeeze()
#             _, invitro_pred = model(batch_inputs)
# 
#             y_true_list.append(y_invitro.cpu())
#             y_pred_list.append(invitro_pred.cpu())
# 
#     y = torch.cat(y_true_list, dim=0)
#     y_hat = torch.cat(y_pred_list, dim=0)
# 
#     if config["num_of_tasks"] > 1:
#         y = pd.DataFrame(y.cpu().detach().numpy())
#         y_hat = pd.DataFrame(y_hat.cpu().detach().numpy())
#         y.columns = config['selected_tasks']
#         y_hat.columns = config['selected_tasks']
#     else:
#         y = pd.DataFrame({config["selected_tasks"]: y.cpu().detach().numpy()})
#         y_hat = pd.DataFrame({config["selected_tasks"]: y_hat.cpu().detach().numpy().reshape(-1)})
# 
#     return y, y_hat

# model = trained_model.eval()
# validation_dataloader = MolbertDataLoader(validation_dataset, 
#                                     batch_size=config_dict["batch_size"],
#                                     pin_memory=False,
#                                     num_workers=0, 
#                                     #persistent_workers = True,
#                                     shuffle = False)
# 
# y_df, y_hat_df = get_model_predictions_MT(model, validation_dataloader, config_dict)

# y_true_val, y_pred_val = y_df.reset_index(drop = True), y_hat_df.reset_index(drop = True)

# In[10]:


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
        

        metrics_list.append(metrics_task)
    metrics_df = pd.DataFrame(metrics_list)
    col = ['balanced_acc', 'f1_score','specificity','sensitivity', 'roc_auc','AUPR', 'average_precision']
    
    return metrics_df[col]


# metrics = compute_binary_classification_metrics_MT(y_true = y_true_val[config_dict['selected_tasks']].values, 
#                                                                 y_pred_proba = expit(y_pred_val[config_dict['selected_tasks']].values),
#                                                                 missing = 'nan')
# metrics.insert(0, 'Tasks', target_names)

# In[11]:


#checkpoint = "/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/BERT_invitro_pretraining/szaialk9/checkpoints/epoch=0.ckpt"
#checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)


# In[12]:


print("Script completed")

# In[ ]:




