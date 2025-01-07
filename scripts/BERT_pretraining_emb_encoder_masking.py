import sys
sys.path.insert(1, '/projects/home/mmasood1/TG GATE/')

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
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error

from tqdm import tqdm
import timeit

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
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.datasets.dataloading import get_dataloaders



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
model_weights_dir = '/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/BERT_pretraining_embeddings_model/naive_BERT/'
pretrained_model_path = '/projects/home/mmasood1/TG GATE/MolBERT/molbert/molbert_100epochs/molbert_100epochs/checkpoints/last.ckpt'
data_dir = '/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/SMILES_len_th_128/'
invitro_pos_weights = "/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/pos_weights.csv"
invivo_pos_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/18_01_2023/Filtered_data_for_BERT/BERT_filtered_preclinical_clinical_pos_weight.csv"
metadata_dir = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/BERT_finetune/MF/"
pred_dir = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/MolBERT/naive_BERT/predicitons/"

model_dir = os.path.dirname(os.path.dirname(pretrained_model_path))
hparams_path = os.path.join(model_dir, 'hparams.yaml')
# load config
with open(hparams_path) as yaml_file:
    config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

config_dict['project_name'] = "BERT_pretraining_masking_physchem_invitro"
config_dict['model_name'] = "BERT_pretraining_emb_encoder_masking"

config_dict['model_weights_dir'] = model_weights_dir
config_dict['pretrained_model_path'] = pretrained_model_path
config_dict["metadata_dir"] = metadata_dir
config_dict["pred_dir"] = pred_dir

config_dict['invitro_pos_weights'] = invitro_pos_weights
config_dict['invivo_pos_weights'] = invivo_pos_weights
config_dict['data_dir'] = data_dir
config_dict['invitro_train'] = data_dir + "train_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['invitro_val'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['invitro_test'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"

#config_dict['invitro_train'] = data_dir + "sampled_data/" + "invitro_train.pkl"
#config_dict['invitro_val'] = data_dir + "sampled_data/" + "invitro_val.pkl"
#config_dict['invitro_test'] = data_dir + "sampled_data/" + "invitro_val.pkl"

config_dict['max_epochs'] = 1
config_dict['unfreeze_epoch'] = 0
config_dict["l2_lambda"] = 0.0
config_dict['embedding_size'] = 50

config_dict['max_seq_length'] = 128
config_dict['bert_output_dim'] = 768

config_dict['optim'] = 'AdamW'#SGD
config_dict['heads_lr'] = 1e-03
config_dict["BERT_lr"] = 3e-5
config_dict["compute_classification"] = True
config_dict["save_pred"] = True

config_dict["seed"] = 42
config_dict['compute_metric_after_n_epochs'] = 5
config_dict['return_trainer'] = True
config_dict['EarlyStopping'] = False

######## invitro #########################
config_dict["invitro_batch_size"] = 256
config_dict['invitro_head_hidden_layer'] = 768

data = pd.read_pickle(config_dict['invitro_train'])
data.drop(['SMILES'], axis = 1, inplace = True)
target_names = data.columns.tolist()
config_dict["output_size"] = len(target_names)
config_dict["num_invitro_tasks"] = len(target_names)
config_dict["num_of_tasks"] = len(target_names)
config_dict["invitro_columns"] = target_names

#### Physchem ##############
config_dict["num_physchem_properties"] = 200

########### loss ####################
config_dict['mode'] = 'classification'
config_dict['missing'] = 'nan'
config_dict['alpha'] = 1.0
config_dict['beta'] = 0.0
config_dict['gamma'] = 2.0
config_dict['loss_type'] = 'Focal_loss' # 'BCE', 'Focal_loss'

############## invivo ###########################
config_dict["invivo_train"] = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/Data_for_BERT_finetuning/complete_training_set.csv"
config_dict["invivo_val"] = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/Data_for_BERT_finetuning/complete_test_set.csv"
config_dict["invivo_batch_size"] = 32
config_dict["invivo_head_hidden_layer"] = 128

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
config_dict["accelerator"] = "gpu"
config_dict["device"] = torch.device("cuda")
config_dict["precision"] = 32

config_dict["distributed_backend"] = "dp"
config_dict["pretrained_crash_model"] = None#"/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/BERT_pretraining_init_MolBERT_masking_physchem_invitro_head/epoch=99-step=0.ckpt"

# make dir to save predictions
if not os.path.exists(config_dict["pred_dir"]):
    os.makedirs(pred_dir)

# dataloaders
featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config_dict["max_seq_length"], permute = False)
config_dict["vocab_size"] = featurizer.vocab_size

invitro_train_dataloader, invitro_val_dataloader = get_dataloaders(
                                                                    featurizer = featurizer, 
                                                                    targets = "invitro", 
                                                                    num_workers = 12,
                                                                    config_dict = config_dict)

invivo_train_dataloader, invivo_val_dataloader = get_dataloaders(
                                                                    featurizer = featurizer, 
                                                                    targets = "invivo", 
                                                                    num_workers = 12,
                                                                    config_dict = config_dict)
config_dict["num_batches"] = len(invitro_train_dataloader)

from itertools import cycle

class combine_dataloaders:
    """
    A custom data loader that combines two MolBERT data loaders.
    Iterates through the first data loader and cycles through the second data loader.
    """

    def __init__(self, 
                 invitro, 
                 invivo,
                 ):
        
        self.dataloader1 = invitro
        self.dataloader2 = invivo

    def __iter__(self):
        iter1 = iter(self.dataloader1)
        iter2 = cycle(self.dataloader2)
        while True:
            try:
                batch1 = next(iter1)
                batch2 = next(iter2)
            except StopIteration:
                break

            yield (batch1,batch2)

        logging.info('Epoch finished.')

    def __len__(self):
        return len(self.dataloader1)
    
    # combine invitro and invivo dataloaders
train_dataloader = combine_dataloaders(invitro_train_dataloader, invivo_train_dataloader)
val_dataloader = combine_dataloaders(invitro_val_dataloader, invivo_val_dataloader)

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
        #self.automatic_optimization = False
        
        self.training_step_invitro_labels, self.training_step_invitro_pred = [],[]
        self.training_step_invivo_labels, self.training_step_invivo_pred = [],[]
        self.training_step_physchem_labels, self.training_step_physchem_pred = [],[]
        self.training_step_Masked_token_labels, self.training_step_Masked_token_pred = [],[]

        self.val_step_invitro_labels, self.val_step_invitro_pred = [],[]
        self.val_step_invivo_labels, self.val_step_invivo_pred = [],[]
        self.val_step_physchem_labels, self.val_step_physchem_pred = [],[]
        self.val_step_Masked_token_labels, self.val_step_Masked_token_pred = [],[]


        self.hparams = args
        self.non_weighted_creterian, self.invitro_weighted_creterien, self.invitro_FL = self.get_creterian(args, targets = "invitro")
        self.non_weighted_creterian, self.invivo_weighted_creterien, self.invivo_FL = self.get_creterian(args, targets = "invivo")

        # get model, load pretrained weights, and freeze encoder        
        model = SmilesMolbertModel(self.hparams)
        if self.hparams.pretrained_model:
            checkpoint = torch.load(self.hparams.pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'], strict = False)

        self.encoder = model.model.bert
        self.Masked_LM_task = model.model.tasks[0]
        self.Physchem_task = model.model.tasks[1]
        self.invitro_task = model.model.tasks[2]
        self.invivo_task = model.model.tasks[3]


    def forward(self, batch_inputs):
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
        embedding_params = list(self.encoder.parameters()) + \
                           list(self.Masked_LM_task.parameters())
        
        Physchem_params = self.Physchem_task.parameters()
        invitro_params = self.invitro_task.parameters()
        invivo_params = self.invivo_task.parameters()

    
        embedding_optimizer = torch.optim.AdamW(embedding_params, lr=self.hparams.BERT_lr)
        Physchem_optimizer = torch.optim.AdamW(Physchem_params, lr=self.hparams.heads_lr)
        invitro_optimizer = torch.optim.AdamW(invitro_params, lr=self.hparams.heads_lr)
        invivo_optimizer = torch.optim.AdamW(invivo_params, lr=self.hparams.heads_lr)

        embedding_scheduler = self._initialise_lr_scheduler(embedding_optimizer)
        Physchem_scheduler = self._initialise_lr_scheduler(Physchem_optimizer)
        invitro_scheduler = self._initialise_lr_scheduler(invitro_optimizer)
        invivo_scheduler = self._initialise_lr_scheduler(invivo_optimizer)
        
        return [embedding_optimizer,
                Physchem_optimizer,
                invitro_optimizer, 
                invivo_optimizer],[
                embedding_scheduler,
                Physchem_scheduler,
                invitro_scheduler,
                invivo_scheduler
                ]
    
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
                batch_labels.view(-1))  
        return loss  
    
    def Physchem_loss(self, batch_labels, batch_predictions):

        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_predictions, batch_labels)
        return loss  

    def unpack_batch(self, batch):
        batch, valid = batch
        (corrupted_batch_inputs,clean_batch_inputs),  corrupted_batch_labels = batch
        return corrupted_batch_inputs,clean_batch_inputs,corrupted_batch_labels

    def combine_invitro_invivo_batch(self, invitro_batch, invivo_batch):
        device = self.hparams.device
        # unpack invivo, invitro batch
        invitro_corrupted_inputs, invitro_clean_inputs, invitro_labels = self.unpack_batch(invitro_batch)
        invivo_corrupted_inputs, invivo_clean_inputs, invivo_labels = self.unpack_batch(invivo_batch)

        # combine inputs
        corrupted_inputs = {k: torch.cat([invitro_corrupted_inputs[k], invivo_corrupted_inputs[k]], dim=0).to(device) for k in invitro_corrupted_inputs.keys()}
        clean_inputs = {k: torch.cat([invitro_clean_inputs[k], invivo_clean_inputs[k]], dim=0).to(device) for k in invitro_clean_inputs.keys()}

        # missing labels
        missing_invitro_labels = torch.full((invivo_labels["invitro"].shape[0], 1234), -1, device = device)
        missing_invivo_labels = torch.full((invitro_labels["invitro"].shape[0], 50), -1, device = device)

        # combine labels
        batch_labels = {"lm_label_ids":torch.cat([invitro_labels["lm_label_ids"], invivo_labels["lm_label_ids"]], dim=0).to(device),
                        "unmasked_lm_label_ids":torch.cat([invitro_labels["unmasked_lm_label_ids"], invivo_labels["unmasked_lm_label_ids"]], dim=0).to(device),
                        "physchem_props":torch.cat([invitro_labels["physchem_props"], invivo_labels["physchem_props"]], dim=0).to(device),
                        "invitro":torch.cat([invitro_labels["invitro"].squeeze(), missing_invitro_labels], dim=0).to(device),
                        "invivo":torch.cat([invivo_labels["invitro"].squeeze(), missing_invivo_labels], dim=0).to(device),
                        }
        return corrupted_inputs, clean_inputs, batch_labels
    
    def step(self, batch, optimizer_idx):
        ######## get batch ##########################
        invitro_batch, invivo_batch = batch
        corrupted_batch_inputs,clean_batch_inputs, corrupted_batch_labels = self.combine_invitro_invivo_batch(invitro_batch, invivo_batch)
        
        if optimizer_idx == 0:
            Masked_token_pred, _,_,_ = self.forward(corrupted_batch_inputs)
            embedding_loss = self.MaskedLM_loss(corrupted_batch_labels["lm_label_ids"], Masked_token_pred)
            
            if self.hparams.compute_classification == True:
            # save predictions for accuracy calculations
                self.training_step_Masked_token_labels.append(corrupted_batch_labels["lm_label_ids"].long().detach().cpu())
                self.training_step_Masked_token_pred.append(Masked_token_pred.detach().cpu())

            return {"loss": embedding_loss,
                    "embedding_loss": embedding_loss,
                    "masking_loss": embedding_loss}
        
        if optimizer_idx == 1:
            _, Physchem_pred, _, _ = self.forward(clean_batch_inputs)
            physchem_loss = self.Physchem_loss(corrupted_batch_labels["physchem_props"], Physchem_pred)

            if self.hparams.compute_classification == True:
            # save predictions for accuracy calculations
                self.training_step_physchem_labels.append(corrupted_batch_labels["physchem_props"].long().detach().cpu())
                self.training_step_physchem_pred.append(Physchem_pred.detach().cpu())
  
            return {"loss": physchem_loss,
                    "physchem_loss": physchem_loss}
        
        if optimizer_idx == 2:
            _, _, invitro_pred, _ = self.forward(clean_batch_inputs)
            invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(corrupted_batch_labels["invitro"], invitro_pred, targets="invitro")
            
            if self.hparams.compute_classification == True:
            # save predictions for accuracy calculations
                self.training_step_invitro_labels.append(corrupted_batch_labels["invitro"].long().detach().cpu())
                self.training_step_invitro_pred.append(torch.sigmoid(invitro_pred.detach().cpu()))

            return {"loss": invitro_weighted_loss,
                    "invitro_weighted_loss": invitro_weighted_loss,
                    "invitro_Non_weighted_loss": invitro_Non_weighted_loss,
                    "invitro_pos_loss": invitro_pos_loss,
                    "invitro_neg_loss": invitro_neg_loss,}
        
        if optimizer_idx == 3:
            _, _, _, invivo_pred = self.forward(clean_batch_inputs)
            invivo_weighted_loss, invivo_Non_weighted_loss, invivo_pos_loss, invivo_neg_loss = self._compute_loss(corrupted_batch_labels["invivo"], invivo_pred, targets="invivo")
            
            if self.hparams.compute_classification == True:
                # save predictions for accuracy calculations
                self.training_step_invivo_labels.append(corrupted_batch_labels["invivo"].long().detach().cpu())
                self.training_step_invivo_pred.append(torch.sigmoid(invivo_pred.detach().cpu()))

            return {"loss": invivo_weighted_loss,
                    "invivo_weighted_loss": invivo_weighted_loss,
                    "invivo_Non_weighted_loss": invivo_Non_weighted_loss,
                    "invivo_pos_loss": invivo_pos_loss,
                    "invivo_neg_loss": invivo_neg_loss}

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self.step(batch, optimizer_idx)
        
    
    def training_step_end(self, outputs):
        # Define the step prefix
        # Calculate mean losses
        losses = {key: outputs[key].mean() for key in outputs.keys()}
        
        # Log the losses with WandB
        log_dict = {f'train_{key}_step': value.item() for key, value in losses.items()}
        log_dict["global_step"] = self.trainer.global_step
        wandb.log(log_dict)
        
        '''
        ################################################################
        # save checkpoint in between
        ################################################################
        interval_batches = int(0.1 * self.hparams.num_batches)
        epoch, global_step = self.trainer.current_epoch, self.trainer.global_step + 1
        if (global_step % interval_batches == 0) and (epoch == 0):
            # Log the current epoch and step for clarity
            print(f"Saving checkpoint at epoch {epoch}, step {global_step}")
            filename = f"epoch_{epoch}_step_{global_step}.ckpt"
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
            self.trainer.save_checkpoint(ckpt_path)
        '''
        return losses
        
    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            ######## get batch ##########################
            invitro_batch, invivo_batch = batch
            corrupted_batch_inputs,clean_batch_inputs, corrupted_batch_labels = self.combine_invitro_invivo_batch(invitro_batch, invivo_batch)
            
            ######## forward pass, and losses ##########
            Masked_token_pred, _,_,_ = self.forward(corrupted_batch_inputs)
            _, Physchem_pred, invitro_pred, invivo_pred = self.forward(clean_batch_inputs)

            embedding_loss = self.MaskedLM_loss(corrupted_batch_labels["lm_label_ids"], Masked_token_pred)
            physchem_loss = self.Physchem_loss(corrupted_batch_labels["physchem_props"], Physchem_pred)
            invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(corrupted_batch_labels["invitro"], invitro_pred, targets="invitro")
            invivo_weighted_loss, invivo_Non_weighted_loss, invivo_pos_loss, invivo_neg_loss = self._compute_loss(corrupted_batch_labels["invivo"], invivo_pred, targets="invivo")
            
            if self.hparams.compute_classification == True:
                # save predictions for accuracy calculations

                self.val_step_Masked_token_labels.append(corrupted_batch_labels["lm_label_ids"].long().detach().cpu())
                self.val_step_Masked_token_pred.append(Masked_token_pred.detach().cpu())

                self.val_step_invitro_labels.append(corrupted_batch_labels["invitro"].long().detach().cpu())
                self.val_step_invitro_pred.append(torch.sigmoid(invitro_pred.detach().cpu()))

                self.val_step_invivo_labels.append(corrupted_batch_labels["invivo"].long().detach().cpu())
                self.val_step_invivo_pred.append(torch.sigmoid(invivo_pred.detach().cpu()))

                self.val_step_physchem_labels.append(corrupted_batch_labels["physchem_props"].long().detach().cpu())
                self.val_step_physchem_pred.append(Physchem_pred.detach().cpu())

            return {
                "loss": embedding_loss,
                "embedding_loss": embedding_loss,
                "masking_loss": embedding_loss,
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
        # Log the losses with WandB
        log_dict = {f'{step_prefix}{key}_step': value.item() for key, value in losses.items()}
        del log_dict[f'{step_prefix}loss_step']

        log_dict["global_step"] = self.trainer.global_step
        if self.trainer.global_step > 0:
            wandb.log(log_dict)
        return losses
    
    def on_epoch_start(self):
        # Save at epoch 0
        if self.current_epoch == 0:
            print(f"Saving checkpoint before first epoch/step")
            epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            filename = f"epoch_{epoch}_step_{global_step}.ckpt"
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if self.current_epoch < self.hparams.unfreeze_epoch:
            self.freeze_network()
            print(f"freezing the network, trainable parameters = {self.count_parameters(self)}")
             
        if self.current_epoch >= self.hparams.unfreeze_epoch:
            self.unfreeze_model()
            print(f"unfreezing the network, trainable parameters = {self.count_parameters(self)}")


    def training_epoch_end(self, outputs):
        step_prefix = "train_"
        epoch, step = self.trainer.current_epoch + 1, self.trainer.global_step

        # Collect predictions and true labels for the complete training set
        invitro_labels = torch.cat(self.training_step_invitro_labels, dim=0)
        invitro_pred = torch.cat(self.training_step_invitro_pred, dim=0)
        
        invivo_labels = torch.cat(self.training_step_invivo_labels, dim=0)
        invivo_pred = torch.cat(self.training_step_invivo_pred, dim=0)

        physchem_labels = torch.cat(self.training_step_physchem_labels, dim=0)
        physchem_pred = torch.cat(self.training_step_physchem_pred, dim=0)

        masking_labels = torch.cat(self.training_step_Masked_token_labels, dim=0)
        masking_pred = torch.cat(self.training_step_Masked_token_pred, dim=0)

        print("######## training ##############")
        print("invitro",invitro_labels.shape, invitro_pred.shape)
        print("invivo",invivo_labels.shape, invivo_pred.shape)
        print("physchem",physchem_labels.shape, physchem_pred.shape)
        print("masking",masking_labels.shape, masking_pred.shape)

        # save predictions
        if self.hparams.save_pred:
            file_name = self.hparams.pred_dir + f"epoch_{epoch}_step_{step}_train_"
            torch.save(invitro_labels, file_name + "invitro_labels.pt")
            torch.save(invitro_pred, file_name + "invitro_pred.pt")
            torch.save(invivo_labels, file_name + "invivo_labels.pt")
            torch.save(invivo_pred, file_name + "invivo_pred.pt") 
            torch.save(masking_labels, file_name + "masking_labels.pt")
            torch.save(masking_pred, file_name + "masking_pred.pt") 
            torch.save(physchem_labels, file_name + "physchem_labels.pt")
            torch.save(physchem_pred, file_name + "physchem_pred.pt")


        # add step and epoch information
        log_dict = {
            "current_epoch": epoch,
            "global_step": step
        }

        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_dict.update({'learning_rate': lr})
  

        # Clear the lists to free memory for the next epoch
        self.training_step_invitro_labels.clear()
        self.training_step_invitro_pred.clear()

        self.training_step_invivo_labels.clear()
        self.training_step_invivo_pred.clear()

        self.training_step_physchem_labels.clear()
        self.training_step_physchem_pred.clear()

        self.training_step_Masked_token_labels.clear()
        self.training_step_Masked_token_pred.clear()

        del invitro_labels,invitro_pred, invivo_labels, invivo_pred, physchem_labels, physchem_pred, masking_labels, masking_pred

        wandb.log(log_dict)
        return log_dict

    def validation_epoch_end(self, outputs):
        step_prefix = "val_"
        epoch, step = self.trainer.current_epoch + 1, self.trainer.global_step + 1

        # Collect predictions and true labels for the complete validation set
        invitro_labels = torch.cat(self.val_step_invitro_labels, dim=0)
        invitro_pred = torch.cat(self.val_step_invitro_pred, dim=0)

        invivo_labels = torch.cat(self.val_step_invivo_labels, dim=0)
        invivo_pred = torch.cat(self.val_step_invivo_pred, dim=0)

        physchem_labels = torch.cat(self.val_step_physchem_labels, dim=0)
        physchem_pred = torch.cat(self.val_step_physchem_pred, dim=0)

        masking_labels = torch.cat(self.val_step_Masked_token_labels, dim=0)
        masking_pred = torch.cat(self.val_step_Masked_token_pred, dim=0)

        print("######## validation ##############")
        print("invitro",invitro_labels.shape, invitro_pred.shape)
        print("invivo",invivo_labels.shape, invivo_pred.shape)
        print("physchem",physchem_labels.shape, physchem_pred.shape)
        print("masking",masking_labels.shape, masking_pred.shape)

        # Save predictions
        if self.hparams.save_pred:
            file_name = self.hparams.pred_dir + f"epoch_{epoch}_step_{step}_val_"
            torch.save(invitro_labels, file_name + "invitro_labels.pt")
            torch.save(invitro_pred, file_name + "invitro_pred.pt")
            torch.save(invivo_labels, file_name + "invivo_labels.pt")
            torch.save(invivo_pred, file_name + "invivo_pred.pt")
            torch.save(masking_labels, file_name + "masking_labels.pt")
            torch.save(masking_pred, file_name + "masking_pred.pt")
            torch.save(physchem_labels, file_name + "physchem_labels.pt")
            torch.save(physchem_pred, file_name + "physchem_pred.pt")

        # Compute losses from predictions
        physchem_loss = self.Physchem_loss(physchem_labels, physchem_pred)
        invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(invitro_labels.float(), invitro_pred, targets="invitro")
        invivo_weighted_loss, invivo_Non_weighted_loss, invivo_pos_loss, invivo_neg_loss = self._compute_loss(invivo_labels.float(), invivo_pred, targets="invivo")
        masking_loss = self.MaskedLM_loss(masking_labels, masking_pred)
        embedding_loss = masking_loss.clone()

        # Create a dictionary to store the losses
        log_dict = {
            f'{step_prefix}embedding_loss_epoch': embedding_loss.item(),
            f'{step_prefix}masking_loss_epoch': masking_loss.item(),
            f'{step_prefix}physchem_loss_epoch': physchem_loss.item(),
            f'{step_prefix}invitro_weighted_loss_epoch': invitro_weighted_loss.item(),
            f'{step_prefix}invitro_Non_weighted_loss_epoch': invitro_Non_weighted_loss.item(),
            f'{step_prefix}invitro_pos_loss_epoch': invitro_pos_loss.item(),
            f'{step_prefix}invitro_neg_loss_epoch': invitro_neg_loss.item(),
            f'{step_prefix}invivo_weighted_loss_epoch': invivo_weighted_loss.item(),
            f'{step_prefix}invivo_Non_weighted_loss_epoch': invivo_Non_weighted_loss.item(),
            f'{step_prefix}invivo_pos_loss_epoch': invivo_pos_loss.item(),
            f'{step_prefix}invivo_neg_loss_epoch': invivo_neg_loss.item(),
        }

        # Add step and epoch information
        log_dict.update({
            "current_epoch": epoch,
            "global_step": step
        })

        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_dict.update({'learning_rate': lr})

        if self.hparams.compute_classification == True:
            
            invitro_score_list = self.compute_metrics(invitro_labels, invitro_pred, targets_type="invitro")
            invivo_score_list = self.compute_metrics(invivo_labels, invivo_pred, targets_type="invivo")
            physchem_score = self.compute_regression_metrics(physchem_labels, physchem_pred)

            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision', 'ECE_score', 'ACE_score']
                
            for i, score in enumerate(invitro_score_list):
                log_dict.update({f'val_invitro_{metric[i]}': score.item()})

            for i, score in enumerate(invivo_score_list):
                log_dict.update({f'val_invivo_{metric[i]}': score.item()})

            log_dict.update({f'val_physchem_MAE': physchem_score.item()})
        
            # Clear the lists to free memory for the next epoch
            self.val_step_invitro_labels.clear()
            self.val_step_invitro_pred.clear()

            self.val_step_invivo_labels.clear()
            self.val_step_invivo_pred.clear()

            self.val_step_physchem_labels.clear()
            self.val_step_physchem_pred.clear()

            self.val_step_Masked_token_labels.clear()
            self.val_step_Masked_token_pred.clear()

            del invitro_labels, invitro_pred, invivo_labels, invivo_pred, physchem_labels, physchem_pred, masking_labels, masking_pred

        wandb.log(log_dict)

        # Log the current epoch and step for clarity
        print(f"Saving checkpoint at epoch {epoch}, step {step}")
        filename = f"epoch_{epoch}_step_{step}.ckpt"
        ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
        self.trainer.save_checkpoint(ckpt_path)
        return log_dict


    
    def on_epoch_end(self):

        weight_norm = self.l2_regularization()
        weight_norm = {'weight_norm': weight_norm}
        wandb.log(weight_norm)

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

    def compute_regression_metrics(self, y_true, y_pred): 
        self.eval()

        targets =  y_true.cpu().detach().tolist()
        preds = y_pred.cpu().detach().tolist()

        targets = np.array(targets).reshape(-1,self.hparams.num_physchem_properties)
        preds = np.array(preds).reshape(-1,self.hparams.num_physchem_properties)
        MAE = mean_absolute_error(targets, preds)
                
        self.train()
        return MAE
   
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

    def prob_to_labels(self, pred, threshold):
	    return (pred >= threshold).astype('int')

    def unfreeze_model(self):
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def freeze_network(self):
        # List of tasks to freeze
        #tasks_to_freeze = ['encoder', 'Masked_LM_task', 'Physchem_task', 'invitro_task']
        tasks_to_freeze = ['encoder']

        
        # Iterate over the tasks and freeze their parameters
        for task_name in tasks_to_freeze:
            task = getattr(self, task_name)
            for param in task.parameters():
                param.requires_grad = False

        # Count the total number of trainable parameters
        total_trainable_params = self.count_parameters(self)

        # Count the number of trainable parameters
        heads_params = 0
        heads_params += self.count_parameters(self.Masked_LM_task)
        heads_params += self.count_parameters(self.Physchem_task)
        heads_params += self.count_parameters(self.invitro_task)
        heads_params += self.count_parameters(self.invivo_task)

        # Assert that they are equal
        assert total_trainable_params == heads_params, (
            f"Total trainable parameters ({total_trainable_params}) do not match heads parameters ({heads_params})"
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
        num_sanity_val_steps = len(val_dataloader),
        val_check_interval = 0.1, 
        #fast_dev_run = True,
        #limit_train_batches = int(10),
        #limit_val_batches = int(10)

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
                                                                val_dataloader =val_dataloader,
                                                                config = config_dict, 
                                                                model_type = 'MLP')