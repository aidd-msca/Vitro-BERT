#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from molbert.datasets.dataloading import MolbertDataLoader
from molbert.datasets.smiles import BertSmilesDataset
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
model_weights_dir = '/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/without_physchem_head/'
pretrained_model_path = '/projects/home/mmasood1/TG GATE/MolBERT/molbert/molbert_100epochs/molbert_100epochs/checkpoints/last.ckpt'
data_dir = '/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/SMILES_len_th_128/'
pos_weights = "/projects/home/mmasood1/arslan_data_repository/invitro/invitro_1m/25_04_2024/pos_weights.csv"
metadata_dir = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/BERT_finetune/MF/"
model_dir = os.path.dirname(os.path.dirname(pretrained_model_path))
hparams_path = os.path.join(model_dir, 'hparams.yaml')
# load config
with open(hparams_path) as yaml_file:
    config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

config_dict['project_name'] = "BERT_invitro_ADME_pretraining"
config_dict['model_name'] = "ADME_Physchem_False"

config_dict['model_weights_dir'] = model_weights_dir
config_dict['pretrained_model_path'] = pretrained_model_path
config_dict["metadata_dir"] = metadata_dir
config_dict['pos_weights'] = pos_weights
config_dict['data_dir'] = data_dir
config_dict['train_file'] = data_dir + "train_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['valid_file'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['test_file'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"

config_dict['mode'] = 'classification'
config_dict['alpha'] = 0.0
config_dict['beta'] = 0.0
config_dict['gamma'] = 0.0

config_dict['max_epochs'] = 20
config_dict['unfreeze_epoch'] = 210
config_dict["l2_lambda"] = 0.0
config_dict['embedding_size'] = 50
config_dict["num_physchem_properties"] = 0

config_dict['optim'] = 'AdamW'#SGD
config_dict['loss_type'] = 'BCE'# Focal_loss

config_dict['lr'] = 1e-05
config_dict["BERT_lr"] = 3e-5
config_dict["batch_size"] = 264
config_dict["seed"] = 42



config_dict['missing'] = 'nan'
config_dict['compute_metric_after_n_epochs'] = 5
config_dict['return_trainer'] = True
config_dict['EarlyStopping'] = False

config_dict["accelerator"] = "gpu"
config_dict["device"] = torch.device("cuda")


data = pd.read_pickle(config_dict['train_file'])
data.drop(['SMILES'], axis = 1, inplace = True)
target_names = data.columns.tolist()

config_dict["output_size"] = len(target_names)
config_dict["num_invitro_tasks"] = len(target_names)
config_dict["num_of_tasks"] = len(target_names)

config_dict["label_column"] = target_names
config_dict["selected_tasks"] = target_names
config_dict['num_mols'] = data.shape[0]
config_dict['max_seq_length'] = 128
config_dict['bert_output_dim'] = 768
config_dict['invitro_head_hidden_layer'] = 2048

config_dict["permute"] = False

config_dict['pretrained_model'] = True
config_dict['freeze_level'] = False
config_dict["gpu"] = -1
config_dict["precision"] = 32
config_dict["distributed_backend"] = "dp"
config_dict["pretrained_crash_model"] = None #"/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k/invitro_with_PhysChem/epoch=2-val_f1_score=0.00.ckpt"


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
            label_column= config_dict["label_column"],
            is_same= config_dict["is_same_smiles"],
            num_invitro_tasks = config_dict["num_invitro_tasks"],
            num_physchem= config_dict["num_physchem_properties"],
            permute= config_dict["permute"],
            named_descriptor_set=config_dict["named_descriptor_set"],
            inference_mode = False
        )

validation_dataset = BertSmilesDataset(
            input_path= config_dict['valid_file'],
            featurizer= featurizer,
            single_seq_len= config_dict["max_seq_length"],
            total_seq_len= config_dict["max_seq_length"],
            label_column= config_dict["label_column"],
            is_same= config_dict["is_same_smiles"],
            num_invitro_tasks = config_dict["num_invitro_tasks"],
            num_physchem= config_dict["num_physchem_properties"],
            permute= config_dict["permute"],
            named_descriptor_set=config_dict["named_descriptor_set"],
            inference_mode = True

        )

test_dataset = BertSmilesDataset(
            input_path= config_dict['test_file'],
            featurizer= featurizer,
            single_seq_len= config_dict["max_seq_length"],
            total_seq_len= config_dict["max_seq_length"],
            label_column= config_dict["label_column"],
            is_same= config_dict["is_same_smiles"],
            num_invitro_tasks = config_dict["num_invitro_tasks"],
            num_physchem= config_dict["num_physchem_properties"],
            permute= config_dict["permute"],
            named_descriptor_set=config_dict["named_descriptor_set"],
            inference_mode = True

)

train_dataloader = MolbertDataLoader(train_dataset, 
                                    batch_size=config_dict["batch_size"],
                                    pin_memory=False,
                                    num_workers=36, 
                                    shuffle = True)

validation_dataloader = MolbertDataLoader(validation_dataset, 
                                    batch_size=config_dict["batch_size"],
                                    pin_memory=False,
                                    num_workers=24, 
                                    shuffle = False)

test_dataloader = MolbertDataLoader(test_dataset, 
                                    batch_size=config_dict["batch_size"],
                                    pin_memory=False,
                                    num_workers=24, 
                                    shuffle = False)

config_dict["num_batches"] = len(train_dataloader)
                                                     


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

        # Compute focal loss for positive and negative examples
        focal_loss_pos = - self.w_p * (1 - p) ** self.gamma * y_true * torch.log(p.clamp(min=1e-8))
        focal_loss_pos_neg = - p ** self.gamma * (1 - y_true) * torch.log((1 - p).clamp(min=1e-8))

        return focal_loss_pos + focal_loss_pos_neg
    
class MolbertModel(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        
        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]

        self.hparams = args
        self.get_creterian(args)

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
        #self.Physchem_task = model.model.tasks[1]
        self.invitro_task = model.model.tasks[1]

        #3checkpoint = torch.load(self.hparams.pretrained_crash_model, map_location=lambda storage, loc: storage)
        #self.load_state_dict(checkpoint['state_dict'], strict = True)

    def forward(self, batch_inputs):
        #input_ids =  batch_inputs["input_ids"]
        #token_type_ids = batch_inputs["token_type_ids"]
        #attention_mask = batch_inputs["attention_mask"]

        sequence_output, pooled_output = self.encoder(**batch_inputs)
        Masked_token_pred = self.Masked_LM_task(sequence_output, pooled_output)
        #Physchem_pred = self.Physchem_task(sequence_output, pooled_output)
        invitro_pred = self.invitro_task(sequence_output, pooled_output)

        return Masked_token_pred, invitro_pred
    
    def get_creterian(self, config):
        # pos weights
    
        if self.hparams.alpha > 0:
            pos_weights = pd.read_csv(config["pos_weights"])
            if self.hparams.num_of_tasks == 1:
                pos_weights = pos_weights.set_index("Targets").reindex([config["selected_tasks"]]).weights.values
            else:
                pos_weights = pos_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
            pos_weights = (config["alpha"] * pos_weights) + (1 - config["alpha"])*1
            self.pos_weights = torch.tensor(pos_weights, device = config["device"])
        else:
            self.pos_weights = torch.tensor([1.0]* config_dict["num_of_tasks"], device = config["device"])

        alpha_null = torch.isnan(self.pos_weights).any()
        assert not alpha_null, "There are null values in the pos_weight tensor"

        # class weights
        if self.hparams.beta > 0:
            if self.hparams.num_of_tasks > 1:
                class_weights = pd.read_csv(config["class_weights"])
                class_weights = class_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
                class_weights = (config["beta"] * class_weights) + (1 - config["beta"])*1
                self.class_weights = torch.tensor(class_weights, device = config["device"])
            else:
                self.class_weights = torch.tensor([1.0], device = config["device"])

            beta_null = torch.isnan(self.class_weights).any()
            assert not beta_null, "There are null values in the class_weight tensor"

            # train_weighted loss, validation no weights
            self.weighted_creterien =  nn.BCEWithLogitsLoss(reduction="none", 
                                                            pos_weight= self.pos_weights,
                                                            weight= self.class_weights)
        else:
            self.weighted_creterien =  nn.BCEWithLogitsLoss(reduction="none", 
                                                            pos_weight= self.pos_weights)
        
        self.FL = FocalLoss(gamma=config['gamma'], pos_weight= self.pos_weights)
        self.non_weighted_creterian =  nn.BCEWithLogitsLoss(reduction="none")

    
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
    
    def _compute_loss(self, y, y_hat):
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

        if self.hparams.loss_type == "BCE":
            weighted_loss = self.weighted_creterien(y_hat, y) * valid_label_mask
        if self.hparams.loss_type == "Focal_loss":
            weighted_loss = self.FL(y_hat, y)* valid_label_mask
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
        
    def training_step(self, batch, batch_idx):
        step_name = "train"
        # compute forward pass
        (batch_inputs, batch_labels), valid = batch
        self.logger.log_metrics({f"{step_name}_valid_SMILES":valid.sum().item()}, step = True)

        y_invitro = batch_labels["invitro"].squeeze()
        Masked_token_pred, invitro_pred = self.forward(batch_inputs)

        # classification loss, masking loss + MSE loss
        weighted_loss, Non_weighted_loss, pos_loss, neg_loss = self._compute_loss(y_invitro, invitro_pred) 
        masking_loss = self.MaskedLM_loss(batch_labels, Masked_token_pred) 
        #physchem_loss = self.Physchem_loss(batch_labels, Physchem_pred)

        total_loss = weighted_loss + masking_loss #+ physchem_loss
        
        self.training_step_ytrue.append(y_invitro.long().detach().cpu())
        self.training_step_ypred.append(torch.sigmoid(invitro_pred.detach().cpu()))

        self.logger.log_metrics({f"{step_name}_total_loss":total_loss.item()}, step = True)
        self.logger.log_metrics({f"{step_name}_weighted_loss":weighted_loss.item()}, step = True)
        self.logger.log_metrics({f"{step_name}_masking_loss":masking_loss.item()}, step = True)
        #self.logger.log_metrics({f"{step_name}_physchem_loss":physchem_loss.item()}, step = True)
        self.logger.log_metrics({f"{step_name}_Non_weighted_loss":Non_weighted_loss.item()}, step = True)
        
        #self.logger.log_metrics({f"{step_name}_pos_loss":pos_loss}, step = True)
        #self.logger.log_metrics({f"{step_name}_neg_loss":neg_loss}, step = True)


        return {"loss": total_loss}
    
    def validation_step(self, batch, batch_idx):
        step_name = "val"
        with torch.no_grad():
            (batch_inputs, batch_labels), _ = batch
            y_invitro = batch_labels["invitro"].squeeze()
            Masked_token_pred, invitro_pred = self.forward(batch_inputs)

            # classification loss, masking loss + MSE loss
            weighted_loss, Non_weighted_loss, pos_loss, neg_loss = self._compute_loss(y_invitro, invitro_pred) 
            masking_loss = self.MaskedLM_loss(batch_labels, Masked_token_pred) 
            #physchem_loss = self.Physchem_loss(batch_labels, Physchem_pred)
            total_loss = weighted_loss + masking_loss #+ physchem_loss

            self.val_step_ytrue.append(y_invitro.long().detach().cpu())
            self.val_step_ypred.append(torch.sigmoid(invitro_pred).detach().cpu())

        self.logger.log_metrics({f"{step_name}_total_loss":total_loss.item()}, step = True)
        self.logger.log_metrics({f"{step_name}_weighted_loss":weighted_loss.item()}, step = True)
        self.logger.log_metrics({f"{step_name}_masking_loss":masking_loss.item()}, step = True)
        #self.logger.log_metrics({f"{step_name}_physchem_loss":physchem_loss.item()}, step = True)

        self.logger.log_metrics({f"{step_name}_Non_weighted_loss":Non_weighted_loss.item()}, step = True)
        #self.logger.log_metrics({f"{step_name}_pos_loss":pos_loss}, step = True)
        #self.logger.log_metrics({f"{step_name}_neg_loss":neg_loss}, step = True)
        return {f'{step_name}_loss':total_loss}
    '''
    def on_epoch_start(self):
        # Check if current epoch is greater than or equal to the desired epoch to unfreeze
        if self.current_epoch >= self.hparams.unfreeze_epoch:
            self.unfreeze_model()

            # Decrease the learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.hparams.BERT_lr
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        last_epoch = self.scheduler.last_epoch,
                                                                        T_max = 10, 
                                                                        eta_min=1e-6)
    '''        

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        wandb.log({f'train_loss_epoch':avg_loss.item()})

        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        wandb.log({'learning_rate': lr})
        
        # Collect predictions and true labels for the complete training set
        train_true = torch.cat(self.training_step_ytrue, dim=0)
        train_preds = torch.cat(self.training_step_ypred, dim=0)

        score_list =  self.compute_metrics(train_true, train_preds)
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
        for i, score in enumerate(score_list):
            wandb.log({f'train_{metric[i]}':score.item()})
        
        # Clear the lists to free memory for the next epoch
        self.training_step_ytrue.clear()
        self.training_step_ypred.clear()
        del train_true,train_preds

        return {'train_loss':avg_loss}
        
    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        wandb.log({f'val_loss_epoch':avg_loss.item()})
        #Collect predictions and true labels for the complete training set
        val_true = torch.cat(self.val_step_ytrue, dim=0)
        val_preds = torch.cat(self.val_step_ypred, dim=0)

        score_list =  self.compute_metrics(val_true,val_preds)
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
        for i, score in enumerate(score_list):
            wandb.log({f'val_{metric[i]}':score.item()})

        # Clear the lists to free memory for the next epoch
        self.val_step_ytrue.clear()
        self.val_step_ypred.clear()
        del val_true, val_preds

        return {'val_loss':avg_loss}
    
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

    def compute_metrics(self, y_true, y_pred): 
        self.eval()

        targets =  y_true.cpu().detach().tolist()
        preds = y_pred.cpu().detach().tolist()

        targets = np.array(targets).reshape(-1,self.hparams.num_of_tasks)
        preds = np.array(preds).reshape(-1,self.hparams.num_of_tasks)

        #if self.hparams.missing == 'nan':
        #    mask = ~np.isnan(targets)
        
        mask = (targets != -1)

        roc_score, blc_acc, sensitivity, specificity, AUPR, f1, average_precision = [],[],[],[],[],[],[]
        ECE_score, ACE_score = [],[]

        for i in range(self.hparams.num_of_tasks):
            
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

    @staticmethod
    def freeze_network(model, freeze_level: int):
        """
        Freezes specific layers of the model depending on the freeze_level argument:

         0: freeze nothing
        -1: freeze all BERT weights but not the task head
        -2: freeze the pooling layer
        -3: freeze the embedding layer
        -4: freeze the task head but not the base layer
        n>0: freeze the bottom n layers of the base model.
        """

        model_bert = model.model.bert
        model_tasks = model.model.tasks

        model_bert_encoder = model.model.bert.encoder
        model_bert_pooler = model.model.bert.pooler
        model_bert_embeddings = model.model.bert.embeddings

        if freeze_level == 0:
            # freeze nothing
            return

        elif freeze_level > 0:
            # freeze the encoder/transformer
            n_encoder_layers = len(model_bert_encoder.layer)

            # we'll always freeze layers bottom up - starting from layers closest to the embeddings
            frozen_layers = min(freeze_level, n_encoder_layers)
            #
            for i in range(frozen_layers):
                layer = model_bert_encoder.layer[i]
                for param in layer.parameters():
                    param.requires_grad = False

        elif freeze_level == -1:
            # freeze everything bert
            for param in model_bert.parameters():
                param.requires_grad = False

        elif freeze_level == -2:
            # freeze the pooling layer
            for param in model_bert_pooler.parameters():
                param.requires_grad = False

        elif freeze_level == -3:
            # freeze the embedding layer
            for param in model_bert_embeddings.parameters():
                param.requires_grad = False

        elif freeze_level == -4:
            # freeze the task head
            for param in model_tasks.parameters():
                param.requires_grad = False

# In[8]:


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
                                        filepath= default_root_dir + '{epoch}-{val_f1_score:.2f}',
                                        save_top_k=-1)

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
        #fast_dev_run = True,
        #limit_train_batches = 5,
        #limit_val_batches = 5

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
                                                                val_dataloader =validation_dataloader,
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




