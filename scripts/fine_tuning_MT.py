#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, yaml
from argparse import Namespace

import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, auc, roc_curve

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    )

import wandb
os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")
import gc


# In[2]:


from molbert.models.finetune import FinetuneSmilesMolbertModel
from molbert.datasets.dataloading import MolbertDataLoader
from molbert.datasets.finetune import BertFinetuneSmilesDataset_MF
from molbert.datasets.finetune import BertFinetuneSmilesDataset

from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import average_precision_score, f1_score

def get_model_predictions_MT(model, selected_dataloader, config, ids):
    model = model.cpu() 

    y_true_list = []
    y_pred_list = []

    for batch in selected_dataloader:
        with torch.no_grad():
            (batch_inputs, batch_labels), _ = batch
            y = batch_labels["finetune"].squeeze()
            y_hat = model(batch_inputs)

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
        
    y.insert(0, "SMILES",ids)
    y_hat.insert(0, "SMILES",ids)
    return y, y_hat

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


# In[3]:

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

class Hidden_block(nn.Module):
    def __init__(self,input_dim, hidden_dim, BatchNorm1d, dropout_p, use_skip_connection):
        super(Hidden_block, self).__init__()
        self.use_batch_norm = BatchNorm1d
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.use_skip_connection = use_skip_connection

        if self.use_batch_norm:
            self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x1):
        x2 = self.layer1(x1)

        if self.use_batch_norm:
            x2 = self.batchnorm1(x2) 

        if self.use_skip_connection:
            x2 = x2 + x1             # Add skip connection
            
        x_out = torch.relu(x2)       # apply activation after addition
        x_out = self.dropout(x_out)
        return x_out
        
class MolbertModel(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        
        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]

        self.hparams = args
        self.get_creterian(args)

        # get model, load pretrained weights, and freeze encoder
        self.encoder = FinetuneSmilesMolbertModel(self.hparams)
        checkpoint = torch.load(self.hparams.pretrained_model_path, map_location=lambda storage, loc: storage)
        self.encoder.load_state_dict(checkpoint['state_dict'], strict = False)
        # Freeze model
        MolbertModel.freeze_network(self.encoder, self.hparams.freeze_level)
        self.encoder = self.encoder.model.bert

        # Model architecture
        self.input_layer = nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim)
        self.Hidden_block = nn.ModuleList([Hidden_block(self.hparams.hidden_dim, 
                                                        self.hparams.hidden_dim, 
                                                        self.hparams.BatchNorm1d, 
                                                        self.hparams.dropout_p,
                                                        self.hparams.use_skip_connection
                                                        ) for _ in range(self.hparams.depth)])
        self.output_layer = nn.Linear(self.hparams.hidden_dim, self.hparams.num_of_tasks)
        
        # dropout and Batchnorm for first layer output
        self.dropout = nn.Dropout(self.hparams.dropout_p)
        if self.hparams.BatchNorm1d:
            self.batchnorm1 = nn.BatchNorm1d(self.hparams.hidden_dim)
        
    def forward(self, batch_inputs):
        input_ids =  batch_inputs["input_ids"]
        token_type_ids = batch_inputs["token_type_ids"]
        attention_mask = batch_inputs["attention_mask"]

        _, pooled_output = self.encoder(input_ids, token_type_ids, attention_mask)
        x1 = self.input_layer(pooled_output)
        if self.hparams.BatchNorm1d:
            x1 = self.batchnorm1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout(x1)
        
        for block in self.Hidden_block:
            x_n = block(x1)  # Apply each Hidden block
        logits = self.output_layer(x_n)
        return logits
    
    def get_creterian(self, config):
        # pos weights
        
        pos_weights = pd.read_csv(config["pos_weights"])
        if self.hparams.num_of_tasks == 1:
            pos_weights = pos_weights.set_index("Targets").reindex([config["selected_tasks"]]).weights.values
        else:
            pos_weights = pos_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
        pos_weights = (config["alpha"] * pos_weights) + (1 - config["alpha"])*1
        self.pos_weights = torch.tensor(pos_weights, device = config["device"])

        # class weights
        if self.hparams.num_of_tasks > 1:
            class_weights = pd.read_csv(config["class_weights"])
            class_weights = class_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
            class_weights = (config["beta"] * class_weights) + (1 - config["beta"])*1
            self.class_weights = torch.tensor(class_weights, device = config["device"])
        else:
            self.class_weights = torch.tensor([1.0], device = config["device"])

        # train_weighted loss, validation no weights
        self.weighted_creterien =  nn.BCEWithLogitsLoss(reduction="none", 
                                                        pos_weight= self.pos_weights,
                                                        weight= self.class_weights)
        
        self.non_weighted_creterian =  nn.BCEWithLogitsLoss(reduction="none")
        self.FL = FocalLoss(gamma=config['gamma'], pos_weight= self.pos_weights)

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
            {'params': decay, 'weight_decay': self.hparams.lambda_l2}]
    
    def configure_optimizers(self):
        optimizer_grouped_parameters = self.add_weight_decay(skip_list=())

        if self.hparams.optim == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_grouped_parameters, 
                                             lr=self.hparams.learning_rate)
        if self.hparams.optim == 'Adam':
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, 
                                             lr=self.hparams.learning_rate)
        if self.hparams.optim == 'AdamW':    
            self.optimizer = AdamW(optimizer_grouped_parameters, 
                                lr=self.hparams.learning_rate, 
                                eps=self.hparams.adam_epsilon)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        T_max = 10, 
                                                                        eta_min=1e-6)
        return {"optimizer": self.optimizer, 
                "lr_scheduler": self.scheduler}

    def compute_regularization(self):
        device = torch.device('cuda')
        encoder_reg = torch.tensor(0., requires_grad=True, device=device)
        task_emb_reg = torch.tensor(0., requires_grad=True, device=device)

        # l2: Apply only on weights, exclude bias
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                encoder_reg = encoder_reg + torch.norm(param, p=2)

        # l1: Apply only on weights, exclude bias
        for name, param in self.task_embedding.named_parameters():
            if 'weight' in name:
                task_emb_reg = task_emb_reg + torch.norm(param, p=1)
                
        return encoder_reg, task_emb_reg
    
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
    
        
    def training_step(self, batch, batch_idx):
        # compute forward pass
        (batch_inputs, batch_labels), _ = batch
        y = batch_labels["finetune"].squeeze()
        y_hat = self.forward(batch_inputs)

        # compute loss
        weighted_loss, Non_weighted_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.training_step_ytrue.append(y.long().cpu())
        self.training_step_ypred.append(torch.sigmoid(y_hat).cpu())

        return {"loss": weighted_loss,
                "weighted_loss":weighted_loss,
                "Non_weighted_loss":Non_weighted_loss,
                "pos_loss":pos_loss, 
                "neg_loss":neg_loss
                }
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        (batch_inputs, batch_labels), _ = batch
        y = batch_labels["finetune"].squeeze()
        y_hat = self.forward(batch_inputs)


        # compute loss
        weighted_loss, Non_weighted_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.val_step_ytrue.append(y.long().cpu())
        self.val_step_ypred.append(torch.sigmoid(y_hat).cpu())
        return {"loss": weighted_loss,
                "weighted_loss":weighted_loss,
                "Non_weighted_loss":Non_weighted_loss,
                "pos_loss":pos_loss, 
                "neg_loss":neg_loss
                }
    
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
                

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_weighted_loss = torch.stack([x['weighted_loss'] for x in outputs]).mean()
        avg_non_weighted_loss = torch.stack([x['Non_weighted_loss'] for x in outputs]).mean()
        avg_pos_loss = torch.stack([x['pos_loss'] for x in outputs]).mean()
        avg_neg_loss = torch.stack([x['neg_loss'] for x in outputs]).mean()
        tensorboard_logs = {
                    'train_total_loss': avg_loss,
                    'train_weighted_loss': avg_weighted_loss,
                    'train_Non_weighted_loss': avg_non_weighted_loss,
                    'train_pos_loss': avg_pos_loss,
                    'train_neg_loss': avg_neg_loss
                    }
        wandb.log(tensorboard_logs)

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

        return {"avg_loss":avg_loss}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_weighted_loss = torch.stack([x['weighted_loss'] for x in outputs]).mean()
        avg_non_weighted_loss = torch.stack([x['Non_weighted_loss'] for x in outputs]).mean()
        avg_pos_loss = torch.stack([x['pos_loss'] for x in outputs]).mean()
        avg_neg_loss = torch.stack([x['neg_loss'] for x in outputs]).mean()
        tensorboard_logs = {
                    'val_total_loss': avg_loss,
                    'val_weighted_loss': avg_weighted_loss,
                    'val_Non_weighted_loss': avg_non_weighted_loss,
                    'val_pos_loss': avg_pos_loss,
                    'val_neg_loss': avg_neg_loss
                    }
        wandb.log(tensorboard_logs)

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

        n_bins = 10

        for i in range(self.hparams.num_of_tasks):
            
            # get valid targets, and convert logits to prob
            valid_targets = targets[:,i][mask[:,i]]
            valid_preds = expit(preds[:,i][mask[:,i]])
            ECE= compute_ece(valid_targets, valid_preds, n_bins=n_bins, equal_intervals = True)
            ACE = compute_ece(valid_targets, valid_preds, n_bins=n_bins, equal_intervals = False)
            ECE_score.append(ECE)
            ACE_score.append(ACE)
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
# In[4]:


# config_dict
model_weights_dir = '/projects/home/mmasood1/Model_weights/preclinical_clinical/BERT/'
pretrained_model_path = '/projects/home/mmasood1/TG GATE/MolBERT/molbert/molbert_100epochs/molbert_100epochs/checkpoints/last.ckpt'
target_dir = '/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/Data_for_BERT_finetuning/'
pos_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/06_10_2023/pos_weights.csv"
class_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/06_10_2023/target_weights.csv"
metadata_dir = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/MolBERT/BERT_MT_BCE_FL_FineTune/Final_models/"
model_dir = os.path.dirname(os.path.dirname(pretrained_model_path))
hparams_path = os.path.join(model_dir, 'hparams.yaml')

# load config
with open(hparams_path) as yaml_file:
    config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

config_dict['project_name'] = "BERT_MT_BCE_FL_FineTune_Final_models"
config_dict['model_weights_dir'] = model_weights_dir
config_dict['pretrained_model_path'] = pretrained_model_path
config_dict["metadata_dir"] = metadata_dir
config_dict['pos_weights'] = pos_weights
config_dict['class_weights'] = class_weights

config_dict['target_dir'] = target_dir
config_dict['train_file'] = target_dir + "complete_training_set.csv"
config_dict['valid_file'] = target_dir + "complete_test_set.csv"
config_dict['test_file'] = target_dir + "complete_test_set.csv"

 # architechture
config_dict["input_dim"] = 768
config_dict["hidden_dim"] = 128
config_dict["depth"] = 1
config_dict["BatchNorm1d"] = True
config_dict["use_skip_connection"] = True
config_dict["dropout_p"] = 0.2

# Training
config_dict['mode'] = 'classification'
config_dict['alpha'] = 1.0
config_dict['beta'] = 0.0
config_dict['epochs'] = 310
config_dict['unfreeze_epoch'] = 210
config_dict['output_size'] = 50
config_dict["optim"] = "Adam"
config_dict['lr_schedulers'] = "CosineAnnealingLR"
config_dict['learning_rate'] = 1e-3
config_dict["BERT_lr"] = 3e-5
config_dict["lambda_l2"] = 1e-2
config_dict["batch_size"] = 32

config_dict['missing'] = 'nan'
config_dict['compute_metric_after_n_epochs'] = 5
config_dict['return_trainer'] = True
config_dict['EarlyStopping'] = False

config_dict["accelerator"] = "gpu"
config_dict["gpu"] =  [0]
config_dict["device"] = torch.device("cuda")
config_dict["seed"] = 42
config_dict["Final_model"] = True


data = pd.read_csv(config_dict['train_file'])
try:
    data.drop(['Scafold','fold'], axis = 1, inplace = True)
except:
    pass
target_names = data.loc[:,"Cytoplasmic alteration (Basophilic/glycogen depletion)":"hepatobiliary_disorders"].columns.tolist()
config_dict["label_column"] = target_names
config_dict["num_of_tasks"] = len(target_names)
config_dict["selected_tasks"] = target_names

preclinical_tasks = config_dict["selected_tasks"][:20]
clinical_tasks = config_dict["selected_tasks"][20:]
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


config_dict['freeze_level'] = -1
# In[8]:


def wandb_init_model(model, 
                     config, 
                     train_dataloader,
                     val_dataloader, 
                     model_type):
    
    default_root_dir = config["model_weights_dir"]
    max_epochs = config["epochs"]
    return_trainer = config["return_trainer"]

    # logger
    model = model(config)
    wandb_logger = WandbLogger( 
                        name = config["model_name"],
                        save_dir = '/projects/home/mmasood1/Model_weights',
                        project= config["project_name"],
                        entity="arslan_masood", 
                        log_model='all',
                        )
    # trainer
    trainer = Trainer(
        max_epochs= int(max_epochs),
        distributed_backend= "dp",
        gpus = config["gpu"],
        #gpus = -1,
        logger = wandb_logger,
        default_root_dir=default_root_dir)

    # model fitting 
    trainer.fit(model, 
                train_dataloader = train_dataloader,
                val_dataloaders = val_dataloader,
                )
    if return_trainer:
        return model, trainer
    else:
        return model

if config_dict["Final_model"]:
    fold_list = [0]
else:
    fold_list = [0,1,2,3,4]
# In[9]:
config_dict["optim"] = "Adam"#, "AdamW"
###################################################
# setting
###################################################
config_dict["loss_type"] = "Focal_loss" #"BCE"
alpha_list = [1.0]
gamma_list = [2.0]
lambda_l2_list = [1e-1]
gpu =  3
config_dict["gpu"] =  [gpu]
config_dict["device"] = torch.device(f"cuda:{gpu}")
####################################################
for alpha in alpha_list:
    config_dict["alpha"] = alpha

    for gamma in gamma_list:
        config_dict["gamma"] = gamma

        for lambda_l2 in lambda_l2_list:
            config_dict["lambda_l2"] = lambda_l2

            y_true = pd.DataFrame()
            y_pred = pd.DataFrame()
            
            for fold in fold_list:
                config_dict["fold"] = fold 

                # get data
                if len(fold_list) == 1:
                    config_dict['train_file'] = config_dict['target_dir'] + "complete_training_set.csv"
                    config_dict['valid_file'] = config_dict['target_dir'] + "complete_test_set.csv"
                    config_dict['test_file'] = config_dict['target_dir'] + "complete_test_set.csv"
                    
                    train_ids = pd.read_csv(config_dict['train_file']).SMILES.values
                    val_ids = pd.read_csv(config_dict['valid_file']).SMILES.values
                    test_ids = pd.read_csv(config_dict['test_file']).SMILES.values
                else:
                    config_dict['train_file'] = config_dict['target_dir'] + f"train_fold{fold}.csv"
                    config_dict['valid_file'] = config_dict['target_dir'] + f"val_fold{fold}.csv"
                    config_dict['test_file'] = config_dict['target_dir'] + "complete_test_set.csv"

                    train_ids = pd.read_csv(config_dict['train_file']).SMILES.values
                    val_ids = pd.read_csv(config_dict['valid_file']).SMILES.values
                    test_ids = pd.read_csv(config_dict['test_file']).SMILES.values

                # dataloaders
                featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config_dict["max_seq_length"], permute = False)
                train_dataset = BertFinetuneSmilesDataset(
                            input_path= config_dict['train_file'],
                            featurizer=featurizer,
                            single_seq_len=config_dict["max_seq_length"],
                            total_seq_len=config_dict["max_seq_length"],
                            label_column=config_dict["label_column"],
                            is_same=False,
                            inference_mode=False,
                        )

                validation_dataset = BertFinetuneSmilesDataset(
                            input_path= config_dict['valid_file'],
                            featurizer=featurizer,
                            single_seq_len=config_dict["max_seq_length"],
                            total_seq_len=config_dict["max_seq_length"],
                            label_column=config_dict["label_column"],
                            is_same=False,
                            inference_mode=True,
                        )

                test_dataset = BertFinetuneSmilesDataset(
                            input_path= config_dict['test_file'],
                            featurizer=featurizer,
                            single_seq_len=config_dict["max_seq_length"],
                            total_seq_len=config_dict["max_seq_length"],
                            label_column=config_dict["label_column"],
                            is_same=False,
                            inference_mode=True,
                )
                ########################################################################
                train_dataloader = MolbertDataLoader(train_dataset, 
                                                    batch_size=config_dict["batch_size"],
                                                    pin_memory=False,
                                                    num_workers=4, 
                                                    #persistent_workers = True,
                                                    shuffle = True)

                validation_dataloader = MolbertDataLoader(validation_dataset, 
                                                    batch_size=config_dict["batch_size"],
                                                    pin_memory=False,
                                                    num_workers=4, 
                                                    #persistent_workers = True,
                                                    shuffle = False)

                test_dataloader = MolbertDataLoader(test_dataset, 
                                                    batch_size=config_dict["batch_size"],
                                                    pin_memory=False,
                                                    num_workers=4, 
                                                    #persistent_workers = True,
                                                    shuffle = False)
                config_dict["num_batches"] = len(train_dataloader)

                # initiate training
                config_dict["model_name"] = rf's{config_dict["seed"]}_alpha_{config_dict["alpha"]}_gamma_{config_dict["gamma"]}_{config_dict["loss_type"]}_λ{config_dict["lambda_l2"]}_{config_dict["optim"]}_f{config_dict["fold"]}'
                seed_everything(seed = config_dict["seed"])
                trained_model, trainer = wandb_init_model(model = MolbertModel, 
                                                        train_dataloader = train_dataloader,
                                                        val_dataloader =validation_dataloader,
                                                        config = config_dict, 
                                                        model_type = 'MLP')
                wandb.finish()

                
                # model evaluation
                model = trained_model.eval()
                validation_dataloader = MolbertDataLoader(validation_dataset, 
                                                    batch_size=config_dict["batch_size"],
                                                    pin_memory=False,
                                                    num_workers=0, 
                                                    #persistent_workers = True,
                                                    shuffle = False)

                y_df, y_hat_df = get_model_predictions_MT(model, validation_dataloader, config_dict, val_ids)
                y_df["fold"], y_hat_df["fold"] = config_dict["fold"], config_dict["fold"]
                y_true = pd.concat([y_true, y_df], axis = 0)
                y_pred = pd.concat([y_pred, y_hat_df], axis = 0)
            
            # after 5 folds
            data_dir = config_dict["metadata_dir"] + "predicitons/"
            result_dir = config_dict["metadata_dir"] + "Results/"  
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                os.makedirs(result_dir)

            y_true_val, y_pred_val = y_true.reset_index(drop = True), y_pred.reset_index(drop = True)
            
            # also save train pred
            train_dataloader = MolbertDataLoader(train_dataset, 
                                    batch_size=config_dict["batch_size"],
                                    pin_memory=False,
                                    num_workers=0, 
                                    #persistent_workers = True,
                                    shuffle = False)
            y_true_train, y_pred_train = get_model_predictions_MT(model, train_dataloader, config_dict, train_ids)
            
            # save predictions
            name = config_dict['model_name'].split('_f')[0] + '.csv'
            y_true_val.to_csv(data_dir + 'y_true_val_' + name, index=False)
            y_pred_val.to_csv(data_dir + 'y_pred_val_' + name, index=False)

            y_true_train.to_csv(data_dir + 'y_true_train_' + name, index=False)
            y_pred_train.to_csv(data_dir + 'y_pred_train_' + name, index=False)    

            # We should compute score fold wise, then mean
            metrics = pd.DataFrame()
            for fold in fold_list:
                
                metrics_fold = compute_binary_classification_metrics_MT(y_true = y_true_val[y_true_val.fold == fold][config_dict['selected_tasks']].values, 
                                                                y_pred_proba = expit(y_pred_val[y_pred_val.fold == fold][config_dict['selected_tasks']].values),
                                                                missing = 'nan')
                metrics_fold.insert(0, 'Tasks', target_names)
                metrics = pd.concat([metrics, metrics_fold])

            metrics = metrics.groupby("Tasks").mean().reset_index()

            # take modality-wise mean
            mean_preformances = {"pathology_mean": metrics[metrics.Tasks.isin(pathological_tasks)].iloc[:,1:].mean(),
                                "blood_mean": metrics[metrics.Tasks.isin(blood_tasks)].iloc[:,1:].mean(),
                                "clinical_mean": metrics[metrics.Tasks.isin(clinical_tasks)].iloc[:,1:].mean(),
                                "combined_all": metrics.iloc[:,1:].mean()}
            mean_preformances = pd.DataFrame(mean_preformances).T
            mean_preformances = mean_preformances.rename_axis('Tasks').reset_index()
            metrics = pd.concat([metrics, mean_preformances], ignore_index=True)    
            metrics.to_csv(result_dir + f'val_metric_' + name, index=False)

            # delete all, also clear gpu memory
            del train_dataloader, validation_dataloader, trained_model, trainer
            torch.cuda.empty_cache()
            gc.collect()

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
            gpu_memory_status = torch.cuda.memory_allocated() / (1024 ** 3)
            print("GPU Memory Status (after clearing):", gpu_memory_status)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++')


print("Script completed")




