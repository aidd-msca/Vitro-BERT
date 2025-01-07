import numpy as np
import pandas as pd
from typing import List, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.beta import Beta

import torchmetrics
from torchmetrics.classification import MultilabelAveragePrecision

import sklearn
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef, roc_curve
#from netcal.metrics import ECE,ACE

from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from libs.BNN import BayesianLinear, kl_divergence_from_nn
from libs.utils import class_weights_for_complete_data, class_weights_for_SIDER, compute_weighted_metrics
from libs.utils import compute_binary_classification_metrics_MT, compute_AUPR_by_ignoring_missing_values
from libs.model_utils import NoamLR
from scipy.special import softmax
#from simpletransformers.classification import ClassificationModel

# Import necessary modules from chemprop
#from chemprop.models.mpn import MPN
#from chemprop.models.ffn import build_ffn
#from chemprop.nn_utils import initialize_weights

from libs.utils import compute_ece


import torch
import torch.nn as nn

def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        BCE_loss: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Sourcecode: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = BCE_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss

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
    
class Vanilla_MLP_classifier(pl.LightningModule):
    def __init__(self, config):
        super(Vanilla_MLP_classifier, self).__init__()

        self.train_step_pos_loss = []
        self.train_step_neg_loss = []
        self.val_step_pos_loss = []
        self.val_step_neg_loss = []

        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.depth = int(config['depth'])
        self.num_of_tasks = config['num_of_tasks']
        self.BatchNorm1d = config['BatchNorm1d']
        self.dropout_p = config['dropout_p']
        self.use_skip_connection = config['use_skip_connection']
        self.loss_type = config['loss_type']
        self.optim = config['optim']
        self.lr = config['lr']
        self.lr_schedulers = config["lr_schedulers"]
        self.epochs = config["epochs"]
        self.compute_metric_after_n_epochs = config["compute_metric_after_n_epochs"]

        self.l2_lambda = config['l2_lambda']
        self.optm_l2_lambda = config['optm_l2_lambda']
        self.batch_size = config['batch_size']
        self.missing = config["missing"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]

        self.thresholds = np.linspace(0,1,20)

        # pos weights
        pos_weights = pd.read_csv(config["pos_weights"])
        if config["num_of_tasks"] == 1:
            pos_weights = pos_weights.set_index("Targets").reindex([config["selected_tasks"]]).weights.values
        else:
            pos_weights = pos_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
        
        pos_weights = (config["alpha"] * pos_weights) + (1 - config["alpha"])*1
        self.pos_weights = torch.tensor(pos_weights, device = config["device"])

        # class weights
        if config['num_of_tasks'] > 1:
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

        # Model architecture
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.Hidden_block = nn.ModuleList([Hidden_block(self.hidden_dim, 
                                                        self.hidden_dim, 
                                                        self.BatchNorm1d, 
                                                        self.dropout_p,
                                                        self.use_skip_connection
                                                        ) for _ in range(self.depth)])
        self.output_layer = nn.Linear(self.hidden_dim, self.num_of_tasks)
        
        # dropout and Batchnorm for first layer output
        self.dropout = nn.Dropout(self.dropout_p)
        if self.BatchNorm1d:
            self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, x_input):
        x1 = self.input_layer(x_input)
        if self.BatchNorm1d:
            x1 = self.batchnorm1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout(x1)
        
        for block in self.Hidden_block:
            x_n = block(x1)  # Apply each Hidden block
        x_output = self.output_layer(x_n)
        return x_output
    
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
            {'params': decay, 'weight_decay': self.optm_l2_lambda}]

    def configure_optimizers(self):
        optimizer_grouped_parameters = self.add_weight_decay(skip_list=())
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_grouped_parameters, 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, 
                                             lr=self.lr)
        if self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                             lr=self.lr)
        
        if self.lr_schedulers == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        T_max = 10, 
                                                                        eta_min=1e-6) 
            return {"optimizer": self.optimizer, 
                    "lr_scheduler": self.scheduler}
        
        if self.lr_schedulers == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        verbose=True,
                                                                        patience=15,
                                                                        min_lr=1e-6,
                                                                        mode = 'min')
            return {
            'optimizer':  self.optimizer,
            'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
            'monitor': 'val_BCE_loss'
            }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        
        # Apply only on weights, exclude bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _compute_loss(self, y, y_hat):
        if self.num_of_tasks == 1:
            y = y.unsqueeze(1)
        # compute losses, wiht masking
        if self.missing == 'nan':
            y = torch.nan_to_num(y, nan = -1)
        
        # masks
        valid_label_mask = (y != -1).float()
        pos_label_mask = (y == 1)
        negative_label_mask = (y == 0)

        if self.loss_type == "BCE":
            weighted_loss = self.weighted_creterien(y_hat, y) * valid_label_mask
        if self.loss_type == "Focal_loss":
            weighted_loss = self.FL(y_hat, y)* valid_label_mask
        Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask

        if self.loss_type == 'Focal_loss_v2':
            Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask
            weighted_loss = sigmoid_focal_loss(inputs = y_hat,
                                                targets = y,
                                                BCE_loss = Non_weighted_loss,
                                                alpha = self.alpha,
                                                gamma = self.gamma)
            weighted_loss = weighted_loss * valid_label_mask
        
        # Non_weighted_loss, positive negative loss
       
        pos_loss = Non_weighted_loss * pos_label_mask
        neg_loss = Non_weighted_loss * negative_label_mask
        pos_loss = pos_loss.sum() / pos_label_mask.sum()
        neg_loss = neg_loss.sum() / negative_label_mask.sum()
    

        # compute mean loss
        Non_weighted_loss = Non_weighted_loss.sum() / valid_label_mask.sum()
        weighted_loss = weighted_loss.sum() / valid_label_mask.sum()

        weight_norm = self.l2_regularization()
        l2_reg_loss = self.l2_lambda*weight_norm
        total_loss = weighted_loss + l2_reg_loss

        return total_loss, weighted_loss, Non_weighted_loss,weight_norm, pos_loss, neg_loss

    def training_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        
        y_hat = self(x)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,weight_norm, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.train_step_pos_loss.append(pos_loss.item())
        self.train_step_neg_loss.append(neg_loss.item())

        self.log('train_BCE_weighted', weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_BCE_non_weighted', Non_weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_weight_norm', weight_norm, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,weight_norm, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.val_step_pos_loss.append(pos_loss.item())
        self.val_step_neg_loss.append(neg_loss.item())

        self.log('val_BCE_weighted', weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_BCE_non_weighted', Non_weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_weight_norm', weight_norm, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):

        pos_loss = torch.tensor(self.train_step_pos_loss)
        neg_loss = torch.tensor(self.train_step_neg_loss)
        geometric_mean = torch.sqrt(pos_loss.nanmean() * neg_loss.nanmean())

        self.log('train_BCE_pos', pos_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('train_BCE_neg', neg_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('train_gm_loss', geometric_mean,on_step=False, on_epoch=True)
    
        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr,on_step=False, on_epoch=True)
        
        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.train_dataloader)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
            for i, score in enumerate(score_list):
                self.log(f'train_{metric[i]}', score, prog_bar=True , on_epoch=True)

        self.train_step_pos_loss.clear()
        self.train_step_neg_loss.clear()


    def on_validation_epoch_end(self):

        pos_loss = torch.tensor(self.val_step_pos_loss)
        neg_loss = torch.tensor(self.val_step_neg_loss)
        geometric_mean = torch.sqrt(pos_loss.nanmean() * neg_loss.nanmean())
        
        self.log('val_BCE_pos', pos_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('val_BCE_neg', neg_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('val_gm_loss', geometric_mean,on_step=False, on_epoch=True)


        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.val_dataloaders)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
            for i, score in enumerate(score_list):
                self.log(f'val_{metric[i]}', score, prog_bar=True , on_epoch=True)

        self.val_step_pos_loss.clear()
        self.val_step_neg_loss.clear()
           
    def compute_metrics(self, dataloader): 
        device = torch.device("cuda") 
        self.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:

                batch_x,batch_targets = batch
                batch_preds = self(batch_x.to(device))

                preds.extend(batch_preds.cpu().detach().tolist())
                targets.extend(batch_targets.cpu().detach().tolist())

            targets = np.array(targets).reshape(-1,self.num_of_tasks)
            preds = np.array(preds).reshape(-1,self.num_of_tasks)

            if self.missing == 'nan':
               mask = ~np.isnan(targets)

            roc_score, blc_acc, sensitivity, specificity, AUPR, f1, average_precision = [],[],[],[],[],[],[]
            ECE_score, ACE_score = [],[]

            n_bins = 10

            for i in range(self.num_of_tasks):
                
                # get valid targets, and convert logits to prob
                valid_targets = targets[:,i][mask[:,i]]
                valid_preds = expit(preds[:,i][mask[:,i]])
                ECE= compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = True)
                ACE = compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = False)
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


####################################################
# MLP head
# ####################################################
class MLP_head(pl.LightningModule):
    def __init__(self, config):
        super(MLP_head, self).__init__()

        self.train_step_pos_loss = []
        self.train_step_neg_loss = []
        self.val_step_pos_loss = []
        self.val_step_neg_loss = []

        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_of_tasks = config['num_of_tasks']
        self.dropout_p = config['dropout_p']
        self.loss_type = config['loss_type']
        self.optim = config['optim']
        self.lr = config['lr']
        self.lr_schedulers = config["lr_schedulers"]
        self.epochs = config["epochs"]
        self.compute_metric_after_n_epochs = config["compute_metric_after_n_epochs"]

        self.optm_l2_lambda = config['optm_l2_lambda']
        self.l2_lambda = config['optm_l2_lambda']
        self.batch_size = config['batch_size']
        self.missing = config["missing"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]

        self.thresholds = np.linspace(0,1,20)

        # pos weights
        pos_weights = pd.read_csv(config["pos_weights"])
        if config["num_of_tasks"] == 1:
            pos_weights = pos_weights.set_index("Targets").reindex([config["selected_tasks"]]).weights.values
        else:
            pos_weights = pos_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
        
        pos_weights = (config["alpha"] * pos_weights) + (1 - config["alpha"])*1
        self.pos_weights = torch.tensor(pos_weights, device = config["device"])

        # class weights
        if config['num_of_tasks'] > 1:
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

        # Model architecture
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.num_of_tasks),
        )
    def forward(self, x_input):
        return self.head(x_input)
    
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
            {'params': decay, 'weight_decay': self.optm_l2_lambda}]

    def configure_optimizers(self):
        optimizer_grouped_parameters = self.add_weight_decay(skip_list=())
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_grouped_parameters, 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, 
                                             lr=self.lr)
        if self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                             lr=self.lr)
        
        if self.lr_schedulers == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        T_max = 10, 
                                                                        eta_min=1e-6) 
            return {"optimizer": self.optimizer, 
                    "lr_scheduler": self.scheduler}
        
        if self.lr_schedulers == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        verbose=True,
                                                                        patience=15,
                                                                        min_lr=1e-6,
                                                                        mode = 'min')
            return {
            'optimizer':  self.optimizer,
            'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
            'monitor': 'val_BCE_loss'
            }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        
        # Apply only on weights, exclude bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _compute_loss(self, y, y_hat):
        if self.num_of_tasks == 1:
            y = y.unsqueeze(1)
        # compute losses, wiht masking
        if self.missing == 'nan':
            y = torch.nan_to_num(y, nan = -1)
        
        # masks
        valid_label_mask = (y != -1).float()
        pos_label_mask = (y == 1)
        negative_label_mask = (y == 0)

        if self.loss_type == "BCE":
            weighted_loss = self.weighted_creterien(y_hat, y) * valid_label_mask
        if self.loss_type == "Focal_loss":
            weighted_loss = self.FL(y_hat, y)* valid_label_mask
        Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask

        if self.loss_type == 'Focal_loss_v2':
            Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask
            weighted_loss = sigmoid_focal_loss(inputs = y_hat,
                                                targets = y,
                                                BCE_loss = Non_weighted_loss,
                                                alpha = self.alpha,
                                                gamma = self.gamma)
            weighted_loss = weighted_loss * valid_label_mask
        
        # Non_weighted_loss, positive negative loss
       
        pos_loss = Non_weighted_loss * pos_label_mask
        neg_loss = Non_weighted_loss * negative_label_mask
        pos_loss = pos_loss.sum() / pos_label_mask.sum()
        neg_loss = neg_loss.sum() / negative_label_mask.sum()
    

        # compute mean loss
        Non_weighted_loss = Non_weighted_loss.sum() / valid_label_mask.sum()
        weighted_loss = weighted_loss.sum() / valid_label_mask.sum()

        weight_norm = self.l2_regularization()
        l2_reg_loss = self.l2_lambda*weight_norm
        total_loss = weighted_loss + l2_reg_loss

        return total_loss, weighted_loss, Non_weighted_loss,weight_norm, pos_loss, neg_loss

    def training_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        
        y_hat = self(x)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,weight_norm, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.train_step_pos_loss.append(pos_loss.item())
        self.train_step_neg_loss.append(neg_loss.item())

        self.log('train_BCE_weighted', weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_BCE_non_weighted', Non_weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_weight_norm', weight_norm, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,weight_norm, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.val_step_pos_loss.append(pos_loss.item())
        self.val_step_neg_loss.append(neg_loss.item())

        self.log('val_BCE_weighted', weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_BCE_non_weighted', Non_weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_weight_norm', weight_norm, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):

        pos_loss = torch.tensor(self.train_step_pos_loss)
        neg_loss = torch.tensor(self.train_step_neg_loss)
        geometric_mean = torch.sqrt(pos_loss.nanmean() * neg_loss.nanmean())

        self.log('train_BCE_pos', pos_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('train_BCE_neg', neg_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('train_gm_loss', geometric_mean,on_step=False, on_epoch=True)
    
        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr,on_step=False, on_epoch=True)
        
        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.train_dataloader)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
            for i, score in enumerate(score_list):
                self.log(f'train_{metric[i]}', score, prog_bar=True , on_epoch=True)

        self.train_step_pos_loss.clear()
        self.train_step_neg_loss.clear()


    def on_validation_epoch_end(self):

        pos_loss = torch.tensor(self.val_step_pos_loss)
        neg_loss = torch.tensor(self.val_step_neg_loss)
        geometric_mean = torch.sqrt(pos_loss.nanmean() * neg_loss.nanmean())
        
        self.log('val_BCE_pos', pos_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('val_BCE_neg', neg_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('val_gm_loss', geometric_mean,on_step=False, on_epoch=True)


        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.val_dataloaders)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
            for i, score in enumerate(score_list):
                self.log(f'val_{metric[i]}', score, prog_bar=True , on_epoch=True)

        self.val_step_pos_loss.clear()
        self.val_step_neg_loss.clear()
           
    def compute_metrics(self, dataloader): 
        device = torch.device("cuda") 
        self.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:

                batch_x,batch_targets = batch
                batch_preds = self(batch_x.to(device))

                preds.extend(batch_preds.cpu().detach().tolist())
                targets.extend(batch_targets.cpu().detach().tolist())

            targets = np.array(targets).reshape(-1,self.num_of_tasks)
            preds = np.array(preds).reshape(-1,self.num_of_tasks)

            if self.missing == 'nan':
               mask = ~np.isnan(targets)

            roc_score, blc_acc, sensitivity, specificity, AUPR, f1, average_precision = [],[],[],[],[],[],[]
            ECE_score, ACE_score = [],[]

            n_bins = 10

            for i in range(self.num_of_tasks):
                
                # get valid targets, and convert logits to prob
                valid_targets = targets[:,i][mask[:,i]]
                valid_preds = expit(preds[:,i][mask[:,i]])
                ECE= compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = True)
                ACE = compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = False)
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

####################################################
# Chemprop Model
# ####################################################
class Custom_Chemprop(pl.LightningModule):
    def __init__(self, args):
        super(Custom_Chemprop, self).__init__()

        self.scheduler_type = args.scheduler_type
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.init_lr = args.init_lr
        self.max_lr = args.max_lr
        self.final_lr = args.final_lr
        self.target_weights = args.target_weights
        self.num_tasks = args.num_tasks

        self.is_atom_bond_targets = args.is_atom_bond_targets
        self.loss_function = args.loss_function
        self.missing_label = args.missing_label_representation
        self.compute_metrics_during_training  = args.compute_metrics_during_training

        # Should we use target weights ?
        if args.use_target_weights:
            if args.data_set == "SIDER":
                complete_data = pd.read_csv(args.data_path + 'SIDER_complete.csv')
                complete_data = complete_data.loc[:,"Hepatobiliary disorders":"Injury, poisoning and procedural complications"]
            else:
                raise ValueError('Provided data_set')
            target_weights = (complete_data == 0).sum() / (complete_data == 1).sum()
            args.target_weights = target_weights.values
            # normalize target weights (Coming from Chemprop)
            avg_weight = sum(args.target_weights)/len(args.target_weights)
            self.target_weights = [w/avg_weight for w in args.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError('Provided target weights must be non-negative.')
            self.target_weights = torch.tensor(self.target_weights, device=args.device).unsqueeze(0)  # shape(1,tasks)
        else:
            self.target_weights = torch.ones(args.num_tasks, device=args.device).unsqueeze(0)

        self.loss_fn =  nn.BCEWithLogitsLoss(reduction="none") 
        self.encoder = MPN(args)  # Adjust parameters accordingly
        self.readout = self.readout = build_ffn(
                                        first_linear_dim = args.hidden_size,
                                        hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                                        num_layers=args.ffn_num_layers,
                                        output_size=args.num_tasks,
                                        dropout=args.dropout,
                                        activation=args.activation                                    )
        initialize_weights(self)
        

    def forward(self, 
                smiles):
        output = self.encoder(smiles)
        output = self.readout(output)
        return output

    def _shared_step(self, batch, batch_idx):
        device = torch.device("cuda")

        # get batch
        smiles, targets = batch
        smiles = [[SMILES] for SMILES in smiles]
        preds = self(smiles)

        # compute losses, wiht masking
        if self.missing_label == -1:
            targets = torch.nan_to_num(targets, nan = -1)
        mask = (targets != self.missing_label).float()
        BCE_loss = self.loss_fn(preds, targets) * mask * self.target_weights
        regularization_loss = torch.tensor([0.0], device = device)
        return BCE_loss, regularization_loss
    
    def training_step(self, batch, batch_idx):
        BCE_loss, regularization_loss = self._shared_step(batch,batch_idx)
        total_loss = BCE_loss.mean() + regularization_loss
        self.log('train_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('train_reg_loss', regularization_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('current_lr', current_lr, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        return total_loss
    
    def validation_step(self, batch, batch_idx):

        BCE_loss, regularization_loss = self._shared_step(batch, batch_idx)
        total_loss = BCE_loss.mean() + regularization_loss
        self.log('val_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('val_reg_loss', regularization_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)

    def on_train_epoch_end(self):
        if self.compute_metrics_during_training == True:
            roc_score,aupr_score,mcc_score =  self.compute_metrics(self.trainer.train_dataloader)
            self.log('train_ROCAUC', roc_score, prog_bar=True)
            self.log('train_AUPR', aupr_score, prog_bar=True)
            self.log('train_MCC', mcc_score, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.compute_metrics_during_training == True:
            roc_score,aupr_score,mcc_score =  self.compute_metrics(self.trainer.val_dataloaders)

            self.log('val_ROCAUC', roc_score, prog_bar=True)
            self.log('val_AUPR', aupr_score, prog_bar=True)
            self.log('val_MCC', mcc_score, prog_bar=True)

    def configure_optimizers(self):
        # build optimizer
        params = [{"params": self.parameters(), "lr": self.init_lr, "weight_decay": 0}]
        self.optimizer = torch.optim.Adam(params)
        # build LR
        self.scheduler = NoamLR(
                                optimizer=self.optimizer,
                                warmup_epochs=self.warmup_epochs,
                                total_epochs= self.epochs,
                                steps_per_epoch= self.steps_per_epoch,
                                init_lr=self.init_lr,
                                max_lr=self.max_lr,
                                final_lr=self.final_lr,
                            )
        
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  {"scheduler": self.scheduler,
                             "interval": "step",
                             "frequency": 1,
                             "name": "learning_rate"}
        }
    def compute_metrics(self, dataloader):   
        preds, targets = [], []
        for batch in dataloader:
            smiles, batch_targets = batch
            smiles = [[SMILES] for SMILES in smiles]

            batch_preds = self(smiles)
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)
            targets.extend(batch_targets.cpu().detach().tolist())

        num_tasks = self.num_tasks
        valid_preds = [[] for _ in range(num_tasks)]
        valid_targets = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):

            for j in range(len(preds)):
                if targets[j][i] != self.missing_label:  # Skip those without targets
                    valid_preds[i].append(preds[j][i])
                    valid_targets[i].append(targets[j][i])
                    
        roc_score, aupr_score, mcc_score = [], [],[]
        for i in range(num_tasks):
            roc_score.append(roc_auc_score(valid_targets[i], valid_preds[i]))
            precision, recall, _ = precision_recall_curve(valid_targets[i], valid_preds[i])
            aupr_score.append(auc(recall, precision))

            threshold = 0.5
            hard_preds = [1 if p > threshold else 0 for p in valid_preds[i]]
            mcc_score.append(matthews_corrcoef(valid_targets[i], hard_preds))

        return np.mean(roc_score),np.mean(aupr_score), np.mean(mcc_score)
####################################################
# MF with Chemprop
# ####################################################
class MatrixFactorizer(pl.LightningModule):
    def __init__(self, args):
        super(MatrixFactorizer, self).__init__()

        self.target_weights = args.target_weights
        self.num_tasks = args.num_tasks
        self.num_mols = args.num_mols
        self.val_num_mols = args.val_num_mols
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.init_lr = args.init_lr
        self.max_lr = args.max_lr
        self.final_lr = args.final_lr

        self.l1_lambda = args.l1_lambda
        self.l2_lambda = args.l2_lambda

        self.is_atom_bond_targets = args.is_atom_bond_targets
        self.loss_function = args.loss_function

        if args.use_target_weights:
            complete_data = pd.read_csv(args.data_path + 'clinical_pre_clinical_06102023.csv')
            complete_data = complete_data.loc[:,"Apoptosis":"hepatobiliary_disorders"]
            target_weights = (complete_data == 0).sum() / (complete_data == 1).sum()
            args.target_weights = target_weights.values

            # normalize target weights (Coming from Chemprop)
            avg_weight = sum(args.target_weights)/len(args.target_weights)
            self.target_weights = [w/avg_weight for w in args.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError('Provided target weights must be non-negative.')
            self.target_weights = torch.tensor(self.target_weights, device=args.device).unsqueeze(0)  # shape(1,tasks)
        else:
            self.target_weights = torch.ones(args.num_tasks, device=args.device).unsqueeze(0)

        self.encoder = MPN(args)  # Adjust parameters accordingly
        self.readout = self.readout = build_ffn(
                                        first_linear_dim = args.hidden_size,
                                        hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                                        num_layers=args.ffn_num_layers,
                                        output_size=args.embedding_size,
                                        dropout=args.dropout,
                                        activation=args.activation,
                                    )
        
        #self.mol_embeddings = nn.Embedding(args.num_mol, args.embedding_size)
        self.task_embedding = nn.Embedding(args.num_tasks, args.embedding_size)
        self.mol_bias = nn.Embedding(args.num_mols, 1)
        self.task_bias = nn.Embedding(args.num_tasks, 1)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        initialize_weights(self)
        
        # Initialize empty tensors to hold ratings and predictions during training and validation "steps"
        self.training_step_ytrue = torch.full((args.num_mols,args.num_tasks),-1.0, device=args.device)
        self.training_step_ypred = torch.full((args.num_mols,args.num_tasks),-1.0, device=args.device)
        self.val_step_ytrue = torch.full((args.val_num_mols,args.num_tasks),-1.0, device=args.device)
        self.val_step_ypred = torch.full((args.val_num_mols,args.num_tasks),-1.0, device=args.device)

    def forward(self, mol_indices, smiles, task_indices):
        
        task_embeddings = self.task_embedding(task_indices)
        mol_embeddings = self.encoder(smiles)
        mol_embeddings = self.readout(mol_embeddings)

        mol_bias = self.mol_bias(mol_indices) # batch*1
        task_bias = self.task_bias(task_indices) # batch * num_tasks * 1
        biases_sum = mol_bias.unsqueeze(1) + task_bias # batch * num_tasks * 1

        # mol_embeddings --> [batch_size, embedding_size]
        # task_embeddings --> [batch_size, num_tasks, embedding_size]
        # Compute dot product between mol_embeddings and task_embeddings using element-wise multiplication and summation
        # pred --> [batch_size, num_tasks]
        
        dot_product = torch.sum(mol_embeddings.unsqueeze(1) * task_embeddings, dim=2) # [batch_size, num_tasks]
        pred = dot_product + biases_sum.squeeze(2)   #[batch_size, num_tasks]
        return pred, [mol_embeddings, task_embeddings, mol_bias, task_bias]

    def _shared_step(self, batch, batch_idx, ytrue_tensor, ypred_tensor):
        
        # get batch
        mol_indices, smiles, task_indices, targets = batch
        smiles = [[SMILES] for SMILES in smiles]

        # get pred
        preds, embeddings = self(mol_indices, smiles, task_indices)
        mol_embeddings, task_embeddings, _, _ = embeddings

        # save pred
        ytrue_tensor[mol_indices,:] = targets.to(torch.float)
        ypred_tensor[mol_indices,:] = preds
        
        # compute losses, wiht masking
        targets = torch.nan_to_num(targets, nan = -1)
        mask = (targets != -1).float()
        BCE_loss = self.loss_fn(preds, targets) * mask * self.target_weights
        
        l1_regularization_loss = self.l1_lambda * (torch.norm(self.task_embedding.weight, p = 1))
        #l2_regularization_loss = self.l2_lambda * (torch.norm(self.mol_embedding, p = 2))
        #regularization_loss = (l1_regularization_loss + l2_regularization_loss)
        regularization_loss = l1_regularization_loss
        
        return BCE_loss, regularization_loss
    
    def training_step(self, batch, batch_idx):
        BCE_loss, regularization_loss = self._shared_step(batch,batch_idx,
                                                         self.training_step_ytrue, 
                                                         self.training_step_ypred)
        total_loss = BCE_loss.mean() + regularization_loss
        self.log('train_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('train_reg_loss', regularization_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('current_lr', current_lr, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)


        return total_loss
    
    def validation_step(self, batch, batch_idx):

        BCE_loss, regularization_loss = self._shared_step(batch, batch_idx,
                                                        self.val_step_ytrue,
                                                        self.val_step_ypred)
        total_loss = BCE_loss.mean() + regularization_loss
        self.log('val_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('val_reg_loss', regularization_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)

    
    def on_train_epoch_end(self):
        device = torch.device("cuda")
        roc_score,aupr_score,mcc_score =  self.compute_metrics(self.training_step_ytrue, 
                                                               self.training_step_ypred)
        self.log('train_ROCAUC', roc_score, prog_bar=True)
        self.log('train_AUPR', aupr_score, prog_bar=True)
        self.log('train_MCC', mcc_score, prog_bar=True)

        # emptying the tensors
        self.training_step_ytrue = torch.full((self.num_mols,self.num_tasks),-1.0, device=device)
        self.training_step_ypred = torch.full((self.num_mols,self.num_tasks),-1.0, device=device)

    def on_validation_epoch_end(self):
        device = torch.device("cuda")
        roc_score,aupr_score,mcc_score =  self.compute_metrics(self.val_step_ytrue,
                                                               self.val_step_ypred)

        self.log('val_ROCAUC', roc_score, prog_bar=True)
        self.log('val_AUPR', aupr_score, prog_bar=True)
        self.log('val_MCC', mcc_score, prog_bar=True)

        # emptying the tensors
        self.val_step_ytrue = torch.full((self.val_num_mols,self.num_tasks),-1.0, device=device)
        self.val_step_ypred = torch.full((self.val_num_mols,self.num_tasks),-1.0, device=device)

    def configure_optimizers(self):
        # build optimizer
        params = [{"params": self.parameters(), "lr": self.init_lr, "weight_decay": 0}]
        self.optimizer = torch.optim.Adam(params)
        # build LR
        self.scheduler = NoamLR(
                                optimizer=self.optimizer,
                                warmup_epochs=self.warmup_epochs,
                                total_epochs= self.epochs,
                                steps_per_epoch= self.steps_per_epoch,
                                init_lr=self.init_lr,
                                max_lr=self.max_lr,
                                final_lr=self.final_lr,
                            )
        
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  {"scheduler": self.scheduler,
                             "interval": "step",
                             "frequency": 1,
                             "name": "learning_rate"}
        }
    def compute_metrics(self, ytrue_tensor, ypred_tensor):   
        
        targets = torch.nan_to_num(ytrue_tensor, nan = -1)
        targets = targets.cpu().detach().tolist()
        preds = ypred_tensor.cpu().detach().tolist()
        num_tasks = self.num_tasks
        valid_preds = [[] for _ in range(num_tasks)]
        valid_targets = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):

            for j in range(len(preds)):
                if targets[j][i] != -1:  # Skip those without targets
                    valid_preds[i].append(preds[j][i])
                    valid_targets[i].append(targets[j][i])
                    
        roc_score, aupr_score, mcc_score = [], [],[]
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

        return np.mean(roc_score),np.mean(aupr_score), np.mean(mcc_score)
    
#######################################################3
# Matrix Factorization model
########################################################

class DimensionReductionMLP(nn.Module):
    def __init__(self, MLP_hidden_layer_dim, num_drug_attributes, latent_dim):
        super(DimensionReductionMLP, self).__init__()
        self.fc1 = nn.Linear(num_drug_attributes, MLP_hidden_layer_dim)
        self.fc2 = nn.Linear(MLP_hidden_layer_dim , latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class MatrixFactorizationModel(pl.LightningModule):
    def __init__(self, config):
        super(MatrixFactorizationModel, self).__init__()

        self.num_drugs = config["num_drugs"]
        self.num_side_effects = config["num_side_effects"]
        self.latent_dim = config["latent_dim"]
        self.MLP_hidden_layer_dim = config["MLP_hidden_layer_dim"]
        self.lambda_reg = config['l2_lambda']
        self.optim = config['optim']
        self.lr = config['lr']
        self.assistive_loss = config['assistive_loss']
        self.use_class_weight_with_BCE = config['use_class_weight_with_BCE']
        self.batch_size = config['batch_size']

        # Initialize empty tensors to hold ratings and predictions during training and validation "steps"
        self.training_step_ytrue = torch.zeros((self.num_drugs, self.num_side_effects), device=torch.device('cuda'))
        self.training_step_ypred = torch.zeros((self.num_drugs, self.num_side_effects), device=torch.device('cuda'))
        self.val_step_ytrue = torch.zeros((self.num_drugs, self.num_side_effects), device=torch.device('cuda'))
        self.val_step_ypred = torch.zeros((self.num_drugs, self.num_side_effects), device=torch.device('cuda'))
    
        # Create the dimensionality reduction MLP
        self.dim_reduction_mlp = DimensionReductionMLP(self.MLP_hidden_layer_dim, config['num_drug_attributes'], self.latent_dim)
        
        self.drug_embeddings = nn.Embedding(self.num_drugs, self.latent_dim)
        self.side_effect_embeddings = nn.Embedding(self.num_side_effects, self.latent_dim)
        
        self.drug_bias = nn.Embedding(self.num_drugs, 1)
        self.side_effect_bias = nn.Embedding(self.num_side_effects, 1)

    def forward(self, drug_indices, side_effect_indices, drug_attributes):
        # Perform dimensionality reduction using MLP
        drug_latent_factors = self.dim_reduction_mlp(drug_attributes)

        side_effect_latent_factors = self.side_effect_embeddings(side_effect_indices)
        drug_biases = self.drug_bias(drug_indices)
        side_effect_biases = self.side_effect_bias(side_effect_indices)

        estimated_ratings = drug_biases + side_effect_biases
        estimated_ratings += (drug_latent_factors * side_effect_latent_factors).sum(1, keepdim=True)
        return torch.sigmoid(estimated_ratings.squeeze())
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=True,
                                                                    patience=5,
                                                                    min_lr=1e-6,
                                                                    mode='max')
        return {
            'optimizer':  self.optimizer,
            'lr_scheduler':  self.scheduler,
            'monitor': 'val_AUPR'
        }
    
    def compute_loss(self, drug_indices, side_effect_indices, ratings, estimated_ratings):
            
        bce = nn.BCELoss(reduction="sum")
        loss = bce(estimated_ratings, ratings)

        # L2 regularization terms for latent factors of drugs and side effects
        drug_latent_factors = self.drug_embeddings(drug_indices)
        side_effect_latent_factors = self.side_effect_embeddings(side_effect_indices)
        regularization_loss = self.lambda_reg * (torch.norm(drug_latent_factors) ** 2
                                                + torch.norm(side_effect_latent_factors) ** 2)
        loss += regularization_loss
        
        return loss
    
    def training_step(self, batch, batch_idx):
        drug_indices, side_effect_indices, ratings, drug_attributes = batch
        estimated_ratings = self.forward(drug_indices, side_effect_indices, drug_attributes)

        loss = self.compute_loss(drug_indices, side_effect_indices, ratings, estimated_ratings)
        self.log('train_loss', loss,  prog_bar=True, on_step=False, on_epoch=True)

      # Fill the training tensors at the appropriate indices
        self.training_step_ytrue[drug_indices.long(), side_effect_indices.long()] = ratings
        self.training_step_ypred[drug_indices.long(), side_effect_indices.long()] = estimated_ratings
        return loss

    def validation_step(self, batch, batch_idx):
        drug_indices, side_effect_indices, ratings, drug_attributes = batch
        estimated_ratings = self.forward(drug_indices, side_effect_indices, drug_attributes)
        
        loss = self.compute_loss(drug_indices, side_effect_indices, ratings, estimated_ratings)
        self.log('val_loss', loss,  prog_bar=True, on_step=False, on_epoch=True)

        # Fill the validation tensors at the appropriate indices
        self.val_step_ytrue[drug_indices.long(), side_effect_indices.long()] = ratings
        self.val_step_ypred[drug_indices.long(), side_effect_indices.long()] = estimated_ratings

    def on_train_epoch_end(self):

        metric = MultilabelAveragePrecision(num_labels=self.num_side_effects, average="macro", thresholds=None)
        AUPR = metric(self.training_step_ypred, self.training_step_ytrue.long())
        self.log('train_AUPR', AUPR, prog_bar=True)

        # Clear to free memory for the next epoch
        self.training_step_ytrue.zero_()
        self.training_step_ypred.zero_()

    def on_validation_epoch_end(self):
        
        metric = MultilabelAveragePrecision(num_labels=self.num_side_effects, average="macro", thresholds=None)
        AUPR = metric(self.val_step_ypred, self.val_step_ytrue.long())
        self.log('val_AUPR', AUPR, prog_bar=True)

        # Clear the to free memory for the next epoch
        self.val_step_ytrue.zero_()
        self.val_step_ypred.zero_()
        
#########################################################
# MLP with skip connection
#########################################################
class ResidualBlock(nn.Module):
    def __init__(self,input_dim, hidden_dim, BatchNorm1d, dropout_p):
        super(ResidualBlock, self).__init__()
        self.use_batch_norm = BatchNorm1d
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

        if self.use_batch_norm:
            self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
            self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        residual = x
        x = self.layer1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = x + residual  # Add skip connection
        x = torch.relu(x)  # Apply activation after the sum
        x = self.dropout(x)
        return x
    
class MLP_with_skip_connection(pl.LightningModule):
    def __init__(self, config):
        super(MLP_with_skip_connection, self).__init__()

        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]

        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.tasks = config['num_of_tasks']
        self.BatchNorm1d = config['BatchNorm1d']
        self.dropout_p = config['dropout_p']
        self.num_residual_blocks = config['num_residual_blocks']

        self.optim = config['optim']
        self.lr = config['lr']

        self.l2_lambda = config['l2_lambda']
        self.use_class_weight_with_BCE = config['use_class_weight_with_BCE']
        self.batch_size = config['batch_size']

        # To handle imblance within class, # Binary loss
        if self.use_class_weight_with_BCE == True:
            self.pos_weight_list = class_weights_for_SIDER(config) 
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_list, reduction="none")

        if self.use_class_weight_with_BCE == False:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none") # Binary loss

        # Model architecture
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.residual_blocks = nn.ModuleList([ResidualBlock(self.hidden_dim, self.hidden_dim, self.BatchNorm1d, self.dropout_p) for _ in range(self.num_residual_blocks)])
        self.output_layer = nn.Linear(self.hidden_dim, self.tasks)
        self.dropout = nn.Dropout(self.dropout_p)

        if self.BatchNorm1d:
            self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        if self.BatchNorm1d:
            x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)  # Apply each residual block
        x = self.output_layer(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              #weight_decay = self.l2_lambda,
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=True,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_AUPR'
        }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _compute_loss(self, y, y_hat):
        # mask missing values
        mask = (y != -1).float()
        
        BCE_loss = self.loss_fn(y_hat, y.float())
        BCE_loss = torch.mul(BCE_loss, mask)
        BCE_loss = torch.sum(BCE_loss) / torch.sum(mask.float())

        l2_reg_loss = self.l2_regularization()
        return BCE_loss, l2_reg_loss

    def training_step(self, batch, batch_idx):
        
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, l2_reg_loss = self._compute_loss(y, y_hat)  
        #BCE_loss = BCE_loss.mean()
        l2_reg_loss = self.l2_lambda*l2_reg_loss
        total_loss = BCE_loss + l2_reg_loss

        self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Append true labels and predictions to lists
        self.training_step_ytrue.append(y.long().cpu())
        self.training_step_ypred.append(torch.sigmoid(y_hat).cpu())
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, l2_reg_loss = self._compute_loss(y, y_hat)  
        #BCE_loss = BCE_loss.mean()
        l2_reg_loss = self.l2_lambda*l2_reg_loss
        total_loss = BCE_loss + l2_reg_loss
            
        self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Append true labels and predictions to lists
        self.val_step_ytrue.append(y.long().cpu())
        self.val_step_ypred.append(torch.sigmoid(y_hat).cpu())

    
    def on_train_epoch_end(self):

        # Collect predictions and true labels for the complete training set
        train_true = torch.cat(self.training_step_ytrue, dim=0)
        train_preds = torch.cat(self.training_step_ypred, dim=0)

        if (train_true == -1).any().item():
            AUPR = compute_AUPR_by_ignoring_missing_values(train_true, train_preds)
        else:
            metric = MultilabelAveragePrecision(num_labels=self.tasks, average="macro", thresholds=None)
            AUPR = metric(train_preds, train_true)
            
        self.log('train_AUPR', AUPR, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.training_step_ytrue.clear()
        self.training_step_ypred.clear()
        del train_true,train_preds

    def on_validation_epoch_end(self):

        #Collect predictions and true labels for the complete training set
        val_true = torch.cat(self.val_step_ytrue, dim=0)
        val_preds = torch.cat(self.val_step_ypred, dim=0)

        if (val_true == -1).any().item():
            AUPR = compute_AUPR_by_ignoring_missing_values(val_true, val_preds)
        else:
            metric = MultilabelAveragePrecision(num_labels=self.tasks, average="macro", thresholds=None)
            AUPR = metric(val_preds, val_true)
            
        self.log('val_AUPR', AUPR, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.val_step_ytrue.clear()
        self.val_step_ypred.clear()
        del val_true, val_preds

#################################################
# MLP with emperical prior
################################################
class MLP_SK_weighted(pl.LightningModule):
    def __init__(self, config):
        super(MLP_SK_weighted, self).__init__()

        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]
        self.w_training, self.w_validation = [],[]
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.tasks = config['num_of_tasks']
        self.BatchNorm1d = config['BatchNorm1d']
        self.dropout_p = config['dropout_p']
        self.num_residual_blocks = config['num_residual_blocks']

        self.optim = config['optim']
        self.lr = config['lr']

        self.l2_lambda = config['l2_lambda']
        self.use_class_weight_with_BCE = config['use_class_weight_with_BCE']
        self.batch_size = config['batch_size']

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        # Model architecture
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.residual_blocks = nn.ModuleList([ResidualBlock(self.hidden_dim, self.hidden_dim, self.BatchNorm1d, self.dropout_p) for _ in range(self.num_residual_blocks)])
        self.output_layer = nn.Linear(self.hidden_dim, self.tasks)
        self.dropout = nn.Dropout(self.dropout_p)

        if self.BatchNorm1d:
            self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        if self.BatchNorm1d:
            x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)  # Apply each residual block
        x = self.output_layer(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              #weight_decay = self.l2_lambda,
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=True,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_AUPR'
        }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _compute_loss(self, y, y_hat):
        
        BCE_loss = self.loss_fn(y_hat, y.float())
        l2_reg_loss = self.l2_regularization()
        return BCE_loss, l2_reg_loss

    def training_step(self, batch, batch_idx):
        
        # compute forward pass
        x, y, w = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, l2_reg_loss = self._compute_loss(y, y_hat)  
        
        if self.use_class_weight_with_BCE:
            BCE_loss = (BCE_loss * w).mean()
        else:
            BCE_loss = BCE_loss.mean()

        l2_reg_loss = self.l2_lambda*l2_reg_loss
        total_loss = BCE_loss + l2_reg_loss

        self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Append true labels and predictions to lists
        self.training_step_ytrue.append(y.long().cpu())
        self.training_step_ypred.append(torch.sigmoid(y_hat).cpu())
        self.w_training.append(w.cpu())
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y, w = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, l2_reg_loss = self._compute_loss(y, y_hat)  

        if self.use_class_weight_with_BCE:
            BCE_loss = (BCE_loss * w).mean()
        else:
            BCE_loss = BCE_loss.mean()

        l2_reg_loss = self.l2_lambda*l2_reg_loss
        total_loss = BCE_loss + l2_reg_loss
            
        self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Append true labels and predictions to lists
        self.val_step_ytrue.append(y.long().cpu())
        self.val_step_ypred.append(torch.sigmoid(y_hat).cpu())
        self.w_validation.append(w.cpu())
    
    def on_train_epoch_end(self):

        # Collect predictions and true labels for the complete training set
        train_true = torch.cat(self.training_step_ytrue, dim=0)
        train_preds = torch.cat(self.training_step_ypred, dim=0)
        train_w = torch.cat(self.w_training, dim=0)
    
        AUPR, AUROC = compute_weighted_metrics(train_true, train_preds, train_w)
        self.log('train_AUPR', AUPR, prog_bar=True)
        self.log('train_AUROC', AUROC, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.training_step_ytrue.clear()
        self.training_step_ypred.clear()
        self.w_training.clear()

        del train_true, train_preds, train_w

    def on_validation_epoch_end(self):

        #Collect predictions and true labels for the complete training set
        val_true = torch.cat(self.val_step_ytrue, dim=0)
        val_preds = torch.cat(self.val_step_ypred, dim=0)
        val_w = torch.cat(self.w_validation, dim=0)

        AUPR, AUROC = compute_weighted_metrics(val_true, val_preds, val_w)
        self.log('val_AUPR', AUPR, prog_bar=True)
        self.log('val_AUROC', AUROC, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.val_step_ytrue.clear()
        self.val_step_ypred.clear()
        self.w_validation.clear()

        del val_true,val_preds,val_w
#################################################
# MLP with emperical prior
################################################
class MLP_with_SK_Emperical_Prior(pl.LightningModule):
    def __init__(self, config):
        super(MLP_with_SK_Emperical_Prior, self).__init__()

        device = torch.device('cuda')
        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.tasks = config['num_of_tasks']
        self.BatchNorm1d = config['BatchNorm1d']
        self.dropout_p = config['dropout_p']
        self.num_residual_blocks = config['num_residual_blocks']

        self.optim = config['optim']
        self.lr = config['lr']

        self.l1_lambda = config['l1_lambda']
        self.l2_lambda = config['l2_lambda']
        self.use_class_weight_with_BCE = config['use_class_weight_with_BCE']
        self.batch_size = config['batch_size']

        self.cov_dist = config["cov_dist"]

        # To handle imblance within class, # Binary loss
        if self.use_class_weight_with_BCE == True:
            self.pos_weight_list = class_weights_for_SIDER(config) 
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_list, reduction="none")

        if self.use_class_weight_with_BCE == False:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none") # Binary loss

        # Model architecture
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.residual_blocks = nn.ModuleList([ResidualBlock(self.hidden_dim, self.hidden_dim, self.BatchNorm1d, self.dropout_p) for _ in range(self.num_residual_blocks)])
        self.output_layer = nn.Linear(self.hidden_dim, self.tasks)
        self.dropout = nn.Dropout(self.dropout_p)

        if self.BatchNorm1d:
            self.batchnorm3 = nn.BatchNorm1d(self.hidden_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        if self.BatchNorm1d:
            x = self.batchnorm3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)  # Apply each residual block
        x = self.output_layer(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              #weight_decay = self.l2_lambda,
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=True,
                                                                    patience=30,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_AUPR'
        }

    def regularization_losses(self):
        device = torch.device('cuda')
        emp_regul_loss = torch.tensor(0., requires_grad=True, device=device)
        norm_regul_loss = torch.tensor(0., requires_grad=True, device=device)

        for name, param in self.named_parameters():
            
            if name == 'input_layer.weight':
                emp_regul_loss =  emp_regul_loss + self.cov_dist.log_prob(param).mean()
           
            else:
                if 'weight' in name and not ('batchnorm' in name or 'bias' in name): 
                    #weight_size = param.shape[1]
                    #distribution = MultivariateNormal(torch.zeros(weight_size, device=device), torch.eye(weight_size, device=device))
                    #norm_regul_loss = norm_regul_loss + distribution.log_prob(param).mean()
                    norm_regul_loss = norm_regul_loss + torch.norm(param, p=2)

        return -emp_regul_loss,norm_regul_loss
    
    def _compute_loss(self, y, y_hat):

        # mask missing values
        mask = (y != -1).float()
        
        BCE_loss = self.loss_fn(y_hat, y.float())
        BCE_loss = torch.mul(BCE_loss, mask)
        BCE_loss = torch.sum(BCE_loss) / torch.sum(mask.float())
        
        emp_prior, norm_prior = self.regularization_losses()
        return BCE_loss, emp_prior, norm_prior

    def training_step(self, batch, batch_idx):
        
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, emp_prior, norm_prior = self._compute_loss(y, y_hat)  

        emp_prior = self.l1_lambda*emp_prior
        norm_prior = self.l2_lambda*norm_prior
        total_loss = BCE_loss.mean() + emp_prior + norm_prior

        self.log('train_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_emp_prior_loss', emp_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_norm_prior_loss', norm_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Append true labels and predictions to lists
        self.training_step_ytrue.append(y.long())
        self.training_step_ypred.append(torch.sigmoid(y_hat))
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, emp_prior, norm_prior = self._compute_loss(y, y_hat)  

        emp_prior = self.l1_lambda*emp_prior
        norm_prior = self.l2_lambda*norm_prior
        total_loss = BCE_loss.mean() + emp_prior + norm_prior

        self.log('val_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_emp_prior_loss', emp_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_norm_prior_loss', norm_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Append true labels and predictions to lists
        self.val_step_ytrue.append(y.long())
        self.val_step_ypred.append(torch.sigmoid(y_hat))
    
    def on_train_epoch_end(self):

        # Collect predictions and true labels for the complete training set
        train_true = torch.cat(self.training_step_ytrue, dim=0)
        train_preds = torch.cat(self.training_step_ypred, dim=0)

        if (train_true == -1).any().item():
            AUPR = compute_AUPR_by_ignoring_missing_values(train_true, train_preds)
        else:
            metric = MultilabelAveragePrecision(num_labels=self.tasks, average="macro", thresholds=None)
            AUPR = metric(train_preds, train_true)
            
        self.log('train_AUPR', AUPR, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.training_step_ytrue.clear()
        self.training_step_ypred.clear()
        del train_true, train_preds

    def on_validation_epoch_end(self):

        #Collect predictions and true labels for the complete training set
        val_true = torch.cat(self.val_step_ytrue, dim=0)
        val_preds = torch.cat(self.val_step_ypred, dim=0)

        if (val_true == -1).any().item():
            AUPR = compute_AUPR_by_ignoring_missing_values(val_true, val_preds)
        else:
            metric = MultilabelAveragePrecision(num_labels=self.tasks, average="macro", thresholds=None)
            AUPR = metric(val_preds, val_true)
            
        self.log('val_AUPR', AUPR, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.val_step_ytrue.clear()
        self.val_step_ypred.clear()
        del val_true, val_preds
#################################################
# Binary-Classification Feed-Forward invitro-preclinical-clinical
################################################
class MT_invitro_preclinical_clinical(pl.LightningModule):
    def __init__(self, config):
        super(MT_invitro_preclinical_clinical, self).__init__()

        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.BatchNorm1d = config['BatchNorm1d']
        self.optim = config['optim']
        self.lr = config['lr']

        self.l2_reg_loss = config['l2_reg_loss']
        self.l1_reg_loss = config['l1_reg_loss']
        self.l2_lambda = config['l2_lambda']
        self.l1_lambda = config['l1_lambda']

        self.assistive_loss = config['assistive_loss']
        self.use_class_weight_with_BCE = config['use_class_weight_with_BCE']
        self.batch_size = config['batch_size']
        self.tasks = config['num_of_tasks']

        # to compute metrices during training
        self.compute_metric_during_training = config["compute_metric_during_training"]
        selected_tasks = config["selected_tasks"]
        self.SIDER_binary_index = selected_tasks.index("SIDER_binary")
        self.PreClinical_binary_index = selected_tasks.index("PreClinical_binary")
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()

        if self.activation == 'LeakyReLU':  
            self.activ = torch.nn.LeakyReLU(0.05)
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        self.roc_auc = torchmetrics.AUROC(task="binary")
        self.AP_score = torchmetrics.AveragePrecision(task="binary")

        # List of losses
        if self.assistive_loss == 'modality_wise_balanced_loss':
            self.modality_wise_balanced_loss = modality_wise_balanced_loss(config)
        if self.assistive_loss == 'MultiTaskLoss':
            self.loss_type = torch.tensor([1.] * self.tasks) # To Balance between tasks
            self.multi_task_loss = MultiTaskLoss(self.loss_type)  

        if self.assistive_loss == 'FocalLoss':
            self.FocalLoss = FocalLoss()

        if self.assistive_loss == 'AsymmetricUnifiedLoss':   
            self.AsymmetricUnifiedLoss = AsymmetricLoss()

        if self.use_class_weight_with_BCE == True:
            self.pos_weight_list = class_weights_for_SIDER(config) # To handle imblance within class
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_list, reduction="none") # Binary loss

        if self.use_class_weight_with_BCE == False:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none") # Binary loss

        # hidden units, size depends upon number of layers
        if self.depth == 'super_shallow':

            layer_input_dim = 4096
            self.arch.append(nn.Sequential(nn.Linear(layer_input_dim, self.tasks)))
            
        elif self.depth == 'shallow':

            layer_input_dim = 4096
            layer_output_dim = 256
            layer_sequence = [nn.Linear(layer_input_dim, layer_output_dim)]

            if self.BatchNorm1d == True: 
                layer_sequence.append(nn.BatchNorm1d(layer_output_dim))

            layer_sequence.extend([self.dropout, self.activ])
            self.arch.append(nn.Sequential(*layer_sequence))

            self.arch.append(nn.Sequential(nn.Linear(layer_output_dim, self.tasks)))
            
        else:
            
            N_0 = 4096
            layer_input_dim = 4096
            for layer_number in range(self.depth):
                
                layer_output_dim = int(N_0*(0.5)**(layer_number + 1))
                layer_sequence = [nn.Linear(layer_input_dim, layer_output_dim)]

                if self.BatchNorm1d == True: 
                    layer_sequence.append(nn.BatchNorm1d(layer_output_dim))

                layer_sequence.extend([self.dropout, self.activ])
                self.arch.append(nn.Sequential(*layer_sequence))
                layer_input_dim = layer_output_dim
            
            self.arch.append(nn.Sequential(nn.Linear(layer_output_dim, self.tasks)))
        
        
    def forward(self, x):
        for i, layers in enumerate(self.arch):     
            x = layers(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              #weight_decay = self.l2_lambda,
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=5,
                                                                    min_lr=1e-6,
                                                                    mode = 'min')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_BCE_loss'
        }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def l1_regularization(self):
        device = torch.device('cuda')
        l1_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
        return l1_reg
    
    def _shared_step(self, batch, batch_idx,prefix):
        # compute forward pass
        x, y = batch
        y_hat = self(x)
        y_prob = self.sigmoid(y_hat)

        if self.compute_metric_during_training == True:
            col = [self.SIDER_binary_index, self.PreClinical_binary_index]
            metrics = compute_binary_classification_metrics_MT(y[:, col].detach().cpu().numpy(), y_prob[:, col].detach().cpu().numpy(), threshold=0.5)

        # Generate a mask to ignore examples with missing labels (-1)
        mask = (y != -1).float()

        BCE_loss = self.loss_fn(y_hat, y.float())
        BCE_loss = torch.mul(BCE_loss, mask)

        if self.assistive_loss == 'MultiTaskLoss':  
            BCE_loss = self.multi_task_loss(BCE_loss)

        if self.assistive_loss == 'FocalLoss':
            BCE_loss = self.FocalLoss(BCE_loss)

        if self.assistive_loss == 'AsymmetricUnifiedLoss':   
            BCE_loss = self.AsymmetricUnifiedLoss(y, BCE_loss)

        loss_pos = torch.mul(BCE_loss,y).mean()
        loss_neg = torch.mul(BCE_loss,(1-y)).mean()
        
        if self.assistive_loss == 'modality_wise_balanced_loss': 
            invitro_loss,preclinical_loss,clinical_loss = self.modality_wise_balanced_loss(BCE_loss, mask)

        
        if self.assistive_loss == None: 
                BCE_loss = BCE_loss.mean()
        
        if self.l2_reg_loss == True:
            l2_reg_loss = self.l2_regularization()
        if self.l2_reg_loss == False:
            l2_reg_loss = torch.tensor(0., requires_grad=True, device=torch.device('cuda'))
        if self.l1_reg_loss == True:
            l1_reg_loss = self.l1_regularization()
        if self.l1_reg_loss == False:
            l1_reg_loss = torch.tensor(0., requires_grad=True, device=torch.device('cuda'))
        
        if (self.assistive_loss == 'modality_wise_balanced_loss') & (self.compute_metric_during_training == True):
                return  invitro_loss,preclinical_loss,clinical_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg, metrics
        
        elif (self.assistive_loss == 'modality_wise_balanced_loss'):
                return  invitro_loss,preclinical_loss,clinical_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg
        else:
            return BCE_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg

    
    def training_step(self, batch, batch_idx):
        
        if self.assistive_loss == 'modality_wise_balanced_loss':
            invitro_loss,preclinical_loss,clinical_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg  = self._shared_step(batch, batch_idx, "train")
            BCE_loss = torch.tensor([invitro_loss,preclinical_loss,clinical_loss]).sum()
            self.log('train_invitro_loss', invitro_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_preclinical_loss', preclinical_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_clinical_loss', clinical_loss, prog_bar=True, on_step=False, on_epoch=True)

            print(invitro_loss.item(),preclinical_loss.item(),clinical_loss.item())
            if self.compute_metric_during_training == True:
                tasks_list = ['SIDER','PreClinical']
                metrics.insert(0, 'task', tasks_list)
                
                # log to wandb
                metrics_list = metrics.columns[1:].tolist()
                for task in tasks_list:
                    for metric in metrics_list:
                        value = metrics[metrics.task == task][metric].values.item()
                        name = 'Training_' + str(task) +'_' + str(metric)
                        self.log(name, value, prog_bar=True, on_step=False, on_epoch=True)
           
        if self.assistive_loss == None:
            BCE_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg  = self._shared_step(batch, batch_idx, "train")
            
        l2_reg_loss = self.l2_lambda*l2_reg_loss
        l1_reg_loss = self.l1_lambda*l1_reg_loss
        total_loss = BCE_loss + l2_reg_loss + l1_reg_loss

        self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss_pos', loss_pos, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss_neg', loss_neg, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l1_reg_loss', l1_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        if self.assistive_loss == 'modality_wise_balanced_loss':
            invitro_loss,preclinical_loss,clinical_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg  = self._shared_step(batch, batch_idx, "val")
            BCE_loss = torch.tensor([invitro_loss,preclinical_loss,clinical_loss]).sum()
            self.log('val_invitro_loss', invitro_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_preclinical_loss', preclinical_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_clinical_loss', clinical_loss, prog_bar=True, on_step=False, on_epoch=True)

            if self.compute_metric_during_training == True:
                tasks_list = ['SIDER','PreClinical']
                metrics.insert(0, 'task', tasks_list)
                
                # log to wandb
                metrics_list = metrics.columns[1:].tolist()
                for task in tasks_list:
                    for metric in metrics_list:
                        value = metrics[metrics.task == task][metric].values.item()
                        name = 'Val_' + str(task) +'_' + str(metric)
                        self.log(name, value, prog_bar=True, on_step=False, on_epoch=True)
            
        if self.assistive_loss == None:
            BCE_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg  = self._shared_step(batch, batch_idx, "val")
            
        l2_reg_loss = self.l2_lambda*l2_reg_loss
        l1_reg_loss = self.l1_lambda*l1_reg_loss
        total_loss = BCE_loss + l2_reg_loss + l1_reg_loss
        self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss_pos', loss_pos, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss_neg', loss_neg, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

#################################################
# Binary-Classification Feed-Forward Baseline
################################################

class FF_Binary_Classifier(pl.LightningModule):
    def __init__(self, config):
        super(FF_Binary_Classifier, self).__init__()

        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.depth = int(config['depth'])
        self.tasks = config['num_of_tasks']
        self.BatchNorm1d = config['BatchNorm1d']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']

        self.l2_lambda = config['l2_lambda']
        self.use_class_weight_with_BCE = config['use_class_weight_with_BCE']
        self.batch_size = config['batch_size']

        # To handle imblance within class, # Binary loss
        if self.use_class_weight_with_BCE == True:
            self.pos_weight_list = class_weights_for_SIDER(config) 
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_list, reduction="none")

        if self.use_class_weight_with_BCE == False:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none") # Binary loss

        if self.depth == 'Logistic_Regression':
            self.fc_layers = nn.ModuleList()
            self.fc_layers.append(nn.Sequential(nn.Linear(self.input_dim, self.tasks)))

        else:
            self.fc_layers = nn.ModuleList()
            self.fc_layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            if self.BatchNorm1d:
                self.fc_layers.append(nn.BatchNorm1d(self.hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=self.dropout_p))

            for i in range(self.depth):
                self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                if self.BatchNorm1d:
                    self.fc_layers.append(nn.BatchNorm1d(self.hidden_dim))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(p=self.dropout_p))

            self.fc_layers.append(nn.Linear(self.hidden_dim, self.tasks))
        
    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              #weight_decay = self.l2_lambda,
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=True,
                                                                    patience=5,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_AUPR'
        }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _compute_loss(self, y, y_hat):

        BCE_loss = self.loss_fn(y_hat, y.float())
        l2_reg_loss = self.l2_regularization()
        return BCE_loss, l2_reg_loss

    def training_step(self, batch, batch_idx):
        
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, l2_reg_loss = self._compute_loss(y, y_hat)  
        BCE_loss = BCE_loss.mean()
        l2_reg_loss = self.l2_lambda*l2_reg_loss
        total_loss = BCE_loss + l2_reg_loss

        self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Append true labels and predictions to lists
        self.training_step_ytrue.append(y.long())
        self.training_step_ypred.append(torch.sigmoid(y_hat))
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        BCE_loss, l2_reg_loss = self._compute_loss(y, y_hat)  
        BCE_loss = BCE_loss.mean()
        l2_reg_loss = self.l2_lambda*l2_reg_loss
        total_loss = BCE_loss + l2_reg_loss
            
        self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Append true labels and predictions to lists
        self.val_step_ytrue.append(y.long())
        self.val_step_ypred.append(torch.sigmoid(y_hat))
    
    def on_train_epoch_end(self):

        # Collect predictions and true labels for the complete training set
        train_true = torch.cat(self.training_step_ytrue, dim=0)
        train_preds = torch.cat(self.training_step_ypred, dim=0)

        metric = MultilabelAveragePrecision(num_labels=self.tasks, average="macro", thresholds=None)
        AUPR = metric(train_preds, train_true)
        self.log('train_AUPR', AUPR, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.training_step_ytrue.clear()
        self.training_step_ypred.clear()

    def on_validation_epoch_end(self):

        #Collect predictions and true labels for the complete training set
        val_true = torch.cat(self.val_step_ytrue, dim=0)
        val_preds = torch.cat(self.val_step_ypred, dim=0)

        metric = MultilabelAveragePrecision(num_labels=self.tasks, average="macro", thresholds=None)
        AUPR = metric(val_preds, val_true)
        self.log('val_AUPR', AUPR, prog_bar=True)

        # Clear the lists to free memory for the next epoch
        self.val_step_ytrue.clear()
        self.val_step_ypred.clear()

########################################################################################33
# FineTune Roberta
##########################################################################################
def finetune_roberta_model(train_df, valid_df, fold_index, NonFilter_valset, task):
    # Fit the model on the training set for the fold
    output_dir = f'/projects/home/mmasood1/Model_weights/finetune/{fold_index}/{task}/'
    model = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250', 
                                
                                args={'evaluate_each_epoch': True, 
                                      'evaluate_during_training_verbose': True, 
                                      'overwrite_output_dir': True, 
                                      'no_save': True, 
                                      'num_train_epochs': 5, 
                                      'auto_weights': True,
                                      'use_early_stopping': True,
                                      'early_stopping_delta': 0.01,
                                      'early_stopping_metric':'mcc',
                                      'early_stopping_metric_minimize': False,
                                      'early_stopping_patience': 3,
                                      })

    model.train_model(train_df, 
                    eval_df=valid_df, 
                    output_dir=output_dir,
                    args={'wandb_project': 'Roberta_FineTune'})
    
    # Make predictions on the validation set for the fold
    result, model_outputs, wrong_predictions = model.eval_model(valid_df, acc=sklearn.metrics.accuracy_score)
    result_fold = pd.DataFrame([result])
    result_fold['task'] = task

    # make predictions
    _, raw_outputs = model.predict(NonFilter_valset.iloc[:,0].tolist())
    y_pred_prob = softmax(raw_outputs, axis=1)[:,1]  
    return y_pred_prob, result_fold

#################################################
#Binary classifier for two modalities with all tasks
################################################
class Clinical_Pre_Clinical_Custom_Arch(pl.LightningModule):
    def __init__(self, config):
        super(Clinical_Pre_Clinical_Custom_Arch, self).__init__()

        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.l2_lambda = config['l2_lambda']
        self.batch_size = config['batch_size']
        self.pos_weight = config['pos_weight']
        self.tasks = config['num_of_tasks']
        self.compute_AUCs_during_training = config['compute_AUCs_during_training']
        self.alpha = config["alpha"]
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            #self.activ = torch.nn.LeakyReLU(0.05)
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        self.roc_auc = torchmetrics.AUROC(task="binary")
        self.AP_score = torchmetrics.AveragePrecision(task="binary")

        # List of losses
        #self.loss_type = torch.tensor([1.] * self.tasks) # To Balance between tasks
        #self.multi_task_loss = MultiTaskLoss(self.loss_type)  
        #self.FocalLoss = FocalLoss()
        self.Rescaled_loss = Rescaled_loss(alpha=self.alpha, gamma=1)
        self.pos_weight_list = class_weights_for_complete_data() # To handle imblance within class
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_list, reduction="none") # Binary loss
        #self.loss_fn = nn.BCELoss(weight=self.pos_weight_list, reduction="none") # Binary loss
        #self.loss_fn = AsymmetricLoss()
        #self.loss_fn = nn.BCELoss(reduction="none")
        # hidden units, size depends upon number of layers
     
        self.input_dim = 4096
        self.output_dim = self.tasks

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            self.activ,
            self.dropout   
        )
        self.block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            self.activ,
            self.dropout
            )
        
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            self.activ,
            self.dropout
            )

        self.output_layer = nn.Sequential(
            nn.Linear(128, self.output_dim),
        )
        
    def forward(self, x):
        x0 = self.input_layer(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x = x0 + x2  # Skip connection
        x = self.output_layer(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'min')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_BCE_loss'
        }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _shared_step(self, batch, batch_idx,prefix):
        # compute forward pass
        x, y = batch
        y_hat = self(x)
        y_prob = self.sigmoid(y_hat)

        # Generate a mask to ignore examples with missing labels (-1)
        mask = (y != -1).float()

        #BCE_loss = self.FocalLoss(BCE_loss)
        
        #BCE_loss = self.AsymmetricUnifiedLoss(y, BCE_loss)
        BCE_loss = self.loss_fn(y_hat, y.float())
        BCE_loss = torch.mul(BCE_loss, mask)
        BCE_loss = self.Rescaled_loss(y,BCE_loss)
        #BCE_loss = self.multi_task_loss(BCE_loss)
        #BCE_loss = self.FocalLoss(BCE_loss)
        BCE_loss = BCE_loss.mean()
        
        l2_reg_loss = self.l2_regularization()

        if self.compute_AUCs_during_training == True:
            # ROC_AUC and PR_AUC
            AUROC_Tasks, APScore_task = [], []

            for task in range(y.shape[1]):
                y_task_valid = y[:,task][mask[:,task]].round().to(torch.int64)
                pred_task_valid = y_prob[:,task][mask[:,task]]
                AUROC_Tasks.append(self.roc_auc(pred_task_valid, y_task_valid).item())
                APScore_task.append(self.AP_score(pred_task_valid, y_task_valid).item())

            AUROC_Tasks = np.array(AUROC_Tasks)
            APScore_task = np.array(APScore_task)

            # Modalilty wide calculations
            ROC_AUC_preclinical = np.around(AUROC_Tasks[:18].mean(),2)
            ROC_AUC_clinical = np.around(AUROC_Tasks[18:67].mean(),2)
            ROC_AUC_preclinical_global = np.around(AUROC_Tasks[67],2)
            ROC_AUC_clinical_global = np.around(AUROC_Tasks[68],2)

            APScore_preclinical = np.around(APScore_task[:18].mean(),2)
            APScore_clinical = np.around(APScore_task[18:67].mean(),2)
            APScore_preclinical_global = np.around(APScore_task[67],2)
            APScore_clinical_global = np.around(APScore_task[68],2)
        
            return BCE_loss, l2_reg_loss, AUROC_Tasks.mean(),APScore_task.mean(),ROC_AUC_preclinical,ROC_AUC_clinical,ROC_AUC_preclinical_global,ROC_AUC_clinical_global, APScore_preclinical, APScore_clinical, APScore_preclinical_global, APScore_clinical_global 
        else:
            return BCE_loss, l2_reg_loss

    
    def training_step(self, batch, batch_idx):
        if self.compute_AUCs_during_training == True:
            BCE_loss, l2_reg_loss, AUROC_Tasks,APScore_task,ROC_AUC_preclinical,ROC_AUC_clinical,ROC_AUC_preclinical_global,ROC_AUC_clinical_global, APScore_preclinical, APScore_clinical, APScore_preclinical_global, APScore_clinical_global  = self._shared_step(batch, batch_idx, "train")
            total_loss = BCE_loss + self.l2_lambda*l2_reg_loss

            self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

            self.log('*train_ROC_AUC_ALLTasks', AUROC_Tasks, on_step=False, on_epoch=True)
            self.log('*train_APScore_ALLTasks', APScore_task, on_step=False, on_epoch=True)

            self.log('*train_ROC_AUC_preclinical', ROC_AUC_preclinical, on_step=False, on_epoch=True)
            self.log('*train_ROC_AUC_clinical', ROC_AUC_clinical, on_step=False, on_epoch=True)
            self.log('*train_ROC_AUC_preclinical_global', ROC_AUC_preclinical_global, on_step=False, on_epoch=True)
            self.log('*train_ROC_AUC_clinical_global', ROC_AUC_clinical_global, on_step=False, on_epoch=True)

            self.log('*train_APScore_preclinical', APScore_preclinical, on_step=False, on_epoch=True)
            self.log('*train_APScore_clinical', APScore_clinical, on_step=False, on_epoch=True)
            self.log('*train_APScore_preclinical_global', APScore_preclinical_global, on_step=False, on_epoch=True)
            self.log('*train_APScore_clinical_global', APScore_clinical_global, on_step=False, on_epoch=True)
        else:
            BCE_loss, l2_reg_loss  = self._shared_step(batch, batch_idx, "train")
            total_loss = BCE_loss + self.l2_lambda*l2_reg_loss

            self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        if self.compute_AUCs_during_training == True:
            BCE_loss, l2_reg_loss, AUROC_Tasks,APScore_task,ROC_AUC_preclinical,ROC_AUC_clinical,ROC_AUC_preclinical_global,ROC_AUC_clinical_global, APScore_preclinical, APScore_clinical, APScore_preclinical_global, APScore_clinical_global  = self._shared_step(batch, batch_idx, "val")
            total_loss = BCE_loss + self.l2_lambda*l2_reg_loss

            self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

            self.log('*val_ROC_AUC_ALLTasks', AUROC_Tasks, on_step=False, on_epoch=True)
            self.log('*val_APScore_ALLTasks', APScore_task, on_step=False, on_epoch=True)

            self.log('*val_ROC_AUC_preclinical', ROC_AUC_preclinical, on_step=False, on_epoch=True)
            self.log('*val_ROC_AUC_clinical', ROC_AUC_clinical, on_step=False, on_epoch=True)
            self.log('*val_ROC_AUC_preclinical_global', ROC_AUC_preclinical_global, on_step=False, on_epoch=True)
            self.log('*val_ROC_AUC_clinical_global', ROC_AUC_clinical_global, on_step=False, on_epoch=True)

            self.log('*val_APScore_preclinical', APScore_preclinical, on_step=False, on_epoch=True)
            self.log('*val_APScore_clinical', APScore_clinical, on_step=False, on_epoch=True)
            self.log('*val_APScore_preclinical_global', APScore_preclinical_global, on_step=False, on_epoch=True)
            self.log('*val_APScore_clinical_global', APScore_clinical_global, on_step=False, on_epoch=True)
        else:
            BCE_loss, l2_reg_loss  = self._shared_step(batch, batch_idx, "val")
            total_loss = BCE_loss + self.l2_lambda*l2_reg_loss
            self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
#################################################
#Binary classifier for two modalities with all tasks
################################################
class Clinical_Pre_Clinical_Binary_All_Tasks(pl.LightningModule):
    def __init__(self, config):
        super(Clinical_Pre_Clinical_Binary_All_Tasks, self).__init__()

        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.l2_lambda = config['l2_lambda']
        self.l1_lambda = config['l1_lambda']
        self.batch_size = config['batch_size']
        self.tasks = config['num_of_tasks']
        self.compute_AUCs_during_training = config['compute_AUCs_during_training']
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            #self.activ = torch.nn.LeakyReLU(0.05)
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        self.roc_auc = torchmetrics.AUROC(task="binary")
        self.AP_score = torchmetrics.AveragePrecision(task="binary")

        # List of losses
        #self.loss_type = torch.tensor([1.] * self.tasks) # To Balance between tasks
        #self.multi_task_loss = MultiTaskLoss(self.loss_type)  

        self.pos_weight_list = class_weights_for_complete_data(config) # To handle imblance within class
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_list, reduction="none") # Binary loss
        #self.loss_fn = nn.BCEWithLogitsLoss(reduction="none") # Binary loss

        #self.loss_fn = nn.BCELoss(weight=self.pos_weight_list, reduction="none") # Binary loss
        #self.loss_fn = AsymmetricLoss()
        #self.loss_fn = nn.BCELoss(reduction="none")
        # hidden units, size depends upon number of layers
        if self.depth == 'shallow':

            layer_input_dim = 4096
            layer_output_dim = 256
            self.arch.append(nn.Sequential(
                nn.Linear(layer_input_dim, layer_output_dim),
                nn.BatchNorm1d(layer_output_dim),
                self.dropout,
                self.activ))
            # alpha and beta parameters, with softplus link function
            self.arch.append(nn.Sequential(
                    nn.Linear(layer_output_dim, self.tasks),
                    #self.sigmoid
                    ))
            
        elif self.depth == 'super_shallow':

            layer_input_dim = 4096
            self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, self.tasks),
                    #self.sigmoid
                    ))
            
        else:
            N_0 = 4096
            layer_input_dim = 4096
            for layer_number in range(self.depth):
                
                layer_output_dim = int(N_0*(0.5)**(layer_number + 1))
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim),
                    #nn.BatchNorm1d(layer_output_dim),
                    self.dropout,
                    self.activ))
                layer_input_dim = layer_output_dim
            
            # alpha and beta parameters, with softplus link function
            self.arch.append(nn.Sequential(
                    nn.Linear(layer_output_dim, self.tasks),
                    #self.sigmoid
                    ))
        
        
    def forward(self, x):
        for i, layers in enumerate(self.arch):     
            x = layers(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              #weight_decay = self.l2_lambda,
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'min')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_BCE_loss'
        }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def l1_regularization(self):
        device = torch.device('cuda')
        l1_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
        return l1_reg
    
    def _shared_step(self, batch, batch_idx,prefix):
        # compute forward pass
        x, y = batch
        y_hat = self(x)
        y_prob = self.sigmoid(y_hat)

        # Generate a mask to ignore examples with missing labels (-1)
        mask = (y != -1).float()

        #BCE_loss = self.FocalLoss(BCE_loss)
        #BCE_loss = self.AsymmetricUnifiedLoss(y, BCE_loss)
        BCE_loss = self.loss_fn(y_hat, y.float())
        BCE_loss = torch.mul(BCE_loss, mask)
        #BCE_loss = self.multi_task_loss(BCE_loss)

        loss_pos = torch.mul(BCE_loss,y).mean()
        loss_neg = torch.mul(BCE_loss,(1-y)).mean()

        BCE_loss = BCE_loss.mean()
        
        l2_reg_loss = self.l2_regularization()
        #l2_reg_loss = torch.tensor(0., requires_grad=True, device=torch.device('cuda'))
        #l1_reg_loss = self.l1_regularization()
        l1_reg_loss = torch.tensor(0., requires_grad=True, device=torch.device('cuda'))


        if self.compute_AUCs_during_training == True:
            # ROC_AUC and PR_AUC
            AUROC_Tasks, APScore_task = [], []

            for task in range(y.shape[1]):
                y_task_valid = y[:,task][mask[:,task]].round().to(torch.int64)
                pred_task_valid = y_prob[:,task][mask[:,task]]
                AUROC_Tasks.append(self.roc_auc(pred_task_valid, y_task_valid).item())
                APScore_task.append(self.AP_score(pred_task_valid, y_task_valid).item())

            AUROC_Tasks = np.array(AUROC_Tasks)
            APScore_task = np.array(APScore_task)

            # Modalilty wide calculations
            ROC_AUC_preclinical = np.around(AUROC_Tasks[:18].mean(),2)
            ROC_AUC_clinical = np.around(AUROC_Tasks[18:67].mean(),2)
            ROC_AUC_preclinical_global = np.around(AUROC_Tasks[67],2)
            ROC_AUC_clinical_global = np.around(AUROC_Tasks[68],2)

            APScore_preclinical = np.around(APScore_task[:18].mean(),2)
            APScore_clinical = np.around(APScore_task[18:67].mean(),2)
            APScore_preclinical_global = np.around(APScore_task[67],2)
            APScore_clinical_global = np.around(APScore_task[68],2)
        
            return BCE_loss, l2_reg_loss,l1_reg_loss, AUROC_Tasks.mean(),APScore_task.mean(),ROC_AUC_preclinical,ROC_AUC_clinical,ROC_AUC_preclinical_global,ROC_AUC_clinical_global, APScore_preclinical, APScore_clinical, APScore_preclinical_global, APScore_clinical_global 
        else:
            return BCE_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg

    
    def training_step(self, batch, batch_idx):
        if self.compute_AUCs_during_training == True:
            BCE_loss, l2_reg_loss,l1_reg_loss, AUROC_Tasks,APScore_task,ROC_AUC_preclinical,ROC_AUC_clinical,ROC_AUC_preclinical_global,ROC_AUC_clinical_global, APScore_preclinical, APScore_clinical, APScore_preclinical_global, APScore_clinical_global  = self._shared_step(batch, batch_idx, "train")
            total_loss = BCE_loss + self.l2_lambda*l2_reg_loss + self.l1_lambda*l1_reg_loss 

            self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

            self.log('*train_ROC_AUC_ALLTasks', AUROC_Tasks, on_step=False, on_epoch=True)
            self.log('*train_APScore_ALLTasks', APScore_task, on_step=False, on_epoch=True)

            self.log('*train_ROC_AUC_preclinical', ROC_AUC_preclinical, on_step=False, on_epoch=True)
            self.log('*train_ROC_AUC_clinical', ROC_AUC_clinical, on_step=False, on_epoch=True)
            self.log('*train_ROC_AUC_preclinical_global', ROC_AUC_preclinical_global, on_step=False, on_epoch=True)
            self.log('*train_ROC_AUC_clinical_global', ROC_AUC_clinical_global, on_step=False, on_epoch=True)

            self.log('*train_APScore_preclinical', APScore_preclinical, on_step=False, on_epoch=True)
            self.log('*train_APScore_clinical', APScore_clinical, on_step=False, on_epoch=True)
            self.log('*train_APScore_preclinical_global', APScore_preclinical_global, on_step=False, on_epoch=True)
            self.log('*train_APScore_clinical_global', APScore_clinical_global, on_step=False, on_epoch=True)
        else:
            BCE_loss, l2_reg_loss, l1_reg_loss, loss_pos, loss_neg  = self._shared_step(batch, batch_idx, "train")
            l2_reg_loss = self.l2_lambda*l2_reg_loss
            l1_reg_loss = self.l1_lambda*l1_reg_loss
            total_loss = BCE_loss + l2_reg_loss + l1_reg_loss

            self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_loss_pos', loss_pos, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_loss_neg', loss_neg, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_l1_reg_loss', l1_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        if self.compute_AUCs_during_training == True:
            BCE_loss, l2_reg_loss,l1_reg_loss, AUROC_Tasks,APScore_task,ROC_AUC_preclinical,ROC_AUC_clinical,ROC_AUC_preclinical_global,ROC_AUC_clinical_global, APScore_preclinical, APScore_clinical, APScore_preclinical_global, APScore_clinical_global  = self._shared_step(batch, batch_idx, "val")
            total_loss = BCE_loss + self.l2_lambda*l2_reg_loss + self.l1_lambda*l1_reg_loss

            self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

            self.log('*val_ROC_AUC_ALLTasks', AUROC_Tasks, on_step=False, on_epoch=True)
            self.log('*val_APScore_ALLTasks', APScore_task, on_step=False, on_epoch=True)

            self.log('*val_ROC_AUC_preclinical', ROC_AUC_preclinical, on_step=False, on_epoch=True)
            self.log('*val_ROC_AUC_clinical', ROC_AUC_clinical, on_step=False, on_epoch=True)
            self.log('*val_ROC_AUC_preclinical_global', ROC_AUC_preclinical_global, on_step=False, on_epoch=True)
            self.log('*val_ROC_AUC_clinical_global', ROC_AUC_clinical_global, on_step=False, on_epoch=True)

            self.log('*val_APScore_preclinical', APScore_preclinical, on_step=False, on_epoch=True)
            self.log('*val_APScore_clinical', APScore_clinical, on_step=False, on_epoch=True)
            self.log('*val_APScore_preclinical_global', APScore_preclinical_global, on_step=False, on_epoch=True)
            self.log('*val_APScore_clinical_global', APScore_clinical_global, on_step=False, on_epoch=True)
        else:
            BCE_loss, l2_reg_loss , l1_reg_loss, loss_pos, loss_neg = self._shared_step(batch, batch_idx, "val")
            l2_reg_loss = self.l2_lambda*l2_reg_loss
            l1_reg_loss = self.l1_lambda*l1_reg_loss
            total_loss = BCE_loss + l2_reg_loss + l1_reg_loss
            self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_loss_pos', loss_pos, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_loss_neg', loss_neg, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
#################################################
# Beta severity model for TG_GATES_JNJ_MiniTox mixed data without extra features
#################################################
class TG_JnJ_MiniTox_Beta_model_dt(LightningModule):
    
    '''
    - Multitask for 15 taks 
    - dose-time information addition in the second last layer
    '''
    def __init__(self, config):
        super().__init__()
        self.input_feature_dim = config['input_feature_dim']
        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.normalized_weighted_loss = config['normalized_weighted_loss']
        self.gamma = config['gamma']
        self.normalized_loss = config['normalized_loss']
        self.scaled_loss = config['scaled_loss']
        self.tasks = config['num_of_tasks']
        self.num_bottleneck_features = config['num_bottleneck_features']
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()
        
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        layer_input_dim = 4096
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        layer_input_dim = 4096
        for layer_number in range(self.depth):
            
            layer_output_dim = int(N_0*(0.5)**(lambd * (layer_number + 1)))
            if layer_number == self.depth -1:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim - self.num_bottleneck_features),
                    self.dropout,
                    self.activ))
            else:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim),
                    self.dropout,
                    self.activ))
            layer_input_dim = layer_output_dim
                
        # alpha and beta parameters, with softplus link function
        self.arch.append(nn.Sequential(
                nn.Linear(layer_output_dim, self.tasks * 2),
                Exp_lin()))
            
            
    def forward(self, x, d, t, dpf):

        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        #dpf = torch.reshape(dpf,(-1,5))
        #dt_dpf = torch.concat((d,t,dpf),axis = 1)
        dt = torch.concat((d,t),axis = 1)

        for i, layers in enumerate(self.arch):
            if i == self.depth:
                x = layers(torch.cat((x,dt), dim = 1))
            else:      
                x = layers(x)
        #print(x.max(), x.min())
        return x

    def SoftPlus_Normalized_weights(self, y, gamma):
        softplus_function = torch.nn.Softplus(beta = gamma, threshold=20)
        normalized_weights = softplus_function(y)/softplus_function(y).sum()
        return normalized_weights
    
    def class_ELL(self, y, class_label):
        #print(y.shape, self.LL.shape)
        mask = y.eq(class_label)
        class_ELL = torch.masked_select(self.LL, mask).mean()
        
        return class_ELL
    
    def log_prior(self, model, sigma):
        device = torch.device('cuda')
        norma_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([sigma], device=device))
        num_param = torch.Tensor([0]).to(device)
        log_prior_total = torch.tensor(0., requires_grad=True).to(device)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                prior_prob = norma_dist.log_prob(param)
                log_prior_total = log_prior_total + prior_prob.sum()
                num_param += len(param.view(-1))
        return num_param, log_prior_total
    
    def compute_loss(self, alphas, betas, y):
        self.eps = 1e-5
        
        # to handle sparsity
        mask = torch.isnan(y)
        
        y_without_nan = y.nan_to_num(0.5)
        y_true_clipped = y_without_nan.clip(self.eps, 1 - self.eps)
        
        dist =  Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2))
        self.LL = dist.log_prob(y_true_clipped)
        self.LL = self.LL[~mask]
        
       # print(y.shape, y_true_clipped[~mask].shape,self.LL.shape )
        
        if self.normalized_weighted_loss == 1:
            normalized_weights =  self.SoftPlus_Normalized_weights(y[~mask], self.gamma)
            self.weighted_LL = normalized_weights * self.LL
 
        num_param, log_prior = self.log_prior(self, self.prior_var)
        EoM = (torch.abs((Beta(alphas, betas).mean - y.nanmean(axis =2)))).mean()
        EoV = (torch.abs((Beta(alphas, betas).variance - torch.var(y, axis = 2)))).mean()
        
        if self.normalized_weighted_loss == 1:
            return self.weighted_LL, num_param, log_prior, EoM, EoV
        else:
            return self.LL, num_param, log_prior, EoM, EoV
    
    def _shared_step(self, batch, batch_nb,prefix):
        # compute forward pass
        x, d, t, dpf, y = batch
        alphas_betas = self(x,d,t, dpf)
        alphas, betas = alphas_betas[:,0:self.tasks],  alphas_betas[:,self.tasks:]
        
        LL, num_param, log_P, EoM, EoV = self.compute_loss(alphas, betas,y)
            
        # compute desired losses
        E_log_P = log_P / num_param
        ΣLL = LL.sum()
        ELL = LL.mean() 
        regular_loss = -(ΣLL + log_P)
        normalized_loss = -(ELL + E_log_P)
        scaled_loss = -(ΣLL + E_log_P)
        #print('loss',ΣLL)
        # class dependent likelihood
        tox_classes = [0.0,0.2,0.4,0.6,0.8,1.0]
        
        # to handle sparsity
        mask = torch.isnan(y)
        
        for class_label in tox_classes:
            class_dependent_LL =  self.class_ELL(y[~mask],class_label)
            self.log(f"ELL_{class_label}_{prefix}",class_dependent_LL, prog_bar=True, on_step=False, on_epoch=True)
            
        # log all the metrics
        self.log(f"ΣLL_{prefix}",ΣLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"ELL_{prefix}",ELL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"log_prior_{prefix}",log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"E_log_P_{prefix}",E_log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"regular_loss_{prefix}",regular_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"normalized_loss_{prefix}",normalized_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"scaled_loss_{prefix}",scaled_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoM_{prefix}",EoM, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoV_{prefix}",EoV, prog_bar=True, on_step=False, on_epoch=True)
        
        return regular_loss, normalized_loss, scaled_loss
        
    def training_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "train")
        if self.normalized_loss == 1:
            return normalized_loss
        
        elif self.scaled_loss == 1:
            return scaled_loss
        else:
            return regular_loss 
    
    def validation_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "val")

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'ΣLL_val'
        }
        
    def training_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "train")
        if self.normalized_loss == 1:
            return normalized_loss
        
        elif self.scaled_loss == 1:
            return scaled_loss
        else:
            return regular_loss 
    
    def validation_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "val")

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'ΣLL_val'
        }
#################################################
# Beta severity model for TG_GATES_JNJ_MiniTox mixed data
#################################################
class TG_JnJ_MiniTox_Beta_model(LightningModule):
    
    '''
    - Multitask for 15 taks 
    - dose-time information addition in the second last layer
    '''
    def __init__(self, config):
        super().__init__()
        self.input_feature_dim = config['input_feature_dim']
        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.normalized_weighted_loss = config['normalized_weighted_loss']
        self.gamma = config['gamma']
        self.normalized_loss = config['normalized_loss']
        self.scaled_loss = config['scaled_loss']
        self.tasks = config['num_of_tasks']
        self.num_bottleneck_features = config['num_bottleneck_features']
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()
        
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        layer_input_dim = 4096
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        layer_input_dim = 4096
        for layer_number in range(self.depth):
            
            layer_output_dim = int(N_0*(0.5)**(lambd * (layer_number + 1)))
            if layer_number == self.depth -1:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim - self.num_bottleneck_features),
                    self.dropout,
                    self.activ))
            else:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim),
                    self.dropout,
                    self.activ))
            layer_input_dim = layer_output_dim
                
        # alpha and beta parameters, with softplus link function
        self.arch.append(nn.Sequential(
                nn.Linear(layer_output_dim, self.tasks * 2),
                Exp_lin()))
            
            
    def forward(self, x, d, t, dpf):

        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dpf = torch.reshape(dpf,(-1,5))
        dt_dpf = torch.concat((d,t,dpf),axis = 1)

        for i, layers in enumerate(self.arch):
            if i == self.depth:
                x = layers(torch.cat((x,dt_dpf), dim = 1))
            else:      
                x = layers(x)
        #print(x.max(), x.min())
        return x

    def SoftPlus_Normalized_weights(self, y, gamma):
        softplus_function = torch.nn.Softplus(beta = gamma, threshold=20)
        normalized_weights = softplus_function(y)/softplus_function(y).sum()
        return normalized_weights
    
    def class_ELL(self, y, class_label):
        #print(y.shape, self.LL.shape)
        mask = y.eq(class_label)
        class_ELL = torch.masked_select(self.LL, mask).mean()
        
        return class_ELL
    
    def log_prior(self, model, sigma):
        device = torch.device('cuda')
        norma_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([sigma], device=device))
        num_param = torch.Tensor([0]).to(device)
        log_prior_total = torch.tensor(0., requires_grad=True).to(device)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                prior_prob = norma_dist.log_prob(param)
                log_prior_total = log_prior_total + prior_prob.sum()
                num_param += len(param.view(-1))
        return num_param, log_prior_total
    
    def compute_loss(self, alphas, betas, y):
        self.eps = 1e-5
        
        # to handle sparsity
        mask = torch.isnan(y)
        
        y_without_nan = y.nan_to_num(0.5)
        y_true_clipped = y_without_nan.clip(self.eps, 1 - self.eps)
        
        dist =  Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2))
        self.LL = dist.log_prob(y_true_clipped)
        self.LL = self.LL[~mask]
        
       # print(y.shape, y_true_clipped[~mask].shape,self.LL.shape )
        
        if self.normalized_weighted_loss == 1:
            normalized_weights =  self.SoftPlus_Normalized_weights(y[~mask], self.gamma)
            self.weighted_LL = normalized_weights * self.LL
 
        num_param, log_prior = self.log_prior(self, self.prior_var)
        EoM = (torch.abs((Beta(alphas, betas).mean - y.nanmean(axis =2)))).mean()
        EoV = (torch.abs((Beta(alphas, betas).variance - torch.var(y, axis = 2)))).mean()
        
        if self.normalized_weighted_loss == 1:
            return self.weighted_LL, num_param, log_prior, EoM, EoV
        else:
            return self.LL, num_param, log_prior, EoM, EoV
    
    def _shared_step(self, batch, batch_nb,prefix):
        # compute forward pass
        x, d, t, dpf, y = batch
        alphas_betas = self(x,d,t, dpf)
        alphas, betas = alphas_betas[:,0:self.tasks],  alphas_betas[:,self.tasks:]
        
        LL, num_param, log_P, EoM, EoV = self.compute_loss(alphas, betas,y)
            
        # compute desired losses
        E_log_P = log_P / num_param
        ΣLL = LL.sum()
        ELL = LL.mean() 
        regular_loss = -(ΣLL + log_P)
        normalized_loss = -(ELL + E_log_P)
        scaled_loss = -(ΣLL + E_log_P)
        #print('loss',ΣLL)
        # class dependent likelihood
        tox_classes = [0.0,0.2,0.4,0.6,0.8,1.0]
        
        # to handle sparsity
        mask = torch.isnan(y)
        
        for class_label in tox_classes:
            class_dependent_LL =  self.class_ELL(y[~mask],class_label)
            self.log(f"ELL_{class_label}_{prefix}",class_dependent_LL, prog_bar=True, on_step=False, on_epoch=True)
            
        # log all the metrics
        self.log(f"ΣLL_{prefix}",ΣLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"ELL_{prefix}",ELL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"log_prior_{prefix}",log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"E_log_P_{prefix}",E_log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"regular_loss_{prefix}",regular_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"normalized_loss_{prefix}",normalized_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"scaled_loss_{prefix}",scaled_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoM_{prefix}",EoM, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoV_{prefix}",EoV, prog_bar=True, on_step=False, on_epoch=True)
        
        return regular_loss, normalized_loss, scaled_loss
        
    def training_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "train")
        if self.normalized_loss == 1:
            return normalized_loss
        
        elif self.scaled_loss == 1:
            return scaled_loss
        else:
            return regular_loss 
    
    def validation_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "val")

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'ΣLL_val'
        }
        
    def training_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "train")
        if self.normalized_loss == 1:
            return normalized_loss
        
        elif self.scaled_loss == 1:
            return scaled_loss
        else:
            return regular_loss 
    
    def validation_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "val")

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'ΣLL_val'
        }
#################################################
# Binary classifier with BCE loss
#################################################
class BinaryClassifier(pl.LightningModule):
    def __init__(self, config):
        super(BinaryClassifier, self).__init__()

        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.l2_lambda = config['l2_lambda']
        self.batch_size = config['batch_size']
        self.pos_weight = config['pos_weight']
        self.tasks = config['num_of_tasks']
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        layer_input_dim = 4096
        for layer_number in range(self.depth):
            
            layer_output_dim = int(N_0*(0.5)**(lambd * (layer_number + 1)))
            if layer_number == self.depth -1:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim - 2),
                    self.dropout,
                    self.activ))
            else:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim),
                    self.dropout,
                    self.activ))
            layer_input_dim = layer_output_dim
                
        # alpha and beta parameters, with softplus link function
        self.arch.append(nn.Sequential(
                nn.Linear(layer_output_dim, self.tasks)))
        
    def forward(self, x,d,t):
        
        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dt = torch.concat((d,t),axis = 1)
        for i, layers in enumerate(self.arch):
            if i == self.depth:
                x = layers(torch.cat((x,dt), dim = 1))
            else:      
                x = layers(x)
        return x
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_avg_pr_auc'
        }
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _shared_step(self, batch, batch_idx,prefix):
        # compute forward pass
        x, d, t, y = batch
        y_hat = self(x,d,t)
        BCE_loss = self.loss_fn(y_hat, y.unsqueeze(1).float())
        l2_reg_loss = self.l2_regularization()

        y_hat_probs = torch.sigmoid(y_hat)
        roc_auc = roc_auc_score(y.cpu().numpy(), y_hat_probs.detach().cpu().numpy())
        avg_pr_auc = average_precision_score(y.cpu().numpy(), y_hat_probs.detach().cpu().numpy())
        
        return BCE_loss, l2_reg_loss, roc_auc, avg_pr_auc
    
    def training_step(self, batch, batch_idx):

        BCE_loss, l2_reg_loss, roc_auc, avg_pr_auc = self._shared_step(batch, batch_idx, "train")
        total_loss = BCE_loss + self.l2_lambda*l2_reg_loss

        self.log('train_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.log('train_roc_auc', roc_auc, on_step=False, on_epoch=True)
        self.log('train_avg_pr_auc', avg_pr_auc, on_step=False, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        
        BCE_loss, l2_reg_loss, roc_auc, avg_pr_auc = self._shared_step(batch, batch_idx, "val")
        total_loss = BCE_loss + self.l2_lambda*l2_reg_loss

        self.log('val_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_roc_auc', roc_auc, on_step=False, on_epoch=True)
        self.log('val_avg_pr_auc', avg_pr_auc, on_step=False, on_epoch=True)

        
    def test_step(self, batch, batch_idx):
        BCE_loss, l2_reg_loss, roc_auc, avg_pr_auc = self._shared_step(batch, batch_idx, "test")
        total_loss = BCE_loss + self.l2_lambda*l2_reg_loss, 

        self.log('test_BCE_loss', BCE_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.log('test_roc_auc', roc_auc, on_step=False, on_epoch=True)
        self.log('val_avg_pr_auc', avg_pr_auc, on_step=False, on_epoch=True)

#################################################
# Beta severity model for TG_GATES_JNJ mixed data
#################################################
class TG_JnJ_Beta_model(LightningModule):
    
    '''
    - Multitask for 15 taks 
    - dose-time information addition in the second last layer
    '''
    def __init__(self, config):
        super().__init__()
        self.input_feature_dim = config['input_feature_dim']
        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.normalized_weighted_loss = config['normalized_weighted_loss']
        self.gamma = config['gamma']
        self.normalized_loss = config['normalized_loss']
        self.scaled_loss = config['scaled_loss']
        self.tasks = config['num_of_tasks']
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()
        
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        layer_input_dim = 4096
        for layer_number in range(self.depth):
            
            layer_output_dim = int(N_0*(0.5)**(lambd * (layer_number + 1)))
            if layer_number == self.depth -1:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim - 2),
                    self.dropout,
                    self.activ))
            else:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim),
                    self.dropout,
                    self.activ))
            layer_input_dim = layer_output_dim
                
        # alpha and beta parameters, with softplus link function
        self.arch.append(nn.Sequential(
                nn.Linear(layer_output_dim, self.tasks * 2),
                Exp_lin()))
            
    def forward(self, x, d, t):
        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dt = torch.concat((d,t),axis = 1)
        for i, layers in enumerate(self.arch):
            if i == self.depth:
                x = layers(torch.cat((x,dt), dim = 1))
            else:      
                x = layers(x)
        return x
    
    def SoftPlus_Normalized_weights(self, y, gamma):
        softplus_function = torch.nn.Softplus(beta = gamma, threshold=20)
        normalized_weights = softplus_function(y)/softplus_function(y).sum()
        return normalized_weights
    
    def class_ELL(self, y, class_label):
        #print(y.shape, self.LL.shape)
        mask = y.eq(class_label)
        class_ELL = torch.masked_select(self.LL, mask).mean()
        
        return class_ELL
    
    def log_prior(self, model, sigma):
        device = torch.device('cuda')
        norma_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([sigma], device=device))
        num_param = torch.Tensor([0]).to(device)
        log_prior_total = torch.tensor(0., requires_grad=True).to(device)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                prior_prob = norma_dist.log_prob(param)
                log_prior_total = log_prior_total + prior_prob.sum()
                num_param += len(param.view(-1))
        return num_param, log_prior_total
    
    def compute_loss(self, alphas, betas, y):
        self.eps = 1e-5
        
        # to handle sparsity
        mask = torch.isnan(y)
        
        y_without_nan = y.nan_to_num(0.5)
        y_true_clipped = y_without_nan.clip(self.eps, 1 - self.eps)
        
        dist =  Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2))
        self.LL = dist.log_prob(y_true_clipped)
        self.LL = self.LL[~mask]
        
       # print(y.shape, y_true_clipped[~mask].shape,self.LL.shape )
        
        if self.normalized_weighted_loss == 1:
            normalized_weights =  self.SoftPlus_Normalized_weights(y[~mask], self.gamma)
            self.weighted_LL = normalized_weights * self.LL
 
        num_param, log_prior = self.log_prior(self, self.prior_var)
        EoM = (torch.abs((Beta(alphas, betas).mean - y.nanmean(axis =2)))).mean()
        EoV = (torch.abs((Beta(alphas, betas).variance - torch.var(y, axis = 2)))).mean()
        
        if self.normalized_weighted_loss == 1:
            return self.weighted_LL, num_param, log_prior, EoM, EoV
        else:
            return self.LL, num_param, log_prior, EoM, EoV
    
    def _shared_step(self, batch, batch_nb,prefix):
        # compute forward pass
        x, d, t, y = batch
        alphas_betas = self(x,d,t)
        alphas, betas = alphas_betas[:,0:self.tasks],  alphas_betas[:,self.tasks:]
        
        LL, num_param, log_P, EoM, EoV = self.compute_loss(alphas, betas,y)
            
        # compute desired losses
        E_log_P = log_P / num_param
        ΣLL = LL.sum()
        ELL = LL.mean() 
        regular_loss = -(ΣLL + log_P)
        normalized_loss = -(ELL + E_log_P)
        scaled_loss = -(ΣLL + E_log_P)
        
        # class dependent likelihood
        tox_classes = [0.0,0.2,0.4,0.6,0.8,1.0]
        
        # to handle sparsity
        mask = torch.isnan(y)
        
        for class_label in tox_classes:
            class_dependent_LL =  self.class_ELL(y[~mask],class_label)
            self.log(f"ELL_{class_label}_{prefix}",class_dependent_LL, prog_bar=True, on_step=False, on_epoch=True)
            
        # log all the metrics
        self.log(f"ΣLL_{prefix}",ΣLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"ELL_{prefix}",ELL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"log_prior_{prefix}",log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"E_log_P_{prefix}",E_log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"regular_loss_{prefix}",regular_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"normalized_loss_{prefix}",normalized_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"scaled_loss_{prefix}",scaled_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoM_{prefix}",EoM, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoV_{prefix}",EoV, prog_bar=True, on_step=False, on_epoch=True)
        
        return regular_loss, normalized_loss, scaled_loss
        
    def training_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "train")
        if self.normalized_loss == 1:
            return normalized_loss
        
        elif self.scaled_loss == 1:
            return scaled_loss
        else:
            return regular_loss 
    
    def validation_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "val")

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'ΣLL_val'
        }
        
    def training_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "train")
        if self.normalized_loss == 1:
            return normalized_loss
        
        elif self.scaled_loss == 1:
            return scaled_loss
        else:
            return regular_loss 
    
    def validation_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "val")

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'ΣLL_val'
        }
#################################################
# Beta severity model for pahtology
#################################################
class Beta_Severity_dt_last_layer(LightningModule):
    
    '''
    - Multitask/Single Task Model 
    - dose-time information addition in the second last layer
    '''
    def __init__(self, config):
        super().__init__()
        self.input_feature_dim = config['input_feature_dim']
        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.normalized_weighted_loss = config['normalized_weighted_loss']
        self.gamma = config['gamma']
        self.normalized_loss = config['normalized_loss']
        self.scaled_loss = config['scaled_loss']
        self.tasks = config['num_of_tasks']
        
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        self.arch = nn.ModuleList()
        
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        layer_input_dim = 4096
        for layer_number in range(self.depth):
            
            layer_output_dim = int(N_0*(0.5)**(lambd * (layer_number + 1)))
            if layer_number == self.depth -1:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim - 2),
                    self.dropout,
                    self.activ))
            else:
                self.arch.append(nn.Sequential(
                    nn.Linear(layer_input_dim, layer_output_dim),
                    self.dropout,
                    self.activ))
            layer_input_dim = layer_output_dim
                
        # alpha and beta parameters, with softplus link function
        self.arch.append(nn.Sequential(
                nn.Linear(layer_output_dim, self.tasks * 2),
                Exp_lin()))
            
    def forward(self, x, d, t):
        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dt = torch.concat((d,t),axis = 1)
        for i, layers in enumerate(self.arch):
            if i == self.depth:
                x = layers(torch.cat((x,dt), dim = 1))
            else:      
                x = layers(x)
        return x
    
    def SoftPlus_Normalized_weights(self, y, gamma):
        softplus_function = torch.nn.Softplus(beta = gamma, threshold=20)
        normalized_weights = softplus_function(y)/softplus_function(y).sum()
        return normalized_weights
    
    def class_ELL(self, y, class_label):
        
        mask = y.eq(class_label)
        class_ELL = torch.masked_select(self.weighted_LL, mask).mean()
        return class_ELL
    
    def log_prior(self, model, sigma):
        device = torch.device('cuda')
        norma_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([sigma], device=device))
        num_param = torch.Tensor([0]).to(device)
        log_prior_total = torch.tensor(0., requires_grad=True).to(device)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                prior_prob = norma_dist.log_prob(param)
                log_prior_total = log_prior_total + prior_prob.sum()
                num_param += len(param.view(-1))
        return num_param, log_prior_total
    
    def compute_loss(self, alphas, betas, y):
        self.eps = 1e-5
        y_true_clipped = y.clip(self.eps, 1 - self.eps)
        #print(alphas.min().item(), betas.min().item(), alphas.max().item(), betas.max().item())

        dist =  Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2))
        
        self.LL = dist.log_prob(y_true_clipped)
        if self.normalized_weighted_loss == 1:
            normalized_weights =  self.SoftPlus_Normalized_weights(y, self.gamma)
            self.weighted_LL = normalized_weights * self.LL
 
        num_param, log_prior = self.log_prior(self, self.prior_var)
        EoM = (torch.abs((Beta(alphas, betas).mean - y.nanmean(axis =2)))).mean()
        EoV = (torch.abs((Beta(alphas, betas).variance - torch.var(y, axis = 2)))).mean()
        
        if self.normalized_weighted_loss == 1:
            return self.weighted_LL, num_param, log_prior, EoM, EoV
        else:
            return self.LL, num_param, log_prior, EoM, EoV
    
    def test_loss(self, alphas, betas, y):
        self.eps = 1e-5
        mask = torch.isnan(y)
        
        y_true_clipped = y.clip(self.eps, 1 - self.eps)
        
        #print(alphas.min().item(), betas.min().item(), alphas.max().item(), betas.max().item())

        dist =  Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2))
        
        LL = dist.log_prob(y_true_clipped[~mask])
        if self.normalized_weighted_loss == 1:
            normalized_weights =  self.SoftPlus_Normalized_weights(y, self.gamma)
            LL = normalized_weights * LL
 
        num_param, log_P = self.log_prior(self, self.prior_var)
        EoM = (torch.abs((Beta(alphas, betas).mean - y.mean(axis =2)))).mean()
        EoV = (torch.abs((Beta(alphas, betas).variance - torch.var(y, axis = 2)))).mean()
        return LL, num_param, log_P, EoM, EoV
    
    def _shared_step(self, batch, batch_nb,prefix):
        # compute forward pass
        x, d, t, y = batch
        alphas_betas = self(x,d,t)
        alphas, betas = alphas_betas[:,0:self.tasks],  alphas_betas[:,self.tasks:]
        
        if prefix == 'test':
            LL, num_param, log_P, EoM, EoV = self.test_loss(alphas, betas,y)
        
        if (prefix == 'train') or (prefix == 'val'):
            LL, num_param, log_P, EoM, EoV = self.compute_loss(alphas, betas,y)
            
        # compute desired losses
        E_log_P = log_P / num_param
        ΣLL = LL.sum()
        ELL = LL.mean() 
        regular_loss = -(ΣLL + log_P)
        normalized_loss = -(ELL + E_log_P)
        scaled_loss = -(ΣLL + E_log_P)
        
        # class dependent likelihood
        tox_classes = [0.0,0.2,0.4,0.6,0.8,1.0]
        for class_label in tox_classes:
            class_dependent_LL =  self.class_ELL(y,class_label)
            self.log(f"ELL_{class_label}_{prefix}",class_dependent_LL, prog_bar=True, on_step=False, on_epoch=True)
            
        # log all the metrics
        self.log(f"ΣLL_{prefix}",ΣLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"ELL_{prefix}",ELL, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"log_prior_{prefix}",log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"E_log_P_{prefix}",E_log_P, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"regular_loss_{prefix}",regular_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"normalized_loss_{prefix}",normalized_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"scaled_loss_{prefix}",scaled_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoM_{prefix}",EoM, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"EoV_{prefix}",EoV, prog_bar=True, on_step=False, on_epoch=True)
        
        return regular_loss, normalized_loss, scaled_loss
        
    def training_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "train")
        if self.normalized_loss == 1:
            return normalized_loss
        
        elif self.scaled_loss == 1:
            return scaled_loss
        else:
            return regular_loss 
    
    def validation_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "val")
        
    def test_step(self, batch, batch_nb):
        regular_loss, normalized_loss, scaled_loss = self._shared_step(batch, batch_nb, "test")

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'max')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'ΣLL_val'
        }

#################################################
# Pathologies and Blood markers
#################################################

class Beta_Severity_PBM(LightningModule):

    def __init__(self, config):
        
        super().__init__()
        self.input_feature_dim = config['input_feature_dim']
        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.scaling_factor = 1/self.batch_size
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.arch = nn.ModuleList()
        
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        for layer_number in range(self.depth):
            
            if layer_number == 0:
                layer_input_dim = 4098
            else:
                layer_input_dim = layer_output_dim
                
            layer_output_dim = int(N_0*(0.5)**(lambd * (layer_number + 1)))
            self.arch.append(nn.Sequential(
                nn.Linear(layer_input_dim, layer_output_dim),
                self.dropout,
                self.activ
            ))
        
        # alpha and beta parameters, with softplus link function
        self.arch.append(nn.Sequential(
                nn.Linear(layer_output_dim, 70),
                nn.Softplus()))


    def forward(self, x, d, t):
        
        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dt = torch.concat((d,t),axis = 1)
        x = torch.cat((x,dt), dim = 1)

        for layers in self.arch:
            x = layers(x)
        return x
    
    def compute_loss(self, alphas, betas, y):
        
        # masking
        self.eps = 1e-5
        mask = torch.isnan(y)
        masked_y = torch.nan_to_num(y, nan = 0.5)
        y_true = masked_y.clip(self.eps, 1 - self.eps)
        
        # normailized by number of examples
        # remove maked values
        
        E_NLL =  - Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2)).log_prob(y_true)
        #print(y_true.shape, mask.shape, E_NLL[~mask].shape)
        E_NLL = E_NLL[~mask].mean()
        
        log_prior = self.log_prior(self, self.prior_var)
        
        beta_dist = Beta(alphas, betas)
        MSE = (beta_dist.mean - y.nanmean(axis =2)) ** 2
        EMSE = MSE.nanmean()
        return E_NLL, log_prior, EMSE
    
    def compute_loss_val(self, alphas, betas, y):
        
        # masking
        self.eps = 1e-5
        mask = torch.isnan(y)
        masked_y = torch.nan_to_num(y, nan = 0.5)
        y_true = masked_y.clip(self.eps, 1 - self.eps)
        
        # normailized by number of examples
        # remove maked values
        
        E_NLL =  - Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2)).log_prob(y_true)
        task_ENLL = E_NLL.nanmean(axis = [0,2])
        ENLL_Pathology = E_NLL[:,0:14,:].mean()
        E_NLL = E_NLL[~mask].mean()
        ENLL_BM = E_NLL - ENLL_Pathology
        
        log_prior = self.log_prior(self, self.prior_var)
        
        beta_dist = Beta(alphas, betas)
        MSE = (beta_dist.mean - y.nanmean(axis =2)) ** 2
        EMSE = MSE.nanmean()
        return E_NLL, log_prior, EMSE, ENLL_Pathology, ENLL_BM
    
    def log_prior(self, model, sigma):
        device = torch.device('cuda')
        norma_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([sigma], device=device))
        num_param = torch.Tensor([0]).to(device)
        log_prior_total = torch.tensor(0., requires_grad=True).to(device)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                prior_prob = norma_dist.log_prob(param)
                log_prior_total = log_prior_total + prior_prob.sum()
                num_param += len(param.view(-1))
        log_prior_total = log_prior_total/num_param
        return log_prior_total
    
    def training_step(self, batch, batch_nb):
        #print("batch", x.shape, y.shape)
        x, d, t, y = batch
        alphas_betas = self(x,d,t)
        alphas, betas = alphas_betas[:,0:35],  alphas_betas[:,35:70]
        E_NLL, log_prior, EMSE = self.compute_loss(alphas, betas,y)
        
        #self.scaling_factor = 1/x.shape[0]
        #log_prior =  self.scaling_factor * log_prior
   
        loss = E_NLL - log_prior
        #print(E_NLL, log_prior)
        self.log("E_NLL_train",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("log_p_train",log_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_train",loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("EMSE_train",EMSE, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, d, t, y = batch
        alphas_betas = self(x,d,t)
        alphas, betas = alphas_betas[:,0:35],  alphas_betas[:,35:70]
        E_NLL, log_prior, EMSE, ENLL_Pathology, ENLL_BM = self.compute_loss_val(alphas, betas,y)
        #log_prior =  self.scaling_factor * log_prior

        loss = E_NLL - log_prior
        self.log("E_NLL_val",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("ENLL_val_Pathology",ENLL_Pathology, prog_bar=True, on_step=False, on_epoch=True)
        self.log("ENLL_val_BM",ENLL_BM, prog_bar=True, on_step=False, on_epoch=True)
        self.log("E_NLL_val",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("log_p_val",log_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_val",loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("EMSE_val",EMSE, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=11,
                                                                    min_lr=1e-6,
                                                                    mode = 'min')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'E_NLL_val'
        }
#################################################
# Multitask ndt beta severity model
#################################################
class Beta_Severity(LightningModule):

    def __init__(self, config):
        
        super().__init__()
        
        self.shared_layer_size = config['shared_layer_size']
        self.mixed_layer_size = config['mixed_layer_size']
        self.activation = config['activation']
        self.optim = config['optim']
        self.lambda_p = config['lambda']
        self.lr = config['lr']
        self.momentum = config['momentum']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.scaling_factor = 1/self.batch_size
        self.w_loss = config['weighted_loss']
        self.gamma = config['gamma']
        
        self.head_layer_size = self.mixed_layer_size + 2
        self.jitter = 1e-6
        self.eps = 1e-6
        
        self.shared1 = nn.Linear(4096, self.shared_layer_size)
        self.shared2 = nn.Linear(self.shared_layer_size, self.shared_layer_size)
        self.shared3 = nn.Linear(self.shared_layer_size, self.mixed_layer_size)
        
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()    
        
        self.alphas = nn.ModuleList()
        self.betas = nn.ModuleList()
        for _ in range(14):
            self.alphas.append(nn.Sequential(
                nn.Linear(self.head_layer_size, self.head_layer_size),
                self.activ,
                nn.Linear(self.head_layer_size, 1),
                nn.Softplus()))
            
            self.betas.append(nn.Sequential(
                nn.Linear(self.head_layer_size, self.head_layer_size),
                self.activ,
                nn.Linear(self.head_layer_size, 1),
                nn.Softplus()))

    def forward(self, x, d, t):
        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dt = torch.concat((d,t),axis = 1)
        
        x = self.activ(self.shared1(x))
        x = self.activ(self.shared2(x))
        x = self.activ(self.shared3(x))
        xdt = torch.cat((x,dt), dim = 1)

        alphas = []
        betas = []
        for alpha_i, beta_i in zip(self.alphas, self.betas):
            alphas.append( alpha_i(xdt) + self.jitter)
            betas.append( beta_i(xdt) + self.jitter)
        
        return alphas, betas
    
    def SoftPlus_W(self, y, gamma):
        #weight_scale = (1/b) * torch.log(1 + torch.exp(b * y))
        psi_function = torch.nn.Softplus(beta = gamma, threshold=20)
        weight_scale = psi_function(y)
        return weight_scale

    def weighted_loss(self, alphas, betas, y):
        weight_scale =  self.SoftPlus_W(y, self.gamma)
        alphas = torch.stack(alphas).swapaxes(0, 1)
        betas = torch.stack(betas).swapaxes(0, 1)
        y_true = y.clip(self.eps, 1 - self.eps)
        weighted_NLL = weight_scale * Beta(alphas,betas).log_prob(y_true)
        weighted_NLL = - weighted_NLL.mean()
        log_prior = self.log_prior(self, self.prior_var)
        beta_dist = Beta(torch.squeeze(alphas), torch.squeeze(betas))
        MSE = (beta_dist.mean - y_true.mean(axis =2)) ** 2
        EMSE = MSE.mean()
        return weighted_NLL, log_prior, EMSE
    
    def compute_loss(self, alphas, betas, y):
        
        alphas = torch.stack(alphas).swapaxes(0, 1)
        betas = torch.stack(betas).swapaxes(0, 1)
        y_true = y.clip(self.eps, 1 - self.eps)
        E_NLL =  - Beta(alphas,betas).log_prob(y_true).mean()
        log_prior = self.log_prior(self, self.prior_var)
        
        beta_dist = Beta(torch.squeeze(alphas), torch.squeeze(betas))
        MSE = (beta_dist.mean - y_true.mean(axis =2)) ** 2
        EMSE = MSE.mean()
        return E_NLL, log_prior, EMSE
    
    def log_prior(self, model, sigma):
        
        log_prior_total = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                prior_prob = (0.5/(sigma**2)) * param**2
                log_prior_total = log_prior_total + prior_prob.sum()
        return log_prior_total
    
    def training_step(self, batch, batch_nb):
        #print("batch", x.shape, y.shape)
        x, d, t, y = batch
        alphas, betas = self(x,d,t)
        
        if self.w_loss == 1:
            E_NLL, log_prior, EMSE = self.weighted_loss(alphas, betas,y)
        if self.w_loss == 0:
            E_NLL, log_prior, EMSE = self.compute_loss(alphas, betas,y)
            
        log_prior =  self.scaling_factor * log_prior
        loss = E_NLL + log_prior
        self.log("E_NLL_train",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("log_p_train",log_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_train",loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("EMSE_train",EMSE, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, d, t, y = batch
        alphas, betas = self(x,d,t)
        E_NLL, log_prior, EMSE = self.compute_loss(alphas, betas,y)
        log_prior =  self.scaling_factor * log_prior
        loss = E_NLL + log_prior
        self.log("E_NLL_val",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("log_p_val",log_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_val",loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("EMSE_val",EMSE, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'min')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'E_NLL_val'
        }
    
###################################################################
def Baseline_loss(y, alpha = 0.118 , beta = 13.145):
        eps = 1e-5
        y_true = y.clip(eps, 1 - eps)
        beta_dist = Beta(alpha, beta)
        MSE = (beta_dist.mean - y_true.mean(axis =2)) ** 2
        EMSE = MSE.mean()
        EMSE_task = MSE.mean(axis = 0)
        E_NLL = - beta_dist.log_prob(y_true).mean()
        E_NLL_task = - beta_dist.log_prob(y_true).mean(axis = [0,2])
        return EMSE, EMSE_task, E_NLL, E_NLL_task
    
##################################################################

class Beta_Severity_VI(LightningModule):

    def __init__(self, config):
        
        super().__init__()
        
        self.shared_layer_size = config['shared_layer_size']
        self.mixed_layer_size = config['mixed_layer_size']
        self.activation = config['activation']
        self.optim = config['optim']
        self.lambda_p = config['lambda']
        self.lr = config['lr']
        self.momentum = config['momentum']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.scaling_factor = 1/self.batch_size
        self.w_loss = config['weighted_loss']
        self.gamma = config['gamma']
        self.complexity_cost_weight = config['complexity_cost_weight']
        self.sample_nbr = config['sample_nbr']
        
        self.head_layer_size = self.mixed_layer_size + 2
        self.jitter = 1e-6
        self.eps = 1e-6
        
        self.shared1 = BayesianLinear(4096, self.shared_layer_size)
        self.shared2 = BayesianLinear(self.shared_layer_size, self.shared_layer_size)
        self.shared3 = BayesianLinear(self.shared_layer_size, self.mixed_layer_size)
        
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()    
        
        self.alphas = nn.ModuleList()
        self.betas = nn.ModuleList()
        for _ in range(14):
            self.alphas.append(nn.Sequential(
                BayesianLinear(self.head_layer_size, self.head_layer_size),
                self.activ,
                BayesianLinear(self.head_layer_size, 1),
                nn.Softplus()))
            
            self.betas.append(nn.Sequential(
                BayesianLinear(self.head_layer_size, self.head_layer_size),
                self.activ,
                BayesianLinear(self.head_layer_size, 1),
                nn.Softplus()))

    def forward(self, x, d, t):
        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dt = torch.concat((d,t),axis = 1)
        
        x = self.activ(self.shared1(x))
        x = self.activ(self.shared2(x))
        x = self.activ(self.shared3(x))
        xdt = torch.cat((x,dt), dim = 1)

        alphas = []
        betas = []
        for alpha_i, beta_i in zip(self.alphas, self.betas):
            alphas.append( alpha_i(xdt) + self.jitter)
            betas.append( beta_i(xdt) + self.jitter)
        
        return alphas, betas
    
    def SoftPlus_W(self, y, gamma):
        #weight_scale = (1/b) * torch.log(1 + torch.exp(b * y))
        psi_function = torch.nn.Softplus(beta = gamma, threshold=20)
        weight_scale = psi_function(y)
        return weight_scale

    def weighted_loss(self, alphas, betas, y):
        weight_scale =  self.SoftPlus_W(y, self.gamma)
        alphas = torch.stack(alphas).swapaxes(0, 1)
        betas = torch.stack(betas).swapaxes(0, 1)
        y_true = y.clip(self.eps, 1 - self.eps)
        weighted_NLL = weight_scale * Beta(alphas,betas).log_prob(y_true)
        weighted_NLL = - weighted_NLL.mean()
        beta_dist = Beta(torch.squeeze(alphas), torch.squeeze(betas))
        MSE = (beta_dist.mean - y_true.mean(axis =2)) ** 2
        EMSE = MSE.mean()
        return weighted_NLL, EMSE
    
    def compute_loss(self, alphas, betas, y):
        alphas = torch.stack(alphas).swapaxes(0, 1)
        betas = torch.stack(betas).swapaxes(0, 1)
        y_true = y.clip(self.eps, 1 - self.eps)
        NLL =  - Beta(alphas,betas).log_prob(y_true).mean()
        beta_dist = Beta(torch.squeeze(alphas), torch.squeeze(betas))
        MSE = (beta_dist.mean - y_true.mean(axis =2)) ** 2
        MSE = MSE.mean()
        return NLL, MSE

    def nn_kl_divergence(self):
        return kl_divergence_from_nn(self)
    
    def sample_elbo(self, x,d,t,y): 
        E_NLL, E_MSE = 0,0
        for _ in range(self.sample_nbr):
            alphas, betas = self(x,d,t)
            if self.w_loss == 0:
                NLL, MSE =  self.compute_loss(alphas, betas,y)
            if self.w_loss == 1:
                NLL, MSE =  self.weighted_loss(alphas, betas,y)
            E_NLL += NLL
            E_MSE += MSE

        E_NLL = E_NLL / self.sample_nbr
        E_MSE = E_MSE / self.sample_nbr
        KL = self.nn_kl_divergence() * self.complexity_cost_weight
        return E_NLL,E_MSE, KL
        
    def training_step(self, batch, batch_nb):
        #print("batch", x.shape, y.shape)
        x, d, t, y = batch
        E_NLL,E_MSE, KL = self.sample_elbo(x,d,t,y)
        loss = E_NLL + KL
        self.log("train_E_NLL",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_KL",KL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss",loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_EMSE",E_MSE, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, d, t, y = batch
        alphas, betas = self(x,d,t)
        E_NLL, EMSE = self.compute_loss(alphas, betas,y)
        
        self.log("val_E_NLL",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_EMSE",EMSE, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'min')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_E_NLL'
        }

#################################################
# Multitask ndt beta severity model
#################################################
class Beta_Severity_CV(LightningModule):

    def __init__(self, config):
        
        super().__init__()
        self.input_feature_dim = config['input_feature_dim']
        self.activation = config['activation']
        self.depth = config['depth']
        self.dropout_p = config['dropout_p']
        self.optim = config['optim']
        self.lr = config['lr']
        self.prior_var = config['prior_var']
        self.batch_size = config['batch_size']
        self.weighted_loss = config['weighted_loss']
        self.gamma = config['gamma']
    
        if self.activation == 'relu':
            self.activ = torch.nn.ReLU()
            
        if self.activation == 'tanh':
            self.activ = torch.nn.Tanh()  
         
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.arch = nn.ModuleList()
        
        # hidden units, size depends upon number of layers
        N_0 = 4096
        lambd = 1
        for layer_number in range(self.depth):
            
            if layer_number == 0:
                layer_input_dim = 4098
            else:
                layer_input_dim = layer_output_dim
                
            layer_output_dim = int(N_0*(0.5)**(lambd * (layer_number + 1)))
            self.arch.append(nn.Sequential(
                nn.Linear(layer_input_dim, layer_output_dim),
                self.dropout,
                self.activ
            ))
        
        # alpha and beta parameters, with softplus link function
        self.arch.append(nn.Sequential(
                nn.Linear(layer_output_dim, 32),
                nn.Softplus()))

    def forward(self, x, d, t):
        
        d = torch.reshape(d,(-1,1))
        t = torch.reshape(t,(-1,1))
        dt = torch.concat((d,t),axis = 1)
        x = torch.cat((x,dt), dim = 1)

        for layers in self.arch:
            x = layers(x)
        return x
    
    def SoftPlus_W(self, y, gamma):
        psi_function = torch.nn.Softplus(beta = gamma, threshold=20)
        weight_scale = psi_function(y)
        return weight_scale
    
    def compute_loss(self, alphas, betas, y):
        self.eps = 1e-5
        y_true = y.clip(self.eps, 1 - self.eps)
        
        # normailized by number of examples
        E_NLL =  - Beta(torch.unsqueeze(alphas, dim = 2),torch.unsqueeze(betas, dim = 2)).log_prob(y_true)
        
        if self.weighted_loss == 1:
            weight_scale =  self.SoftPlus_W(y, self.gamma)
            E_NLL = weight_scale * E_NLL
        E_NLL = E_NLL.mean()
        log_prior = self.log_prior(self, self.prior_var)
        beta_dist = Beta(alphas, betas)
        MSE = (beta_dist.mean - y_true.mean(axis =2)) ** 2
        EMSE = MSE.mean()
        return E_NLL, log_prior, EMSE
    
    def log_prior(self, model, sigma):
        device = torch.device('cuda')
        norma_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([sigma], device=device))
        num_param = torch.Tensor([0]).to(device)
        log_prior_total = torch.tensor(0., requires_grad=True).to(device)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                prior_prob = norma_dist.log_prob(param)
                log_prior_total = log_prior_total + prior_prob.sum()
                num_param += len(param.view(-1))
        log_prior_total = log_prior_total/num_param
        return log_prior_total
    
    def training_step(self, batch, batch_nb):
        #print("batch", x.shape, y.shape)
        x, d, t, y = batch
        alphas_betas = self(x,d,t)
        alphas, betas = alphas_betas[:,0:15],  alphas_betas[:,15:]
        E_NLL, log_prior, EMSE = self.compute_loss(alphas, betas,y)
   
        loss = E_NLL - log_prior
        #print(E_NLL, log_prior)
        self.log("E_NLL_train",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("log_p_train",log_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_train",loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("EMSE_train",EMSE, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, d, t, y = batch
        alphas_betas = self(x,d,t)
        alphas, betas = alphas_betas[:,0:15],  alphas_betas[:,15:]
        E_NLL, log_prior, EMSE = self.compute_loss(alphas, betas,y)

        loss = E_NLL - log_prior
        self.log("E_NLL_val",E_NLL, prog_bar=True, on_step=False, on_epoch=True)
        self.log("log_p_val",log_prior, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_val",loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("EMSE_val",EMSE, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr,
                                             weight_decay = self.lambda_p,
                                             momentum = self.momentum)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=False,
                                                                    patience=10,
                                                                    min_lr=1e-6,
                                                                    mode = 'min')
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'E_NLL_val'
        }