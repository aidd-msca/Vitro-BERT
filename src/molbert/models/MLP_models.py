# Third-party imports
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
import pytorch_lightning as pl
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    auc,
    roc_curve
)

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
        self.w_p = self.w_p.to(y_pred.device)

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
        self.epochs = config["max_epochs"]
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
        self.pos_weights = torch.tensor(pos_weights, device = self.device)

        # class weights
        if config['num_of_tasks'] > 1 and config["beta"] > 0:
            class_weights = pd.read_csv(config["class_weights"])
            class_weights = class_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
            class_weights = (config["beta"] * class_weights) + (1 - config["beta"])*1
            self.class_weights = torch.tensor(class_weights, device = self.device)
        else:
            self.class_weights = torch.tensor([1.0], device = self.device)

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
        l2_reg = torch.tensor(0., requires_grad=True, device=self.device)
        
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
            nan_mask = torch.isnan(y)
            y[nan_mask] = -1
        
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
        total_loss, weighted_loss, Non_weighted_loss, weight_norm, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.train_step_pos_loss.append(pos_loss.item())
        self.train_step_neg_loss.append(neg_loss.item())

        # Return dictionary of losses
        return {
            "loss": total_loss,
            "weighted_loss": weighted_loss,
            "non_weighted_loss": Non_weighted_loss,
            "weight_norm": weight_norm,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss
        }

    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss, weight_norm, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.val_step_pos_loss.append(pos_loss.item())
        self.val_step_neg_loss.append(neg_loss.item())

        # Return dictionary of losses
        return {
            "loss": total_loss,
            "weighted_loss": weighted_loss,
            "non_weighted_loss": Non_weighted_loss,
            "weight_norm": weight_norm,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss
        }

    def training_step_end(self, outputs):
        # Define the step prefix
        step_prefix = "train_"
        
        # Calculate mean losses
        losses = {key: outputs[key].mean() for key in outputs.keys()}
        
        # Log the losses with WandB only on main process
        if self.trainer.is_global_zero:
            log_dict = {f'{step_prefix}{key}_step': value.item() for key, value in losses.items()}
            log_dict["global_step"] = self.trainer.global_step
            self.logger.experiment.log(log_dict)
        
        return losses

    def validation_step_end(self, outputs):
        # Define the step prefix
        step_prefix = "val_"
        
        # Calculate mean losses
        losses = {key: outputs[key].mean() for key in outputs.keys()}
        
        # Log the losses with WandB only on main process
        if self.trainer.is_global_zero:
            log_dict = {f'{step_prefix}{key}_step': value.item() for key, value in losses.items()}
            log_dict["global_step"] = self.trainer.global_step
            self.logger.experiment.log(log_dict)
        
        return losses

    def training_epoch_end(self, outputs):
        pos_loss = torch.tensor(self.train_step_pos_loss)
        neg_loss = torch.tensor(self.train_step_neg_loss)
        pos_loss_mean = pos_loss[~torch.isnan(pos_loss)].mean()
        neg_loss_mean = neg_loss[~torch.isnan(neg_loss)].mean()
        geometric_mean = torch.sqrt(pos_loss_mean * neg_loss_mean)

        log_dict = {
            'train_BCE_pos_epoch': pos_loss_mean.item(),
            'train_BCE_neg_epoch': neg_loss_mean.item(),
            'train_gm_loss_epoch': geometric_mean.item(),
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
            'current_epoch': self.current_epoch,
            'global_step': self.trainer.global_step
        }

        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list = self.compute_metrics(self.trainer.train_dataloader)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision', 'ECE_score', 'ACE_score']
            
            for i, score in enumerate(score_list):
                log_dict[f'train_{metric[i]}_epoch'] = score

        self.logger.experiment.log(log_dict)
        self.train_step_pos_loss.clear()
        self.train_step_neg_loss.clear()
        return log_dict

    def validation_epoch_end(self, outputs):
        pos_loss = torch.tensor(self.val_step_pos_loss)
        neg_loss = torch.tensor(self.val_step_neg_loss)
        pos_loss_mean = pos_loss[~torch.isnan(pos_loss)].mean()
        neg_loss_mean = neg_loss[~torch.isnan(neg_loss)].mean()
        geometric_mean = torch.sqrt(pos_loss_mean * neg_loss_mean)

        log_dict = {
            'val_BCE_pos_epoch': pos_loss_mean.item(),
            'val_BCE_neg_epoch': neg_loss_mean.item(),
            'val_gm_loss_epoch': geometric_mean.item(),
            'current_epoch': self.current_epoch,
            'global_step': self.trainer.global_step
        }

        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list = self.compute_metrics(self.trainer.val_dataloaders[0])
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision', 'ECE_score', 'ACE_score']
            
            for i, score in enumerate(score_list):
                log_dict[f'val_{metric[i]}_epoch'] = score

        self.logger.experiment.log(log_dict)

        self.val_step_pos_loss.clear()
        self.val_step_neg_loss.clear()
        return log_dict
           
    def compute_metrics(self, dataloader): 
        self.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch_x,batch_targets = batch
                batch_preds = self(batch_x.to(self.device))

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

    
    def compute_metrics_external_val(self, dataloader, config): 
        self.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch_x,batch_targets = batch
                batch_preds = self(batch_x.to(self.device))

                preds.extend(batch_preds.cpu().detach().tolist())
                targets.extend(batch_targets.cpu().detach().tolist())

            # select only relevent preddictions
            targets = np.array(targets).reshape(-1,config["num_of_tasks_ex_val"])
            preds = pd.DataFrame(preds)
            preds.columns = config["selected_tasks"]
            preds.columns = preds.columns.str.replace(r"\(.*\)", "", regex=True).str.strip()
            preds = preds[config["selected_tasks_ex_val"]].values
            

            if self.missing == 'nan':
               mask = ~np.isnan(targets)

            roc_score, blc_acc, sensitivity, specificity, AUPR, f1, average_precision = [],[],[],[],[],[],[]
            ECE_score, ACE_score = [],[]

            n_bins = 10

            for i in range(config["num_of_tasks_ex_val"]):
                
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
    
############################################
# ECE and ACE
############################################
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


