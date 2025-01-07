#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import gc


# In[2]:


import os, yaml
from argparse import Namespace

import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, auc, roc_curve

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything


import wandb
os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")


# In[4]:


from molbert.models.finetune import FinetuneSmilesMolbertModel
from molbert.datasets.dataloading import MolbertDataLoader
from molbert.datasets.finetune import BertFinetuneSmilesDataset_MF
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer


# In[6]:


    
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
                    
        if self.hparams.freeze_level == "complete_BERT":
            for param in self.encoder.model.bert.parameters():
                param.requires_grad = False

        # Task embeddings and biases
        self.task_embedding = nn.Embedding(self.hparams.num_of_tasks, self.hparams.embedding_size)
        self.mol_bias = nn.Embedding(self.hparams.num_mols, 1)
        self.task_bias = nn.Embedding(self.hparams.num_of_tasks, 1)
        
    def forward(self, batch_inputs , mol_indices, task_indices):

        mol_embeddings = self.encoder(batch_inputs)
        mol_embeddings = mol_embeddings["finetune"]
        task_embeddings = self.task_embedding(task_indices)
        mol_bias = self.mol_bias(mol_indices) # batch*1
        task_bias = self.task_bias(task_indices) # batch * num_tasks * 1
        biases_sum = mol_bias.unsqueeze(1) + task_bias # batch * num_tasks * 1
        
        dot_product = torch.sum(mol_embeddings.unsqueeze(1) * task_embeddings, dim=2) # [batch_size, num_tasks]
        logits = dot_product + biases_sum.squeeze(2)   #[batch_size, num_tasks]
        return logits, [mol_embeddings, task_embeddings, mol_bias, task_bias]
    
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

    def configure_optimizers(self):
        if self.hparams.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.hparams.lr)
        if self.hparams.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              #weight_decay = self.l2_lambda,
                                             lr=self.hparams.lr)
        
        if self.hparams.lr_schedulers == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        T_max = 10, 
                                                                        eta_min=1e-6) 
            return {"optimizer": self.optimizer, 
                    "lr_scheduler": self.scheduler}
        
        if self.hparams.lr_schedulers == "ReduceLROnPlateau":
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
    
    def compute_regularization(self):
        encoder_reg = torch.tensor(0., requires_grad=True, device= self.hparams.device)
        task_emb_reg = torch.tensor(0., requires_grad=True, device=self.hparams.device)

        # l2: Apply only on weights, exclude bias
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                encoder_reg = encoder_reg + torch.norm(param, p=2)

        # l1: Apply only on weights, exclude bias
        for name, param in self.task_embedding.named_parameters():
            if 'weight' in name:
                task_emb_reg = task_emb_reg + torch.norm(param, p=1)
                
        return encoder_reg, task_emb_reg
    
    def label_smoothing(self, y_true, num_classes = 2):
        y_ls = y_true * (1 - self.hparams.epsilon) + self.hparams.epsilon / num_classes
        return y_ls

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

        if self.hparams.loss_type == 'Label_smoothing':
            weighted_loss = self.weighted_creterien(y_hat, self.label_smoothing(y)) * valid_label_mask
            Non_weighted_loss = self.non_weighted_creterian(y_hat, self.label_smoothing(y)) * valid_label_mask

        if self.hparams.loss_type == 'Focal_loss':
            Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask
            weighted_loss = sigmoid_focal_loss(inputs = y_hat,
                                                targets = y,
                                                BCE_loss = Non_weighted_loss,
                                                alpha = self.hparams.alpha,
                                                gamma = self.hparams.gamma)
            weighted_loss = weighted_loss * valid_label_mask
            
        if self.hparams.loss_type == 'BCE':
            weighted_loss = self.weighted_creterien(y_hat, y) * valid_label_mask
            Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask
        
        # Non_weighted_loss, positive negative loss
        pos_loss = Non_weighted_loss * pos_label_mask
        neg_loss = Non_weighted_loss * negative_label_mask
        pos_loss = pos_loss.sum() / pos_label_mask.sum()
        neg_loss = neg_loss.sum() / negative_label_mask.sum()
    
        # compute mean loss
        Non_weighted_loss = Non_weighted_loss.sum() / valid_label_mask.sum()
        weighted_loss = weighted_loss.sum() / valid_label_mask.sum()

        encoder_reg, task_emb_reg = self.compute_regularization()
        encoder_reg = self.hparams.l2_lambda*encoder_reg
        
        task_emb_reg = self.hparams.l1_lambda*task_emb_reg
        total_reg = encoder_reg + task_emb_reg

        total_loss = weighted_loss + total_reg

        return total_loss, weighted_loss, Non_weighted_loss,total_reg, pos_loss, neg_loss
    
        
    def training_step(self, batch, batch_idx):
        # compute forward pass
        (batch_inputs, batch_labels), _ = batch

        y = batch_labels["finetune"].squeeze()
        mol_indices = batch_labels["mol_indices"]
        task_indices = batch_labels["task_indices"]

        y_hat, _ = self.forward(batch_inputs,
                                        mol_indices,
                                        task_indices
                                        )

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.training_step_ytrue.append(y.long().cpu())
        self.training_step_ypred.append(torch.sigmoid(y_hat).cpu())

        return {"loss": total_loss,
                "weighted_loss":weighted_loss,
                "Non_weighted_loss":Non_weighted_loss,
                "l2_reg_loss":l2_reg_loss, 
                "pos_loss":pos_loss, 
                "neg_loss":neg_loss
                }
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        (batch_inputs, batch_labels), _ = batch

        y = batch_labels["finetune"].squeeze()
        mol_indices = batch_labels["mol_indices"]
        task_indices = batch_labels["task_indices"]

        y_hat, _ = self.forward(batch_inputs,
                                mol_indices,
                                task_indices
                                )

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.val_step_ytrue.append(y.long().cpu())
        self.val_step_ypred.append(torch.sigmoid(y_hat).cpu())

        return {"loss": total_loss,
                "weighted_loss":weighted_loss,
                "Non_weighted_loss":Non_weighted_loss,
                "l2_reg_loss":l2_reg_loss, 
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
        avg_l2_reg_loss = torch.stack([x['l2_reg_loss'] for x in outputs]).mean()
        avg_pos_loss = torch.stack([x['pos_loss'] for x in outputs]).mean()
        avg_neg_loss = torch.stack([x['neg_loss'] for x in outputs]).mean()
        tensorboard_logs = {
                    'train_total_loss': avg_loss,
                    'train_weighted_loss': avg_weighted_loss,
                    'train_Non_weighted_loss': avg_non_weighted_loss,
                    'train_l2_reg_loss': avg_l2_reg_loss,
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
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR']
            
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
        avg_l2_reg_loss = torch.stack([x['l2_reg_loss'] for x in outputs]).mean()

        avg_pos_loss = torch.stack([x['pos_loss'] for x in outputs])
        avg_pos_loss = avg_pos_loss[~torch.isnan(avg_pos_loss)]
        avg_pos_loss = avg_pos_loss.mean()
        print("avg_pos_loss",avg_pos_loss)

        avg_neg_loss = torch.stack([x['neg_loss'] for x in outputs]).mean()
        tensorboard_logs = {
                    'val_total_loss': avg_loss,
                    'val_weighted_loss': avg_weighted_loss,
                    'val_Non_weighted_loss': avg_non_weighted_loss,
                    'val_l2_reg_loss': avg_l2_reg_loss,
                    'val_pos_loss': avg_pos_loss,
                    'val_neg_loss': avg_neg_loss
                    }
        wandb.log(tensorboard_logs)

        #Collect predictions and true labels for the complete training set
        val_true = torch.cat(self.val_step_ytrue, dim=0)
        val_preds = torch.cat(self.val_step_ypred, dim=0)

        score_list =  self.compute_metrics(val_true,val_preds)
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR']
            
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

        roc_score, blc_acc, sensitivity, specificity, AUPR, f1_score, average_precision = [],[],[],[],[],[],[]
        for i in range(self.hparams.num_of_tasks):
                
                # get valid targets, and convert logits to prob
                valid_targets = targets[:,i][mask[:,i]]
                valid_preds = expit(preds[:,i][mask[:,i]])
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
                    
                except:
                    roc_score.append(np.nan)
                    #print('Performance metric is null')
                
        self.train()
        return np.nanmean(roc_score), np.nanmean(blc_acc), np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(AUPR)

    
    def prob_to_labels(self, pred, threshold):
	    return (pred >= threshold).astype('int')

    def unfreeze_model(self):
        for param in self.encoder.model.bert.parameters():
            param.requires_grad = True


# In[7]:


# config_dict
model_weights_dir = '/projects/home/mmasood1/Model_weights/preclinical_clinical/BERT/'
pretrained_model_path = '/projects/home/mmasood1/TG GATE/MolBERT/molbert/molbert_100epochs/molbert_100epochs/checkpoints/last.ckpt'
data_dir = '/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/Data_for_BERT_finetuning/'
pos_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/06_10_2023/pos_weights.csv"
class_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/06_10_2023/target_weights.csv"
metadata_dir = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/BERT_finetune/MF/"
model_dir = os.path.dirname(os.path.dirname(pretrained_model_path))
hparams_path = os.path.join(model_dir, 'hparams.yaml')

# load config
with open(hparams_path) as yaml_file:
    config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

config_dict['project_name'] = "BERT_finetuning_MF"
config_dict['model_name'] = None

config_dict['model_weights_dir'] = model_weights_dir
config_dict['pretrained_model_path'] = pretrained_model_path
config_dict["metadata_dir"] = metadata_dir
config_dict['pos_weights'] = pos_weights
config_dict['class_weights'] = class_weights

config_dict['data_dir'] = data_dir
config_dict['train_file'] = data_dir + "complete_training_set.csv"
config_dict['valid_file'] = data_dir + "complete_test_set.csv"
config_dict['test_file'] = data_dir + "complete_test_set.csv"

config_dict['mode'] = 'classification'
config_dict['loss_type'] = 'BCE'

config_dict['alpha'] = 0.0
config_dict['beta'] = 0.0
config_dict['gamma'] = 0.0
config_dict['epsilon'] = 0.0

config_dict['epochs'] = 310
config_dict['unfreeze_epoch'] = 210
config_dict["l2_lambda"] = 1e-4
config_dict["l1_lambda"] = 1e-05
config_dict['embedding_size'] = 50
config_dict["freeze_level"] = "complete_BERT"


config_dict['optim'] = 'Adam'#SGD
config_dict['lr_schedulers'] = "CosineAnnealingLR"
config_dict['lr'] = 1e-3
config_dict["BERT_lr"] = 3e-5
config_dict["batch_size"] = 64
config_dict['seed'] = 42



config_dict['missing'] = 'nan'
config_dict['compute_metric_after_n_epochs'] = 5
config_dict['return_trainer'] = True
config_dict['EarlyStopping'] = False

config_dict["accelerator"] = "gpu"
config_dict["gpu"] =  [0]
config_dict["device"] = torch.device("cuda") #torch.device("cuda:1") #


data = pd.read_csv(config_dict['train_file'])
try:
    data.drop(['Scafold','fold'], axis = 1, inplace = True)
except:
    pass
target_names = data.loc[:,"Cytoplasmic alteration (Basophilic/glycogen depletion)":"hepatobiliary_disorders"].columns.tolist()

#target_names = data.loc[:,"DILI_binary":"hepatobiliary_disorders"].columns.tolist()
config_dict["output_size"] = len(target_names)
config_dict["label_column"] = target_names
config_dict["invitro_tasks"] = 0

config_dict["num_of_tasks"] = len(target_names)
config_dict["selected_tasks"] = target_names
config_dict['num_mols'] = data.shape[0]


# In[8]:

def get_dataloaders(config_dict):
    # Dataloader
    featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config_dict["max_seq_length"], permute = False)
    train_dataset = BertFinetuneSmilesDataset_MF(
                input_path= config_dict['train_file'],
                featurizer=featurizer,
                single_seq_len=config_dict["max_seq_length"],
                total_seq_len=config_dict["max_seq_length"],
                label_column=config_dict["label_column"],
                is_same=False,
                inference_mode=True,
            )

    validation_dataset = BertFinetuneSmilesDataset_MF(
                input_path= config_dict['valid_file'],
                featurizer=featurizer,
                single_seq_len=config_dict["max_seq_length"],
                total_seq_len=config_dict["max_seq_length"],
                label_column=config_dict["label_column"],
                is_same=False,
                inference_mode=True,
            )

    test_dataset = BertFinetuneSmilesDataset_MF(
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
                                        shuffle = True)

    validation_dataloader = MolbertDataLoader(validation_dataset, 
                                        batch_size=config_dict["batch_size"],
                                        pin_memory=False,
                                        num_workers=4, 
                                        shuffle = False)

    test_dataloader = MolbertDataLoader(test_dataset, 
                                        batch_size=config_dict["batch_size"],
                                        pin_memory=False,
                                        num_workers=4, 
                                        shuffle = False)

    config_dict["num_batches"] = len(train_dataloader)

    return train_dataloader, validation_dataloader, test_dataloader
# In[10]:


def wandb_init_model(model, 
                     config, 
                     train_dataloader,
                     val_dataloader, 
                     model_type):
    
    default_root_dir = config["model_weights_dir"]
    max_epochs = config["epochs"]
    return_trainer = config["return_trainer"]

    '''
    run = wandb.init(
                        project= config["project_name"],
                        dir = '/projects/home/mmasood1/Model_weights',
                        entity="arslan_masood", 
                        reinit = True, 
                        config = config,
                        name = config["model_name"],
                        settings=wandb.Settings(start_method="fork"))
    '''
    # logger
    model = model(config)
    wandb_logger = WandbLogger(project= config["project_name"],
                            save_dir = '/projects/home/mmasood1/Model_weights',
                            name = config["model_name"],
                            log_model = False,
                            offline = False)
    # trainer
    trainer = Trainer(
        max_epochs= int(max_epochs),
        gpus = config_dict["gpu"],
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


# In[12]:
np.random.seed(config_dict["seed"])
seed_everything(config_dict["seed"])

alpha_list = [1.0]
beta_list = [1.0]
#gamma_list = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
gamma_list = [0.0]

epsilon_list = [0.0]

l1_lambda_list = [1e-5]
l2_lambda_list = [1e-4] 

for alpha in alpha_list:
    config_dict["alpha"] = alpha

    for beta in beta_list:
        config_dict["beta"] = beta

        for gamma in gamma_list:
            config_dict["gamma"] = gamma

            for epsilon in epsilon_list:
                config_dict["epsilon"] = epsilon

                for l2_lambda in l2_lambda_list:
                    config_dict["l2_lambda"] = l2_lambda

                    for l1_lambda in l1_lambda_list:
                        config_dict["l1_lambda"] = l1_lambda

                        config_dict["model_name"] = rf's{config_dict["seed"]}_alpha_{config_dict["alpha"]}_beta_{config_dict["beta"]}_gamma_{config_dict["gamma"]}_epsilon_{config_dict["epsilon"]}_λ1_{config_dict["l1_lambda"]}_λ2_{config_dict["l2_lambda"]}_E_{config_dict["embedding_size"]}'
                        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(config_dict)
                        trained_model, trainer = wandb_init_model(model = MolbertModel, 
                                                                                        train_dataloader = train_dataloader,
                                                                                        val_dataloader =validation_dataloader,
                                                                                        config = config_dict, 
                                                                                        model_type = 'MLP')

                        # In[13]:

                        data_dir = config_dict["metadata_dir"] + "predicitons/"
                        result_dir = config_dict["metadata_dir"] + "Results/"  
                        if not os.path.exists(data_dir):
                            os.makedirs(data_dir)
                            os.makedirs(result_dir)


                        # In[15]:

                        model = trained_model.eval()
                        config = config_dict
                        model = model.cpu() 

                        y_true_list = []
                        y_pred_list = []

                        for batch in validation_dataloader:
                            
                            (batch_inputs, batch_labels), _ = batch
                            y = batch_labels["finetune"].squeeze()
                            mol_indices = batch_labels["mol_indices"].cpu()
                            task_indices = batch_labels["task_indices"].cpu()

                            y_hat, embeddings = model(batch_inputs,mol_indices,task_indices)

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

                        y.to_csv(data_dir + f'y_true_test_{config_dict["model_name"]}.csv',index=False)
                        y_hat.to_csv(data_dir + f'y_pred_test_{config_dict["model_name"]}.csv',index=False)


                        #####################################################################################3
                        # Compute compute_binary_classification_metrics: Multitask
                        ######################################################################################
                        from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
                        from sklearn.metrics import average_precision_score, f1_score

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


                        # In[17]:


                        config = {}
                        data = pd.read_csv(config_dict["data_dir"] + "train_fold0.csv")
                        target_names = data.loc[:,"Cytoplasmic alteration (Basophilic/glycogen depletion)":"hepatobiliary_disorders"].columns.tolist()
                        #target_names = data.loc[:,"DILI_binary":"hepatobiliary_disorders"].columns.tolist()
                        config["num_of_tasks"] = len(target_names)
                        config["selected_tasks"] = target_names

                        preclinical_tasks = config["selected_tasks"][:20]
                        clinical_tasks = config["selected_tasks"][20:]

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


                        # In[19]:


                        metrics = compute_binary_classification_metrics_MT(y_true = y[config['selected_tasks']].values, 
                                                                            y_pred_proba = y_hat[config['selected_tasks']].values,
                                                                                                missing = 'nan')
                        metrics.insert(0, 'Tasks', target_names)
                        mean_preformances = {"pathology_mean": metrics[metrics.Tasks.isin(pathological_tasks)].iloc[:,1:].mean(),
                                            "blood_mean": metrics[metrics.Tasks.isin(blood_tasks)].iloc[:,1:].mean(),
                                            "preclinical_mean": metrics[metrics.Tasks.isin(preclinical_tasks)].iloc[:,1:].mean(),
                                            "clinical_mean": metrics[metrics.Tasks.isin(clinical_tasks)].iloc[:,1:].mean(),
                                            "combined_ex_BM":metrics[metrics.Tasks.isin(clinical_tasks + pathological_tasks)].iloc[:,1:].mean(),
                                            "combined_all": metrics.iloc[:,1:].mean()}
                        mean_preformances = pd.DataFrame(mean_preformances).T
                        mean_preformances = mean_preformances.rename_axis('Tasks').reset_index()
                        metrics = pd.concat([metrics, mean_preformances], ignore_index=True) 
                        metrics.to_csv(result_dir + f'val_metric_{config_dict["model_name"]}.csv', index=False)
                        metrics.tail(5)
                        
                        # delete all, also clear gpu memory
                        wandb.finish()
                        del train_dataloader, validation_dataloader, test_dataloader, trained_model, trainer
                        torch.cuda.empty_cache()
                        gc.collect()

                        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
                        gpu_memory_status = torch.cuda.memory_allocated() / (1024 ** 3)
                        print("GPU Memory Status (after clearing):", gpu_memory_status)
                        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    # In[ ]:




