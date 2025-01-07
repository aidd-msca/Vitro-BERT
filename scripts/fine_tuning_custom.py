#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, yaml
from typing import Dict, Tuple, List
from argparse import Namespace

import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,precision_recall_curve, auc, roc_curve

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR


# In[2]:


from transformers import BertModel
from transformers.modeling_bert import BertEncoder, BertPooler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)


# In[3]:


from molbert.models.base import SuperPositionalBertEmbeddings
from molbert.utils.lm_utils import BertConfigExtras
from molbert.tasks.tasks import BaseTask, FinetuneTask

from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.datasets.finetune import BertFinetuneSmilesDataset
from molbert.datasets.dataloading import MolbertDataLoader


# In[4]:


MolbertBatchType = Tuple[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]
class SuperPositionalBertModel(BertModel):
    """
    Same as BertModel, BUT
    uses SuperPositionalBertEmbeddings instead of BertEmbeddings
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = SuperPositionalBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()


# In[5]:


class FinetuneHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.finetune_net = nn.Sequential(nn.Linear(config.hidden_size, config.output_size))

    def forward(self, pooled_output):
        return self.finetune_net(pooled_output)
    
class MolbertModel(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        
        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]

        self.hparams = args
        self.config = self.get_config()
        #self.tasks = self.get_tasks(self.config)
        #print(self.tasks)

        self.get_creterian(args)
        self.bert = SuperPositionalBertModel(self.config)
        self.bert.init_weights()
        self.bert = self.load_model_weights(
                                model=self.bert, 
                                checkpoint_file=self.hparams.pretrained_model_path)

        self.head = FinetuneHead(self.config)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output = self.bert(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = self.head(pooled_output)
        return logits
        #return {task.name: task(sequence_output, pooled_output) for task in self.tasks}
    
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
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = self._initialise_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def _initialise_lr_scheduler(self, optimizer):

        #num_batches = len(self.trainer.train_dataloader) // self.hparams.batch_size
        num_batches = self.hparams.num_batches
        num_training_steps = num_batches // self.hparams.accumulate_grad_batches * self.hparams.max_epochs
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

        return {'scheduler': scheduler, 'monitor': 'valid_loss', 'interval': 'step', 'frequency': 1}
    
    def l2_regularization(self):
        device = torch.device('cuda')
        l2_reg = torch.tensor(0., requires_grad=True, device=device)

        # Apply only on weights, exclude bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
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

        l2_reg_loss = self.l2_regularization()
        l2_reg_loss = self.hparams.l2_lambda*l2_reg_loss
        total_loss = weighted_loss + l2_reg_loss

        return total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss
    
    def training_step(self, batch, batch_idx):
        # compute forward pass
        (batch_inputs, batch_labels), _ = batch
        y = batch_labels["finetune"].squeeze()
        y_hat = self.forward(**batch_inputs)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  

        wandb.log({'train_total_loss': total_loss.item()})
        wandb.log({'train_weighted_loss': weighted_loss.item()})
        wandb.log({'train_Non_weighted_loss': Non_weighted_loss.item()})
        wandb.log({'train_l2_reg_loss': l2_reg_loss.item()})
        wandb.log({'train_pos_loss': pos_loss.item()})
        wandb.log({'train_neg_loss': neg_loss.item()})

        self.training_step_ytrue.append(y.long().cpu())
        self.training_step_ypred.append(torch.sigmoid(y_hat).cpu())

        return {"loss": total_loss}
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        (batch_inputs, batch_labels), _ = batch
        y = batch_labels["finetune"].squeeze()
        y_hat = self.forward(**batch_inputs)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        wandb.log({'val_total_loss': total_loss.item()})
        wandb.log({'val_weighted_loss': weighted_loss.item()})
        wandb.log({'val_Non_weighted_loss': Non_weighted_loss.item()})
        wandb.log({'val_l2_reg_loss': l2_reg_loss.item()})
        wandb.log({'val_pos_loss': pos_loss.item()})
        wandb.log({'val_neg_loss': neg_loss.item()})
        #return {"loss": total_loss}
        self.val_step_ytrue.append(y.long().cpu())
        self.val_step_ypred.append(torch.sigmoid(y_hat).cpu())

    
    def on_epoch_end(self):
        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        wandb.log({'learning_rate': lr})
        
        # Collect predictions and true labels for the complete training set
        train_true = torch.cat(self.training_step_ytrue, dim=0)
        train_preds = torch.cat(self.training_step_ypred, dim=0)

        score_list =  self.compute_metrics(train_true, train_preds)
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision']
            
        for i, score in enumerate(score_list):
                wandb.log({f'train_{metric[i]}':score.item()})
        
        # Clear the lists to free memory for the next epoch
        self.training_step_ytrue.clear()
        self.training_step_ypred.clear()
        del train_true,train_preds

    def validation_epoch_end(self, outputs):
        #Collect predictions and true labels for the complete training set
        val_true = torch.cat(self.val_step_ytrue, dim=0)
        val_preds = torch.cat(self.val_step_ypred, dim=0)

        score_list =  self.compute_metrics(val_true,val_preds)
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision']
            
        for i, score in enumerate(score_list):
            wandb.log({f'val_{metric[i]}':score.item()})

        # Clear the lists to free memory for the next epoch
        self.val_step_ytrue.clear()
        self.val_step_ypred.clear()
        del val_true, val_preds

    def compute_metrics(self, y_true, y_pred): 
        device = torch.device("cuda") 
        self.eval()

        targets =  y_true.cpu().detach().tolist()
        preds = y_pred.cpu().detach().tolist()

        targets = np.array(targets).reshape(-1,self.hparams.num_of_tasks)
        preds = np.array(preds).reshape(-1,self.hparams.num_of_tasks)

        if self.hparams.missing == 'nan':
            mask = ~np.isnan(targets)

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
                    f1 = [f1_score(valid_targets, self.prob_to_labels(valid_preds, t)) for t in self.thresholds]
                    f1_score.append(np.nanmax(f1))
                    average_precision.append(average_precision_score(valid_targets, valid_preds))
                    
                except:
                    roc_score.append(np.nan)
                    AUPR.append(np.nan)
                    average_precision.append(np.nan)
                    #print('Performance metric is null')
                
        self.train()
        return np.nanmean(roc_score), np.nanmean(blc_acc), np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(AUPR), np.nanmean(f1_score), np.nanmean(average_precision)

    
    def prob_to_labels(self, pred, threshold):
	    return (pred >= threshold).astype('int')
    
    def load_model_weights(self, model, checkpoint_file):
        """
        PL `load_from_checkpoint` seems to fail to reload model weights. This function loads them manually.
        See: https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        print(f'Loading model weights from {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        # load weights from checkpoint, strict=False allows to ignore some weights
        # e.g. weights of a head that was used during pretraining but isn't present during finetuning
        # and also allows to missing keys in the checkpoint, e.g. heads that are used for finetuning
        # but weren't present during pretraining
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model

    def get_config(self):
        if not hasattr(self.hparams, 'vocab_size') or not self.hparams.vocab_size:
            self.hparams.vocab_size = 42

        if self.hparams.tiny:
            config = BertConfigExtras(
                vocab_size_or_config_json_file=self.hparams.vocab_size,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=self.hparams.max_position_embeddings,
                mode=self.hparams.mode,
                output_size=self.hparams.output_size,
                label_column=self.hparams.label_column,
            )
        else:
            config = BertConfigExtras(
                vocab_size_or_config_json_file=self.hparams.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=self.hparams.max_position_embeddings,
                mode=self.hparams.mode,
                output_size=self.hparams.output_size,
                label_column=self.hparams.label_column,
            )
        return config
    
    def get_tasks(self, config):
        """ Task list should be converted to nn.ModuleList before, not done here to hide params from torch """
        tasks: List[BaseTask] = [FinetuneTask(name='finetune', config=config)]

        return tasks
    '''
    def load_datasets(self):
        featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(self.hparams.max_seq_length)

        train_dataset = BertFinetuneSmilesDataset(
            input_path=self.hparams.train_file,
            featurizer=featurizer,
            single_seq_len=self.hparams.max_seq_length,
            total_seq_len=self.hparams.max_seq_length,
            label_column=self.hparams.label_column,
            is_same=False,
        )

        validation_dataset = BertFinetuneSmilesDataset(
            input_path=self.hparams.valid_file,
            featurizer=featurizer,
            single_seq_len=self.hparams.max_seq_length,
            total_seq_len=self.hparams.max_seq_length,
            label_column=self.hparams.label_column,
            is_same=False,
        )

        test_dataset = BertFinetuneSmilesDataset(
            input_path=self.hparams.test_file,
            featurizer=featurizer,
            single_seq_len=self.hparams.max_seq_length,
            total_seq_len=self.hparams.max_seq_length,
            label_column=self.hparams.label_column,
            is_same=False,
            inference_mode=True,
        )

        return {'train': train_dataset, 'valid': validation_dataset, 'test': test_dataset}
    
    def train_dataloader(self) -> DataLoader:
        dataset = self.datasets['train']
        return self._get_dataloader(dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        dataset = self.datasets['valid']
        return self._get_dataloader(dataset)

    def test_dataloader(self) -> DataLoader:
        dataset = self.datasets['test']
        return self._get_dataloader(dataset)

    def _get_dataloader(self, dataset, **kwargs):
        return MolbertDataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, **kwargs
        )
    
'''


# In[6]:


import pandas as pd

model_weights_dir = '/projects/home/mmasood1/Model_weights/preclinical_clinical/BERT/'
pretrained_model_path = '/projects/home/mmasood1/TG GATE/MolBERT/molbert/molbert_100epochs/molbert_100epochs/checkpoints/last.ckpt'
data_dir = '/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/Data_for_BERT_finetuning/'
pos_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/06_10_2023/pos_weights.csv"
class_weights = "/projects/home/mmasood1/arslan_data_repository/Mix_clinical_pre_clinical/06_10_2023/target_weights.csv"
metadata_dir = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/BERT_finetune/inference_True/"
model_dir = os.path.dirname(os.path.dirname(pretrained_model_path))
hparams_path = os.path.join(model_dir, 'hparams.yaml')

# load config
with open(hparams_path) as yaml_file:
    config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

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
config_dict['alpha'] = 1.0
config_dict['beta'] = 0
config_dict['epochs'] = 50
config_dict["l2_lambda"] = 0.0
config_dict['missing'] = 'nan'
config_dict['compute_metric_after_n_epochs'] = 5
config_dict['return_trainer'] = True
config_dict['EarlyStopping'] = False
config_dict['project_name'] = "BERT_Testing"
config_dict['model_name'] = "Test"

config_dict["accelerator"] = "gpu"
config_dict["gpu"] =  [0]
config_dict["device"] = torch.device("cuda")


data = pd.read_csv(config_dict["data_dir"] + "train_fold0.csv")
target_names = data.loc[:,"Cytoplasmic alteration (Basophilic/glycogen depletion)":"hepatobiliary_disorders"].columns.tolist()
#target_names = data.loc[:,"DILI_binary":"hepatobiliary_disorders"].columns.tolist()
config_dict["output_size"] = len(target_names)
config_dict["label_column"] = target_names

config_dict["num_of_tasks"] = len(target_names)
config_dict["selected_tasks"] = target_names


# In[7]:


from molbert.datasets.finetune import BertFinetuneSmilesDataset
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer

featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config_dict["max_seq_length"])


train_dataset = BertFinetuneSmilesDataset(
            input_path= config_dict['train_file'],
            featurizer=featurizer,
            single_seq_len=config_dict["max_seq_length"],
            total_seq_len=config_dict["max_seq_length"],
            label_column=config_dict["label_column"],
            is_same=False,
            inference_mode=True
        )

validation_dataset = BertFinetuneSmilesDataset(
            input_path= config_dict['valid_file'],
            featurizer=featurizer,
            single_seq_len=config_dict["max_seq_length"],
            total_seq_len=config_dict["max_seq_length"],
            label_column=config_dict["label_column"],
            is_same=False,
            inference_mode=True
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


# In[8]:


model = MolbertModel(config_dict)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)


# In[ ]:





# In[9]:


import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


def wandb_init_model(model, 
                     config, 
                     train_dataloader,
                     val_dataloader, 
                     model_type):
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
        use_EarlyStopping = config["EarlyStopping"]
        max_epochs = config["epochs"]
        accelerator =config["accelerator"]
        return_trainer = config["return_trainer"]

    #if use_pretrained_model:
    #    model = pretrained_model(model,config)
    #else:
    model = model(config)
    wandb_logger = WandbLogger( 
                        name = config["model_name"],
                        save_dir = '/projects/home/mmasood1/Model_weights',
                        project= config["project_name"],
                        entity="arslan_masood", 
                        log_model='all',
                        #reinit = True, 
                        #config = config,
                        #settings=wandb.Settings(start_method="fork")
                        )
    wandb_logger.watch(model)
    wandb_logger.log_dir = '/projects/home/mmasood1/Model_weights'
    
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

    '''
    checkpoint_callback = ModelCheckpoint(
                            monitor=None,  # Metric to monitor for saving the best model
                            mode='min',          # Minimize the monitored metric
                            filepath = '/projects/home/mmasood1/Model_weights/',  # Directory to store checkpoints
                            #filename='model-{epoch:02d}-{val_BCE_non_weighted:.2f}',  # Checkpoint filename format
                            #filename=config['chkp_file_name'],  # Checkpoint filename format
                            #save_top_k=1,
                            #save_last = True
                            )
    callback.append(checkpoint_callback)
    '''


    trainer = Trainer(
        #callbacks=callback,
        max_epochs= int(max_epochs),
        #accelerator= accelerator, 
        #devices= config['gpu'],
        #limit_val_batches = limit_val_batches,
        #precision=16,
        #enable_progress_bar = True,
        #profiler="simple",
        #enable_model_summary=True,
        #auto_select_gpus= True,
        gpus = -1,
        #logger = wandb_logger,
        default_root_dir=default_root_dir)

    # model fitting 
    trainer.fit(model, 
                train_dataloader = train_dataloader,
                val_dataloaders = val_dataloader,
                )
    if return_trainer:
        return model, run, trainer
    else:
        return model, run


# In[10]:


os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")


# In[11]:


trained_model, run, trainer = wandb_init_model(model = MolbertModel, 
                                                                train_dataloader = train_dataloader,
                                                                val_dataloader =validation_dataloader,
                                                                config = config_dict, 
                                                                model_type = 'MLP')


# model = MolbertModel(config_dict)
# checkpoint_file = "/projects/home/mmasood1/Model_weights/preclinical_clinical/Vanilla_MLP/lightning_logs/version_9/checkpoints/epoch=7.ckpt"
# checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
# # load weights from checkpoint, strict=False allows to ignore some weights
# # e.g. weights of a head that was used during pretraining but isn't present during finetuning
# # and also allows to missing keys in the checkpoint, e.g. heads that are used for finetuning
# # but weren't present during pretraining
# model.load_state_dict(checkpoint['state_dict'], strict=False)

# In[12]:


data_dir = config_dict["metadata_dir"] + "predicitons/"
result_dir = config_dict["metadata_dir"] + "Results/"  
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    os.makedirs(result_dir)


# In[13]:


model = trained_model.eval()
#model = model.eval()
config = config_dict
device = torch.device('cuda')
model = model.cpu() 

y_true_list = []
y_pred_list = []

for batch in validation_dataloader:
    
    (batch_inputs, batch_labels), _ = batch
    y = batch_labels["finetune"].squeeze()

    input_ids = batch_inputs["input_ids"].cpu()
    token_type_ids = batch_inputs["token_type_ids"].cpu()
    attention_mask = batch_inputs["attention_mask"].cpu()


    y_hat = model(input_ids,token_type_ids, attention_mask)

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

y.to_csv(data_dir + 'y_true_test.csv',index=False)
y_hat.to_csv(data_dir + 'y_pred_test.csv',index=False)


# In[14]:


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
        

        metrics_list.append(metrics_task)
    metrics_df = pd.DataFrame(metrics_list)
    col = ['balanced_acc', 'f1_score','specificity','sensitivity', 'roc_auc','AUPR', 'average_precision']
    
    return metrics_df[col]


# In[15]:


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
config["batch_size"] = 64
config["seed"] = 4
config["split_method"] = "StratifiedGroupKFold"
config["test_set_creteria"] = "most_diverse_fold"
config["task_for_stratification"] = "DILI_binary"


# In[16]:


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
        

        metrics_list.append(metrics_task)
    metrics_df = pd.DataFrame(metrics_list)
    col = ['balanced_acc', 'f1_score','specificity','sensitivity', 'roc_auc','AUPR', 'average_precision']
    
    return metrics_df[col]


# In[ ]:





# In[17]:


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
metrics.to_csv(result_dir + f'val_metric.csv', index=False)


# In[18]:


metrics.tail(5)


# In[ ]:




