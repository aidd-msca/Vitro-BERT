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
model_weights_dir = '/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/BERT_pretraining_init_MolBERT_masking_physchem_invitro_head/'
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

config_dict['project_name'] = "BERT_pretraining_masking_physchem_invitro"
config_dict['model_name'] = "BERT_init_masking_physchem_invitro_head"

config_dict['model_weights_dir'] = model_weights_dir
config_dict['pretrained_model_path'] = pretrained_model_path
config_dict["metadata_dir"] = metadata_dir
config_dict['invitro_pos_weights'] = invitro_pos_weights
config_dict['invivo_pos_weights'] = invivo_pos_weights
config_dict['data_dir'] = data_dir
config_dict['invitro_train'] = data_dir + "train_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['invitro_val'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"
config_dict['invitro_test'] = data_dir + "test_set_invitro_1m_300k_ADME_filtered.pkl"

config_dict['max_epochs'] = 100
config_dict['unfreeze_epoch'] = 210
config_dict["l2_lambda"] = 0.0
config_dict['embedding_size'] = 50

config_dict['max_seq_length'] = 128
config_dict['bert_output_dim'] = 768

config_dict['optim'] = 'AdamW'#SGD
config_dict['lr'] = 1e-05
config_dict["BERT_lr"] = 3e-5
config_dict["compute_classification"] = False
config_dict["seed"] = 42
config_dict['compute_metric_after_n_epochs'] = 5
config_dict['return_trainer'] = True
config_dict['EarlyStopping'] = False

config_dict["accelerator"] = "gpu"
config_dict["device"] = torch.device("cuda")
config_dict["precision"] = 32

######## invitro #########################
config_dict["invitro_batch_size"] = 200
config_dict['invitro_head_hidden_layer'] = 2048

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
config_dict["invivo_batch_size"] = 200

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
config_dict["pretrained_crash_model"] = None#"/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/BERT_pretraining_init_MolBERT_masking_head/epoch=39-step=0.ckpt"

# dataloaders
featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config_dict["max_seq_length"], permute = False)
config_dict["vocab_size"] = featurizer.vocab_size
invitro_train_dataloader, invitro_val_dataloader = get_dataloaders(
                                                                    featurizer = featurizer, 
                                                                    targets = "invitro", 
                                                                    num_workers = 24,
                                                                    config_dict = config_dict)
config_dict["num_batches"] = len(invitro_train_dataloader)

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
        
        self.training_step_ytrue, self.training_step_ypred = [],[]
        self.val_step_ytrue, self.val_step_ypred = [],[]

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

        # encoder --> masking head
        #         --> Physchem head
        #         --> invitro head

        self.encoder = model.model.bert
        self.Masked_LM_task = model.model.tasks[0]
        self.Physchem_task = model.model.tasks[1]
        self.invitro_task = model.model.tasks[2]


    def forward(self, batch_inputs):
        sequence_output, pooled_output = self.encoder(**batch_inputs)
        Masked_token_pred = self.Masked_LM_task(sequence_output, pooled_output)
        Physchem_pred = self.Physchem_task(sequence_output, pooled_output)
        invitro_pred = self.invitro_task(sequence_output, pooled_output)

        
        return Masked_token_pred, Physchem_pred, invitro_pred
    
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
        
    def training_step(self, batch, batch_idx):

        # get batch
        batch, valid = batch
        (corrupted_batch_inputs,clean_batch_inputs),  corrupted_batch_labels = batch

        # forward pass with clean and corrupted sequence
        Masked_token_pred, _,_ = self.forward(corrupted_batch_inputs)
        _, Physchem_pred, invitro_pred = self.forward(clean_batch_inputs)

        # loss 
        masking_loss = self.MaskedLM_loss(corrupted_batch_labels, Masked_token_pred) 
        physchem_loss = self.Physchem_loss(corrupted_batch_labels, Physchem_pred)
        invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(corrupted_batch_labels["invitro"].squeeze(), invitro_pred, targets = "invitro") 

        total_loss = masking_loss + physchem_loss + invitro_weighted_loss

        return {
                "loss": total_loss,
                "masking_loss": masking_loss,
                "physchem_loss": physchem_loss,
                "invitro_weighted_loss": invitro_weighted_loss,
                "invitro_Non_weighted_loss": invitro_Non_weighted_loss,
                "invitro_pos_loss": invitro_pos_loss,
                "invitro_neg_loss": invitro_neg_loss,
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
        if (global_step % interval_batches == 0) and (epoch == 0):
            # Log the current epoch and step for clarity
            print(f"Saving checkpoint at epoch {epoch}, step {global_step}")
            filename = f"epoch_{epoch}_step_{global_step}.ckpt"
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
            self.trainer.save_checkpoint(ckpt_path)
        return losses
    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            # get batch
            batch, valid = batch
            (corrupted_batch_inputs,clean_batch_inputs),  corrupted_batch_labels = batch

            # forward pass with clean and corrupted sequence
            Masked_token_pred, _,_ = self.forward(corrupted_batch_inputs)
            _, Physchem_pred, invitro_pred = self.forward(clean_batch_inputs)

            # loss 
            masking_loss = self.MaskedLM_loss(corrupted_batch_labels, Masked_token_pred) 
            physchem_loss = self.Physchem_loss(corrupted_batch_labels, Physchem_pred)
            invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(corrupted_batch_labels["invitro"].squeeze(), invitro_pred, targets = "invitro") 

            total_loss = masking_loss + physchem_loss + invitro_weighted_loss

        return {
                "loss": total_loss,
                "masking_loss": masking_loss,
                "physchem_loss": physchem_loss,
                "invitro_weighted_loss": invitro_weighted_loss,
                "invitro_Non_weighted_loss": invitro_Non_weighted_loss,
                "invitro_pos_loss": invitro_pos_loss,
                "invitro_neg_loss": invitro_neg_loss,
            }
    
    def validation_step_end(self, outputs):
        # Define the step prefix
        step_prefix = "val_"
       # Calculate mean losses
        losses = {key: outputs[key].mean() for key in outputs.keys()}
        # Log the losses with WandB
        log_dict = {f'{step_prefix}{key}_step': value.item() for key, value in losses.items()}
        log_dict["global_step"] = self.trainer.global_step
        if self.trainer.global_step > 0:
            wandb.log(log_dict)
        return losses
    
    def on_epoch_start(self):
        # Save at epoch 0
        if self.current_epoch == 0:
            epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            print(f"Saving checkpoint before first epoch/step")
            filename = f"epoch_{epoch}_step_{global_step}.ckpt"
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
            self.trainer.save_checkpoint(ckpt_path)

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
        wandb.log(log_dict)
        return losses
        
    def validation_epoch_end(self, outputs):

        step_prefix = "val_"
        losses = {key: torch.stack([x[key] for x in outputs]).mean().item() for key in outputs[0].keys()}
        # Log the losses with WandB
        log_dict = {f'{step_prefix}{key}_epoch': loss for key, loss in losses.items()}
        log_dict.update({
            "current_epoch": self.trainer.current_epoch + 1,
            "global_step": self.trainer.global_step
        })
        if self.trainer.global_step > 0:
            wandb.log(log_dict)
        
        return losses
    
    def test_step(self, batch, batch_idx):

        with torch.no_grad():
            # compute forward pass
            batch, valid = batch
            (corrupted_batch_inputs,_),  _ = batch
            sequence_output, pooled_output = self.encoder(**corrupted_batch_inputs)
            
            # set invalid outputs to 0s
            valid_tensor = torch.tensor(
                valid, dtype=sequence_output.dtype, device=sequence_output.device, requires_grad=False
            )
            pooled_output = pooled_output * valid_tensor[:, None]

            return {"pooled_output": pooled_output.detach().cpu().numpy(),
                    "valid": valid.detach().cpu().numpy()
                    }
        
    def test_epoch_end(self, outputs):
        outputs = {key: np.concatenate([output[key] for output in outputs], axis=0) for key in outputs[0]}
        embeddings = pd.DataFrame(outputs["pooled_output"])
        embeddings.insert(0,"valid", outputs["valid"])

        if self.hparams.dataloader_name == "invitro_train":
            SMILES = pd.read_pickle(self.hparams.invitro_train).SMILES
        if self.hparams.dataloader_name == "invitro_val":
            SMILES = pd.read_pickle(self.hparams.invitro_val).SMILES
        if self.hparams.dataloader_name == "invivo_train":
            SMILES = pd.read_csv(self.hparams.invivo_train).SMILES
        if self.hparams.dataloader_name == "invivo_val":
            SMILES = pd.read_csv(self.hparams.invivo_val).SMILES

        embeddings.insert(0,"SMILES", SMILES)

        file_name = self.hparams.metadata_dir + f"{self.hparams.dataloader_name}_epoch_{self.hparams.epoch_number}_step_{self.hparams.step_number}.csv"
        #file_name = self.hparams.metadata_dir + f"{self.hparams.dataloader_name}_epoch_init_step_{self.hparams.step_number}.csv"

        embeddings.to_csv(file_name, index = False)
        #return mean_losses

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
#########################################################################    
invitro_train_dataloader, invitro_val_dataloader = get_dataloaders(
                                                                    featurizer = featurizer, 
                                                                    targets = "invitro", 
                                                                    num_workers = 6,
                                                                    config_dict = config_dict,
                                                                    train_shuffle = False)
invivo_train_dataloader, invivo_val_dataloader = get_dataloaders(
                                                                    featurizer = featurizer, 
                                                                    targets = "invivo", 
                                                                    num_workers = 6,
                                                                    config_dict = config_dict,
                                                                    train_shuffle = False)

config_dict["metadata_dir"] = "/projects/home/mmasood1/trained_model_predictions/SIDER_PreClinical/MolBERT/BERT_pretraining_init_MolBERT_masking_physchem_invitro_head/embeddings/"
model_weights_path = "/projects/home/mmasood1/Model_weights/invitro/invitro_1million/MolBERT/Retrain_on_top_of_BERT/complete_1m_300k_ADME/BERT_pretraining_init_MolBERT_masking_physchem_invitro_head/"
config_dict["gpu"] = [2]

data_loader_name_list = [#"invitro_train",
                         #"invitro_val",
                        "invivo_train", 
                        "invivo_val"]

data_loader_list = [#invitro_train_dataloader,
                    #invitro_val_dataloader,
                    invivo_train_dataloader, 
                    invivo_val_dataloader]

os.makedirs(config_dict["metadata_dir"], exist_ok= True)
#step_list = [0,567,1134,1701,2268,2835,3402,3969,4536,5103,5670]
#epoch_list = [0]
#for epoch_number in epoch_list:

step_list = [0]
epoch_start = 80
for epoch_number in range(epoch_start,epoch_start + 20):

    for step_number in step_list:
        for i, d_loader in enumerate(data_loader_list):

            config_dict["step_number"]  = step_number
            config_dict["epoch_number"]  = epoch_number
            config_dict["dataloader_name"] = data_loader_name_list[i]

            # load epoch specific weights
            model = MolbertModel(config_dict)
            checkpoint = torch.load(model_weights_path + f"epoch={epoch_number}-step={step_number}.ckpt", map_location=lambda storage, loc: storage)
            #checkpoint = torch.load(model_weights_path + f"epoch_{epoch_number}_step_{step_number}.ckpt", map_location=lambda storage, loc: storage)

            model.load_state_dict(checkpoint['state_dict'], strict = True)
            
            model = model.eval()
            trainer = Trainer(
                gpus = config_dict["gpu"],
                distributed_backend= config_dict["distributed_backend"],
                #limit_test_batches = 2
                )
            
            trainer.test(model = model, 
                    test_dataloaders = d_loader)
    
print("script complete")