# Standard library imports
import gc
import logging
import os
import sys
import warnings
from argparse import Namespace


# Third-party imports
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)

import pytorch_lightning as pl
from transformers import (
    AdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
import torch.distributed as dist

# Configure warnings and logging
warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

# Environment setup
#os.environ["WANDB_SILENT"] = "true"
#self.logger.experiment.login(key="27edf9c66b032c03f72d30e923276b93aa736429")
#sys.path.append('/scratch/work/masooda1/ToxBERT/src')

# Local imports
from molbert.models.smiles import SmilesMolbertModel

############################################
# ToxBERT model
############################################
class ToxBERT_model(pl.LightningModule):

    def __init__(self, args: Namespace):
        super().__init__()
        
        # Initialize lists to store predictions and labels
        self.training_step_invitro_labels, self.training_step_invitro_pred = [],[]
        self.training_step_physchem_labels, self.training_step_physchem_pred = [],[]
        self.training_step_masked_lm_labels, self.training_step_masked_lm_pred = [],[]
        self.val_step_invitro_labels, self.val_step_invitro_pred = [],[]
        self.val_step_physchem_labels, self.val_step_physchem_pred = [],[]
        self.val_step_masked_lm_labels, self.val_step_masked_lm_pred = [],[]

        self.hparams = args
        self.non_weighted_creterian, self.invitro_weighted_creterien, self.invitro_FL = self.get_creterian(args, targets = "invitro")

        # get model, load pretrained weights, and freeze encoder        
        model = SmilesMolbertModel(self.hparams)
        if self.hparams.pretrained_model:
            checkpoint = torch.load(self.hparams.pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'], strict = False)
        
        if self.hparams.freeze_level:
            # Freeze model
            ToxBERT_model.freeze_network(model, self.hparams.freeze_level)

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
        l2_reg = torch.tensor(0., requires_grad=True, device=self.device)
        
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
        (corrupted_batch_inputs,clean_batch_inputs), corrupted_batch_labels = batch
        
        # forward pass with clean and corrupted sequence
        Masked_token_pred, _,_ = self.forward(corrupted_batch_inputs)
        _, Physchem_pred, invitro_pred = self.forward(clean_batch_inputs)

        # loss 
        masking_loss = self.MaskedLM_loss(corrupted_batch_labels, Masked_token_pred) 
        physchem_loss = self.Physchem_loss(corrupted_batch_labels, Physchem_pred)
        invitro_weighted_loss, invitro_Non_weighted_loss, invitro_pos_loss, invitro_neg_loss = self._compute_loss(corrupted_batch_labels["invitro"].squeeze(), invitro_pred, targets = "invitro") 

        total_loss = masking_loss + physchem_loss + invitro_weighted_loss

        self.training_step_invitro_labels.append(corrupted_batch_labels["invitro"].squeeze().long().detach().cpu())
        self.training_step_invitro_pred.append(torch.sigmoid(invitro_pred.detach().cpu()))
        self.training_step_masked_lm_labels.append(corrupted_batch_labels['lm_label_ids'].view(-1).long().detach().cpu())
        self.training_step_masked_lm_pred.append(torch.sigmoid(Masked_token_pred.view(-1, self.hparams.vocab_size).detach().cpu()))
        self.training_step_physchem_labels.append(corrupted_batch_labels["physchem_props"].long().detach().cpu())
        self.training_step_physchem_pred.append(Physchem_pred.detach().cpu())

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
        # Log the losses with WandB only on main process
        if self.trainer.is_global_zero:
            log_dict = {f'{step_prefix}{key}_step': value.item() for key, value in losses.items()}
            log_dict["global_step"] = self.trainer.global_step
            self.logger.experiment.log(log_dict)
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

            # save predictions for accuracy calculations
            self.val_step_invitro_labels.append(corrupted_batch_labels["invitro"].squeeze().long().detach().cpu())
            self.val_step_invitro_pred.append(torch.sigmoid(invitro_pred.detach().cpu()))
            self.val_step_masked_lm_labels.append(corrupted_batch_labels['lm_label_ids'].view(-1).long().detach().cpu())
            self.val_step_masked_lm_pred.append(torch.sigmoid(Masked_token_pred.view(-1, self.hparams.vocab_size).detach().cpu()))
            self.val_step_physchem_labels.append(corrupted_batch_labels["physchem_props"].long().detach().cpu())
            self.val_step_physchem_pred.append(Physchem_pred.detach().cpu())
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
            self.logger.experiment.log(log_dict)
        return losses

    def on_epoch_start(self):
        """Handle any epoch start operations that are common to both training and validation"""
        # Save at epoch 0, but only from the main process
        if self.current_epoch == 0 and self.trainer.is_global_zero:
            epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            print(f"Saving checkpoint before first epoch/step")
            filename = f"epoch_{epoch}_step_{global_step}.ckpt"
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, filename)
            #self.trainer.save_checkpoint(ckpt_path)

    def training_epoch_end(self, outputs): 
        log_dict = self.compute_epoch_metrics(outputs, mode='train')
        if self.trainer.is_global_zero:
            self.logger.experiment.log(log_dict)
        return log_dict

    def validation_epoch_end(self, outputs): 
        log_dict = self.compute_epoch_metrics(outputs, mode='val')
        if self.trainer.is_global_zero:
            self.logger.experiment.log(log_dict)
        return log_dict
    
    def on_epoch_end(self):

        weight_norm = self.l2_regularization()
        weight_norm = {'weight_norm': weight_norm}
        self.logger.experiment.log(weight_norm)

       # Then clean the cache
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            with torch.cuda.device(f'cuda:{gpu_id}'):
                torch.cuda.empty_cache()
        # then collect the garbage
        gc.collect()
        print("!!!!!!!!! ALL CLEAR !!!!!!!!!!!!!!!!!")

    def test_step(self, batch, batch_idx):

        with torch.no_grad():
            # compute forward pass
            batch, valid = batch
            (_,clean_batch_inputs), _ = batch
            sequence_output, pooled_output = self.encoder(**clean_batch_inputs)
            
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
        if self.hparams.dataloader_name == "invivo_test":
            SMILES = pd.read_csv(self.hparams.invivo_test).SMILES

        embeddings.insert(0,"SMILES", SMILES)

        file_name = self.hparams.metadata_dir + f"{self.hparams.dataloader_name}_{self.hparams.split_type}_epoch_{self.hparams.epoch_number}_step_{self.hparams.step_number}.csv"
        #file_name = self.hparams.metadata_dir + f"{self.hparams.dataloader_name}_epoch_init_step_{self.hparams.step_number}.csv"

        embeddings.to_csv(file_name, index = False)
    def compute_epoch_metrics(self, outputs, mode='train'):
        # Calculate local losses for this GPU
        losses = {key: torch.stack([x[key] for x in outputs]).mean() for key in outputs[0].keys()}

        # Select appropriate data based on mode
        if mode == 'train':
            invitro_labels = torch.cat(self.training_step_invitro_labels, dim=0)
            invitro_pred = torch.cat(self.training_step_invitro_pred, dim=0)
            masked_lm_labels = torch.cat(self.training_step_masked_lm_labels, dim=0)
            masked_lm_pred = torch.cat(self.training_step_masked_lm_pred, dim=0)
            physchem_labels = torch.cat(self.training_step_physchem_labels, dim=0)
            physchem_pred = torch.cat(self.training_step_physchem_pred, dim=0)
        else:  # val
            invitro_labels = torch.cat(self.val_step_invitro_labels, dim=0)
            invitro_pred = torch.cat(self.val_step_invitro_pred, dim=0)
            masked_lm_labels = torch.cat(self.val_step_masked_lm_labels, dim=0)
            masked_lm_pred = torch.cat(self.val_step_masked_lm_pred, dim=0)
            physchem_labels = torch.cat(self.val_step_physchem_labels, dim=0)
            physchem_pred = torch.cat(self.val_step_physchem_pred, dim=0)
        # Print shapes for debugging
        print(f"\n{mode} epoch end (Rank {self.trainer.global_rank})")
        print(f"invitro_labels: {invitro_labels.shape}")
        print(f"invitro_pred: {invitro_pred.shape}")
        print(f"masked_lm_labels: {masked_lm_labels.shape}")
        print(f"masked_lm_pred: {masked_lm_pred.shape}")
        print(f"physchem_labels: {physchem_labels.shape}")
        print(f"physchem_pred: {physchem_pred.shape}")

        # Compute metrics
        invitro_score_list = self.compute_classification_metrics(invitro_labels, invitro_pred, targets_type="invitro")
        masked_lm_accuracy = self.compute_masked_lm_accuracy(masked_lm_labels, masked_lm_pred)
        physchem_score_list = self.compute_physchem_metrics(physchem_labels, physchem_pred)

        # Create log dictionary starting with losses
        log_dict = {f'{mode}_{key}_epoch': value for key, value in losses.items()}
        
        # Add metrics
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision', 'ECE_score', 'ACE_score']
        physchem_metric = ['MAE', 'MSE', 'R2']

        # Add invitro metrics
        for i, score in enumerate(invitro_score_list):
            log_dict.update({f'{mode}_invitro_{metric[i]}': score.item()})

        # Add masked_lm accuracy
        log_dict.update({f'{mode}_masked_lm_accuracy': masked_lm_accuracy.item()})

        # Add physchem metrics
        for i, score in enumerate(physchem_score_list):
            log_dict.update({f'{mode}_physchem_{physchem_metric[i]}': score.item()})

        # Add epoch and step info
        log_dict.update({
            "current_epoch": self.trainer.current_epoch + 1,
            "global_step": self.trainer.global_step
        })

        # Clear all lists
        if mode == 'train':
            self.training_step_invitro_labels, self.training_step_invitro_pred = [], []
            self.training_step_physchem_labels, self.training_step_physchem_pred = [], []
            self.training_step_masked_lm_labels, self.training_step_masked_lm_pred = [], []
        else:  # val
            self.val_step_invitro_labels, self.val_step_invitro_pred = [], []
            self.val_step_physchem_labels, self.val_step_physchem_pred = [], []
            self.val_step_masked_lm_labels, self.val_step_masked_lm_pred = [], []

        return log_dict

    def compute_masked_lm_accuracy(self, labels, preds):
        """Compute accuracy for masked language modeling task."""
        # Convert predictions to token indices
        pred_tokens = torch.argmax(preds, dim=1)  # Shape: [8192]
        
        # Ignore positions with -1 labels (masked positions)
        valid_mask = (labels != -1)
        valid_labels = labels[valid_mask]
        valid_preds = pred_tokens[valid_mask]

        # Compute accuracy
        correct = (valid_labels == valid_preds).float()
        accuracy = correct.mean()
        
        return accuracy
    def compute_physchem_metrics(self, labels, preds):
        """Compute regression metrics for physchem predictions using scikit-learn."""
        
        # Convert tensors to numpy arrays
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        # Compute metrics using scikit-learn
        mae = mean_absolute_error(labels, preds)
        mse = mean_squared_error(labels, preds)
        r2 = r2_score(labels, preds)
        
        return [mae, mse, r2]

    def compute_classification_metrics(self, y_true, y_pred, targets_type): 
        self.eval()
        targets =  y_true.cpu().detach().tolist()
        preds = y_pred.cpu().detach().tolist()

        if targets_type == "invitro":
            num_of_tasks = self.hparams.num_of_tasks
            targets = np.array(targets).reshape(-1,num_of_tasks)
            preds = np.array(preds).reshape(-1,num_of_tasks)
        if targets_type == "masked_lm":
            num_of_tasks = self.hparams.vocab_size

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


############################################
# Focal Loss
############################################    
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