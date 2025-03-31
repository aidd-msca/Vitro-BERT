import os
import sys
import yaml
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

# Add the source directory to Python path
sys.path.append('/scratch/work/masooda1/ToxBERT/src')

from molbert.models.ToxBERT import ToxBERT_model
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.datasets.dataloading import get_dataloaders
from training.utils import setup_wandb, load_config

import wandb

def main():
    # Set up wandb
    os.environ["WANDB_SILENT"] = "true"
    wandb.login(key="27edf9c66b032c03f72d30e923276b93aa736429")

    # Load configuration
    config_path = "/scratch/work/masooda1/ToxBERT/scripts/config/BERT_init_masking_physchem_invitro_head.yaml"
    config_dict = load_config(config_path)

    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Initialize featurizer
    featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(
        config_dict["max_seq_length"], 
        permute=False
    )
    config_dict["vocab_size"] = featurizer.vocab_size

    # Get dataloaders
    invitro_train_dataloader, invitro_val_dataloader = get_dataloaders(
        featurizer=featurizer,
        targets="invitro",
        num_workers=config_dict["num_workers"],  # Increased from 1 to 12 based on warning
        config_dict=config_dict
    )
    config_dict["num_batches"] = len(invitro_train_dataloader)
    config_dict["num_sanity_val_steps"] = len(invitro_val_dataloader)

    # Set random seed
    seed_everything(config_dict["seed"])

    # Initialize model and trainer
    model = ToxBERT_model(config_dict)
    trainer = setup_wandb(config_dict)

    # Train model
    trainer.fit(model, invitro_train_dataloader, invitro_val_dataloader)

    print("Script completed")

if __name__ == "__main__":
    main()