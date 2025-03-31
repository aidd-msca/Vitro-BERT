#!/usr/bin/env python
"""Main training script for BERT invitro model."""

import os
import yaml
import argparse
from pathlib import Path

import torch
import pandas as pd
from pytorch_lightning import seed_everything

from datasets.data_utils import MolbertDataLoader
from src.datasets.smiles import BertSmilesDataset
from src.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from models.bert import MolbertModel
from src.training.trainer import train_model, cleanup_training

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BERT ADME model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--wandb_key', type=str,
                      help='WandB API key')
    return parser.parse_args()

def load_config(config_path):
    """Load and process configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set device
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return config

def setup_data(config):
    """Setup datasets and dataloaders."""
    # Initialize featurizer
    featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(
        config["max_seq_length"], 
        permute=config["permute"]
    )
    config["vocab_size"] = featurizer.vocab_size
    
    # Create datasets
    train_dataset = BertSmilesDataset(
        input_path=config['train_file'],
        featurizer=featurizer,
        single_seq_len=config["max_seq_length"],
        total_seq_len=config["max_seq_length"],
        label_column=config["label_column"],
        num_invitro_tasks=config["num_invitro_tasks"],
        num_physchem=config["num_physchem_properties"],
        permute=config["permute"],
        inference_mode=False
    )
    
    val_dataset = BertSmilesDataset(
        input_path=config['valid_file'],
        featurizer=featurizer,
        single_seq_len=config["max_seq_length"],
        total_seq_len=config["max_seq_length"],
        label_column=config["label_column"],
        num_invitro_tasks=config["num_invitro_tasks"],
        num_physchem=config["num_physchem_properties"],
        permute=config["permute"],
        inference_mode=False
    )
    
    # Create dataloaders
    train_dataloader = MolbertDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=24,
        shuffle=True
    )
    
    val_dataloader = MolbertDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=24,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed_everything(config["seed"])
    
    # Setup WandB if key provided
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    
    # Setup data
    train_dataloader, val_dataloader = setup_data(config)
    
    # Train model
    model, trainer = train_model(
        model_class=MolbertModel,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    # Cleanup
    cleanup_training()

if __name__ == "__main__":
    main()