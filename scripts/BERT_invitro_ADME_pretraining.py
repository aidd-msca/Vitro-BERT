import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
import argparse
import wandb

# Get project root from environment variable or use default
PROJECT_ROOT = os.getenv('TOXBERT_ROOT', str(Path(__file__).parent.parent))
sys.path.append(PROJECT_ROOT)

# Add src directory to path
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_PATH)

from molbert.models.ToxBERT import ToxBERT_model
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.datasets.dataloading import get_dataloaders
from training.utils import setup_wandb, load_config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train ToxBERT model on invitro data')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to configuration YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save model outputs')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                      help='Path to pretrained model weights')
    parser.add_argument('--wandb_key', type=str, default=None,
                      help='Weights & Biases API key (optional)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up wandb if key is provided
    if args.wandb_key:
        os.environ["WANDB_SILENT"] = "true"
        wandb.login(key=args.wandb_key)
    
    # Load configuration
    config_dict = load_config(args.config_path)
    
    # Update config with command line arguments
    config_dict["data_dir"] = args.data_dir
    config_dict["output_dir"] = args.output_dir
    config_dict["pretrained_model_path"] = args.pretrained_weights
    config_dict["model_weights_dir"] = args.output_dir
    
    # Update file paths in config
    config_dict["invitro_pos_weights"] = os.path.join(args.data_dir, "split_ratio_Random.csv")
    
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
        num_workers=config_dict["num_workers"],
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

    print("Training completed successfully")

if __name__ == "__main__":
    main()