"""Training utilities for BERT ADME model."""

import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

def setup_wandb_logging(config):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=config["project_name"],
        dir=config["model_weights_dir"],
        entity=config.get("wandb_entity", "arslan_masood"),
        reinit=True,
        config=config,
        name=config["model_name"],
        settings=wandb.Settings(start_method="fork")
    )

def get_wandb_logger(config):
    """Get WandB logger instance."""
    return WandbLogger(
        name=config["model_name"],
        save_dir=config["model_weights_dir"],
        project=config["project_name"],
        entity=config.get("wandb_entity", "arslan_masood"),
        log_model=False,
    )

def get_checkpoint_callback(root_dir):
    """Get model checkpoint callback."""
    return ModelCheckpoint(
        filepath=os.path.join(root_dir, '{epoch}-{step}'),
        save_top_k=-1,
        verbose=True
    )

def get_trainer(config, logger, checkpoint_callback):
    """Initialize PyTorch Lightning trainer."""
    return Trainer(
        max_epochs=int(config["max_epochs"]),
        distributed_backend=config["distributed_backend"],
        gpus=config["gpu"],
        logger=logger,
        precision=config["precision"],
        default_root_dir=config["model_weights_dir"],
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=config.get("pretrained_crash_model")
    )

def train_model(model_class, config, train_dataloader, val_dataloader):
    """Main training function."""
    # Setup WandB
    setup_wandb_logging(config)
    
    # Initialize model
    model = model_class(config)
    
    # Setup logging and checkpointing
    logger = get_wandb_logger(config)
    checkpoint_callback = get_checkpoint_callback(config["model_weights_dir"])
    
    # Initialize trainer
    trainer = get_trainer(config, logger, checkpoint_callback)
    
    # Train model
    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    if config.get("return_trainer", True):
        return model, trainer
    return model

def cleanup_training():
    """Cleanup after training."""
    wandb.finish()