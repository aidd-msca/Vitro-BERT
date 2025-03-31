from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import os, wandb, yaml, torch
import pandas as pd

############################################
# Setup W&B
############################################

def setup_wandb(config):
    """Initialize and setup W&B logging. Model saving is enabled by default."""
    
    wandb_logger = WandbLogger(
        name=config["model_name"],
        save_dir=config["model_weights_dir"],
        project=config["project_name"],
        entity="arslan_masood",
        log_model=False,
    )

    # Set up checkpoint callback - default to True unless explicitly set to False
    if config.get("save_model", True):  # Changed default to True
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(config["model_weights_dir"], '{epoch}-{step}'),
            save_top_k=config["save_top_k"],
            verbose=True
        )
    else:
        checkpoint_callback = False

    trainer = Trainer(
        max_epochs=config["max_epochs"],
        distributed_backend=config["distributed_backend"],
        gpus=config["gpus"],
        logger=wandb_logger,
        precision=config["precision"],
        default_root_dir=config["model_weights_dir"],
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=config["pretrained_crash_model"],
        check_val_every_n_epoch=config["compute_metric_after_n_epochs"],
        #val_check_interval = 0.1, 
        fast_dev_run = config["fast_dev_run"],
        #limit_train_batches = config["limit_train_batches"],
        #limit_val_batches = config["limit_val_batches"],
        num_sanity_val_steps = config["num_sanity_val_steps"]
    )

    return trainer

############################################
# Load Config
############################################

def load_config(custom_config_path):
    """Load and merge configurations from base hparams and custom config."""
    # 1. First load custom config to get the pretrained model path
    with open(custom_config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. Load base hparams using path from config
    model_dir = os.path.dirname(os.path.dirname(config["pretrained_model_path"]))
    hparams_path = os.path.join(model_dir, 'hparams.yaml')
    
    # Load base config
    with open(hparams_path) as yaml_file:
        base_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        
    # 3. Update base config with custom config
    base_config.update(config)
    
    # Construct full paths for data files
    base_config["invitro_train"] = os.path.join(base_config["data_dir"], base_config["invitro_train"])
    base_config["invitro_val"] = os.path.join(base_config["data_dir"], base_config["invitro_val"])
    base_config["invitro_test"] = os.path.join(base_config["data_dir"], base_config["invitro_test"])
    
    # 4. Add computed fields from data
    data = pd.read_pickle(base_config["invitro_train"])  # Now using the full path
    data.drop(['SMILES'], axis=1, inplace=True)
    target_names = data.columns.tolist()
    
    # Add additional computed fields
    base_config.update({
        "output_size": len(target_names),
        "num_invitro_tasks": len(target_names),
        "num_of_tasks": len(target_names),
        "invitro_columns": target_names,
        "num_physchem_properties": 200,
        "device": torch.device("cuda"),
        "compute_classification": False,
        "permute": False
    })
    
    return base_config