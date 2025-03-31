import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import argparse

# Add project root to path
sys.path.append('/scratch/work/masooda1/ToxBERT/src')
from datasets.data_utils import get_stratified_folds, dataloader_for_numpy
from training.utils import setup_wandb
from molbert.models.MLP_models import Vanilla_MLP_classifier

def parse_args():
    parser = argparse.ArgumentParser(description='MLP training with BERT features')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the config file')
    return parser.parse_args()

def create_dataloaders(X, y, ids, batch_size, shuffle=True, num_workers=4):
    return DataLoader(
        dataloader_for_numpy(X, y, x_type='Fingerprints'),
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=shuffle)

def run_cv_fold(config, fold_data, fold, param_id):
    """Run a single fold of cross-validation"""
    config["fold"] = fold
    config["model_name"] = f'{config["split_type"]}_pretrained_epoch_{config["feature_epoch"]}_cv_fold_{fold}_{param_id}_seed_{config["seed"]}'
    
    # Get fold data and create dataloaders
    train_fold = fold_data[fold]['train']
    val_fold = fold_data[fold]['val']
    
    train_dataloader = create_dataloaders(
        train_fold['X'], train_fold['y'], train_fold['ids'],
        config["batch_size"], shuffle=True, num_workers=config["num_workers"]
    )
    val_dataloader = create_dataloaders(
        val_fold['X'], val_fold['y'], val_fold['ids'],
        config["batch_size"], shuffle=False, num_workers=config["num_workers"]
    )

    # Train and evaluate
    seed_everything(config["seed"])
    model = Vanilla_MLP_classifier(config)
    trainer = setup_wandb(config)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    return trainer.callback_metrics['val_average_precision_epoch'].item()

def prepare_data(config, split_type, epoch, step):
    # Load data
    train_labels = pd.read_csv(config["train_target_file"].format(split_type=split_type))
    test_labels = pd.read_csv(config["test_target_file"].format(split_type=split_type))
    
    train_features = pd.read_csv(config["train_BERT_features"].format(
        split_type=split_type, epoch=epoch, step=step))
    test_features = pd.read_csv(config["test_BERT_features"].format(
        split_type=split_type, epoch=epoch, step=step))

    # Prepare SMILES data
    train_set_SMILES = pd.DataFrame({'SMILES': train_labels.SMILES})
    test_set_SMILES = pd.DataFrame({'SMILES': test_labels.SMILES})

    # Merge and prepare features
    train_X = pd.merge(train_set_SMILES, train_features, on="SMILES", how="inner")
    train_X = train_X.drop(columns=['SMILES', 'valid']).values

    test_X = pd.merge(test_set_SMILES, test_features, on="SMILES", how="inner")
    test_X = test_X.drop(columns=['SMILES', 'valid']).values

    # Get target names
    target_names = train_labels.drop("SMILES", axis=1).columns.tolist()
    config["num_of_tasks"] = len(target_names)
    config["selected_tasks"] = target_names

    return (train_X, train_labels, train_set_SMILES, 
            test_X, test_labels, test_set_SMILES)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_hp_combination(hp_idx, config):
    """Get hyperparameter combination based on index"""
    hp_search = config["hp_search"]
    
    # Just return the hyperparameters at the given index
    return {
        "alpha": hp_search["alpha"][0],  # Lists will contain only one value each
        "gamma": hp_search["gamma"][0],
        "optm_l2_lambda": hp_search["optm_l2_lambda"][0],
        "dropout_p": hp_search["dropout_p"][0]
    }

def run_single_hp_combination(config, train_X, train_labels, train_set_SMILES, hp_params):
    """Run 5-fold CV for a single hyperparameter combination"""
    fold_data = get_stratified_folds(
        train_X,
        train_labels[config["selected_tasks"]].values,
        train_set_SMILES.values,
        num_of_folds=5,
        config=config
    )
    
    config.update(hp_params)
    param_id = f'alpha_{hp_params["alpha"]}_gamma_{hp_params["gamma"]}_l2_{hp_params["optm_l2_lambda"]}_dropout_{hp_params["dropout_p"]}'
    
    # Run 5-fold CV
    fold_scores = []
    for fold in range(5):
        score = run_cv_fold(config, fold_data, fold, param_id)
        fold_scores.append(score)
    
    return {
        'param_id': param_id,
        'params': hp_params,
        'scores': fold_scores,
        'mean_score': float(np.mean(fold_scores)),
        'std_score': float(np.std(fold_scores))
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Get hyperparameter combination index from environment
    hp_idx = int(os.environ.get('HP_IDX', 0))
    hp_params = get_hp_combination(hp_idx, config)
    print(hp_params)
    
    split_type = config["split_type"]
    data = prepare_data(config, split_type, epoch=config["feature_epoch"], step=config["feature_step"])
    train_X, train_labels, train_set_SMILES, _, _, _ = data
    
    # Run CV for this hyperparameter combination
    result = run_single_hp_combination(config, train_X, train_labels, train_set_SMILES, hp_params)
    
    # Save result
    result_dir = os.path.join(
        config["metadata_dir"], 
        f"hp_search_{split_type}_epoch_{config['feature_epoch']}"
    )
    os.makedirs(result_dir, exist_ok=True)
    
    result_path = os.path.join(
        result_dir,
        f"hp_{hp_idx}.json"
    )
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()