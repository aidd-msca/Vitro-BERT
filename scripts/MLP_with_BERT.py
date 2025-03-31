import os, sys
import json
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import wandb
import argparse

sys.path.append('/scratch/work/masooda1/ToxBERT/src')
from datasets.data_utils import get_stratified_folds, dataloader_for_numpy
from training.utils import setup_wandb
from molbert.models.MLP_models import Vanilla_MLP_classifier

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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

def get_external_validation_set(config, epoch, step):
    # Load data
    DM_labels = pd.read_csv(config["external_val_labels"])
    DM_features = pd.read_csv(config["external_val_features"].format(epoch=epoch, step=step))

    DM_labels = DM_labels.drop_duplicates(subset = ["SMILES"]).reset_index(drop = True)
    DM_features = DM_features.drop_duplicates(subset = ["SMILES"]).reset_index(drop = True)

    # Prepare SMILES data
    DM_SMILES = pd.DataFrame({'SMILES': DM_labels.SMILES})

    # Merge and prepare features
    DM_X = pd.merge(DM_SMILES, DM_features, on="SMILES", how="inner")
    DM_X = DM_X.drop(columns=['SMILES', 'valid']).values

    # Get target names
    target_names = DM_labels.drop("SMILES", axis=1).columns.tolist()
    config["num_of_tasks_ex_val"] = len(target_names)
    config["selected_tasks_ex_val"] = target_names

    return (DM_X, DM_labels, DM_SMILES)

def create_dataloaders(X, y, ids, batch_size, shuffle=True, num_workers=4):
    return DataLoader(
        dataloader_for_numpy(X, y, x_type='Fingerprints'),
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=shuffle,
    )

def convert_tensors_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_python(item) for item in obj]
    return obj

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

def run_hyperparameter_search(config, train_X, train_labels, train_set_SMILES):
    """Run hyperparameter search with single seed 5-fold CV"""
    cv_results = {}
    all_scores = {}  # Store scores for all parameter combinations
    
    # Generate stratified folds with fixed seed from config
    fold_data = get_stratified_folds(
        train_X,
        train_labels[config["selected_tasks"]].values,
        train_set_SMILES.values,
        num_of_folds=5,
        config=config
    )
    
    # Iterate through hyperparameter combinations
    for alpha in config["hp_search"]["alpha"]:
        for gamma in config["hp_search"]["gamma"]:
            for optm_l2_lambda in config["hp_search"]["optm_l2_lambda"]:
                for dropout_p in config["hp_search"]["dropout_p"]:
                    param_id = f'alpha_{alpha}_gamma_{gamma}_l2_{optm_l2_lambda}_dropout_{dropout_p}'
                    cv_results[param_id] = []
                    
                    # Update config with current parameters
                    config.update({
                        "alpha": alpha,
                        "gamma": gamma,
                        "optm_l2_lambda": optm_l2_lambda,
                        "dropout_p": dropout_p
                    })

                    # Run 5-fold CV
                    fold_scores = []
                    for fold in range(5):
                        score = run_cv_fold(config, fold_data, fold, param_id)
                        fold_scores.append(score)
                        
                    cv_results[param_id] = fold_scores
                    all_scores[param_id] = np.mean(fold_scores)
    
    # Find best hyperparameters
    best_param_id = max(all_scores.keys(), key=lambda k: all_scores[k])
    best_score = all_scores[best_param_id]
    best_std = np.std(cv_results[best_param_id])
    
    # Parse best parameters
    alpha, gamma, optm_l2_lambda, dropout_p = best_param_id.split('_')[1::2]
    best_params = {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "optm_l2_lambda": float(optm_l2_lambda),
        "dropout_p": float(dropout_p)
    }
    
    return best_params, cv_results, best_score, best_std

def train_final_model(config, train_X, train_y, train_ids, test_X, test_y, test_ids, seed, ex_val = None):
    """Train final model with given seed"""
    config["seed"] = seed
    config["model_name"] = f"{config['split_type']}_pretrained_epoch_{config['feature_epoch']}_final_model_seed_{seed}"
    
    # Create dataloaders
    train_dataloader = create_dataloaders(
        train_X, train_y, train_ids,
        config["batch_size"], shuffle=True, num_workers=config["num_workers"]
    )
    test_dataloader = create_dataloaders(
        test_X, test_y, test_ids,
        config["batch_size"], shuffle=False, num_workers=config["num_workers"]
    )
    
    # Train model
    seed_everything(seed)
    model = Vanilla_MLP_classifier(config)
    trainer = setup_wandb(config)
    trainer.fit(model, train_dataloader, test_dataloader)
    
    results = convert_tensors_to_python(trainer.callback_metrics)
    
    # ext val set
    if config["external_val"]:
        (ex_val_X, ex_val_labels, ex_val_SMILES) = ex_val
        ex_val_X, ex_val_labels, ex_val_SMILES = ex_val_X, ex_val_labels[config["selected_tasks_ex_val"]].values, ex_val_SMILES.values,
        
        ex_test_dataloader = create_dataloaders(
            ex_val_X, ex_val_labels, ex_val_SMILES,
            config["batch_size"], shuffle=False, num_workers=config["num_workers"])
        
        ex_val_results = {}
        score_list = model.compute_metrics_external_val(ex_test_dataloader, config)
        metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision', 'ECE_score', 'ACE_score']
        for i, score in enumerate(score_list):
            ex_val_results[f'ex_val_{metric[i]}_epoch'] = score
        results.update(ex_val_results)

    return results

def parse_args():
    parser = argparse.ArgumentParser(description='MLP training with BERT features')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the config file')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    split_type = config["split_type"]

    # Create results directory if it doesn't exist
    os.makedirs(config["metadata_dir"], exist_ok=True)

    # Prepare data
    data = prepare_data(config, split_type, epoch=config["feature_epoch"], step=config["feature_step"])
    (train_X, train_labels, train_set_SMILES,
     test_X, test_labels, test_set_SMILES) = data

    # Run hyperparameter search with fixed seed from config
    best_params, cv_results, best_score, best_std = run_hyperparameter_search(
        config, train_X, train_labels, train_set_SMILES
    )
    
    # Update config with best parameters
    config.update(best_params)
    config["Final_model"] = True
    print("##########################")
    print(config)
    print("##########################")

    if config["external_val"]:
        ex_val_data = get_external_validation_set(config, config["feature_epoch"], config["feature_step"])
    else:
        ex_val_data = None

    # Train final models with different seeds
    final_results = {}
    for seed in range(5):  # Seeds 0 to 4
        metrics = train_final_model(config,data,
            train_X, train_labels[config["selected_tasks"]].values, train_set_SMILES.values,
            test_X, test_labels[config["selected_tasks"]].values, test_set_SMILES.values,
            seed, ex_val_data
        )
        final_results[f'seed_{seed}'] = metrics

    # Save detailed results with split type and feature epoch in filename
    result_path = os.path.join(
        config["metadata_dir"], 
        f"final_results_{split_type}_pretrained_epoch_{config['feature_epoch']}.json"
    )
    
    # Update results dictionary to include feature epoch
    results = {
        'split_type': split_type,
        'feature_epoch': config['feature_epoch'],
        'best_parameters': best_params,
        'cv_results': cv_results,
        'cv_score': {
            'mean': float(best_score),
            'std': float(best_std)
        },
        'final_models': final_results
    }

    # Save results
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary with feature epoch
    print(f"\nResults for {split_type} at epoch {config['feature_epoch']}:")
    print(f"Best parameters: {best_params}")
    print(f"CV score: {best_score:.4f} ± {best_std:.4f}")
    
    # Print final model results
    print("\nFinal Model Results:")
    for seed in range(5):
        average_precision_epoch = final_results[f'seed_{seed}']['val_average_precision_epoch']
        print(f"Seed {seed}: val_average_precision_epoch = {average_precision_epoch:.4f}")

if __name__ == "__main__":
    main()