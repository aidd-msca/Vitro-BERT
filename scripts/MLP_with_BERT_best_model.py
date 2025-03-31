import os
import sys
import json
import glob
import argparse
from MLP_with_BERT import (
    load_config, 
    prepare_data, 
    train_final_model,
    get_external_validation_set
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train final MLP model with best hyperparameters')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the config file')
    parser.add_argument('--split-type', type=str, required=True,
                      help='Split type (e.g., Split_Structure)')
    parser.add_argument('--feature-epoch', type=str, required=True,
                      help='Feature epoch (e.g., init, 0, 4, 9)')
    return parser.parse_args()

def find_best_hyperparameters(results_dir, split_type, feature_epoch):
    """Find best hyperparameters from all CV results"""
    pattern = os.path.join(results_dir, f"hp_search_{split_type}_epoch_{feature_epoch}", "hp_*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        raise FileNotFoundError(f"No hyperparameter search results found matching pattern: {pattern}")
    
    best_score = -float('inf')
    best_params = None
    
    for file in result_files:
        with open(file, 'r') as f:
            result = json.load(f)
            mean_score = result['mean_score']
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = result['params']
    
    if best_params is None:
        raise ValueError("Could not find valid hyperparameters in results")
    
    return best_params, best_score

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if results already exist
    result_path = os.path.join(
        config["metadata_dir"], 
        f"best_model_results_{args.split_type}_pretrained_epoch_{args.feature_epoch}.json"
    )
    
    if os.path.exists(result_path):
        print(f"\nResults already exist for split_type={args.split_type}, feature_epoch={args.feature_epoch}")
        print(f"Skipping this combination. File: {result_path}")
        return
    
    # Override split_type and feature_epoch from arguments
    config["split_type"] = args.split_type
    config["feature_epoch"] = args.feature_epoch
    
    # Find best hyperparameters from previous search results
    best_params, best_score = find_best_hyperparameters(
        config["metadata_dir"],
        args.split_type,
        args.feature_epoch
    )
    
    print(f"\nBest hyperparameters found (score: {best_score:.4f}):")
    print(json.dumps(best_params, indent=2))
    
    # Update config with best parameters
    config.update(best_params)
    config["Final_model"] = True

    print("##########################")
    print(config)
    print("##########################")
    
    # Prepare data
    data = prepare_data(config, args.split_type, epoch=args.feature_epoch, step=config["feature_step"])
    (train_X, train_labels, train_set_SMILES,
     test_X, test_labels, test_set_SMILES) = data
    
    # Load external validation data if needed
    if config["external_val"]:
        ex_val_data = get_external_validation_set(config, args.feature_epoch, config["feature_step"])
    else:
        ex_val_data = None
    
    # Train final models with different seeds
    final_results = {}
    for seed in range(5):  # Seeds 0 to 4
        print(f"\nTraining model with seed {seed}...")
        metrics = train_final_model(
            config,
            train_X, train_labels[config["selected_tasks"]].values, train_set_SMILES.values,
            test_X, test_labels[config["selected_tasks"]].values, test_set_SMILES.values,
            seed, ex_val_data
        )
        final_results[f'seed_{seed}'] = metrics
    
    # Save results
    result_path = os.path.join(
        config["metadata_dir"], 
        f"best_model_results_{args.split_type}_pretrained_epoch_{args.feature_epoch}.json"
    )
    
    results = {
        'split_type': args.split_type,
        'feature_epoch': args.feature_epoch,
        'parameters': best_params,
        'best_cv_score': best_score,
        'final_models': final_results
    }
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final model results
    print("\nFinal Model Results:")
    for seed in range(5):
        average_precision_epoch = final_results[f'seed_{seed}']['val_average_precision_epoch']
        print(f"Seed {seed}: val_average_precision_epoch = {average_precision_epoch:.4f}")

if __name__ == "__main__":
    main()