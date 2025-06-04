import os
import sys
from pathlib import Path

# Get project root from environment variable or use default
PROJECT_ROOT = os.getenv('TOXBERT_ROOT', str(Path(__file__).parent.parent))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from src.datasets.data_utils import normalize_smiles_parallel

def filter_assays(df: pd.DataFrame, min_pos_neg_per_assay: int = 10) -> pd.DataFrame:
    """
    Filter assays based on minimum number of positive and negative samples.
    
    Args:
        df (pd.DataFrame): Input DataFrame with assays
        min_pos_neg_per_assay (int): Minimum number of positive/negative samples required per assay
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    pos_counts = (df.iloc[:, 1:] == 1).sum(axis=0)
    neg_counts = (df.iloc[:, 1:] == 0).sum(axis=0)
    valid_assays = (pos_counts >= min_pos_neg_per_assay) & (neg_counts >= min_pos_neg_per_assay)
    selected_columns = ['smiles', 'Normalized_SMILES'] + valid_assays[valid_assays].index.tolist()
    return df[selected_columns]

def plot_distributions(filtered_df: pd.DataFrame, pos_counts_assays: pd.Series, 
                      neg_counts_assays: pd.Series, valid_assays: pd.Series):
    """
    Create visualization plots for assay and drug distributions.
    
    Args:
        filtered_df (pd.DataFrame): Filtered DataFrame
        pos_counts_assays (pd.Series): Counts of positive samples per assay
        neg_counts_assays (pd.Series): Counts of negative samples per assay
        valid_assays (pd.Series): Boolean mask for valid assays
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot assay distributions
    pos_assays = pos_counts_assays[valid_assays].sort_values(ascending=False)
    neg_assays = neg_counts_assays[valid_assays].sort_values(ascending=False)
    
    ax1.plot(pos_assays.values, label='Positive', color='red')
    ax1.plot(neg_assays.values, label='Negative', color='green')
    ax1.set_yscale('log')
    ax1.set_title('Distribution of Positive/Negative Samples per Assay')
    ax1.set_xlabel('Assay Index (sorted by count)')
    ax1.set_ylabel('Count (log scale)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot drug distributions
    pos_drugs = (filtered_df.iloc[:, 2:] == 1).sum(axis=1).sort_values(ascending=False)
    neg_drugs = (filtered_df.iloc[:, 2:] == 0).sum(axis=1).sort_values(ascending=False)
    measurements = (filtered_df.iloc[:, 2:] != -1).sum(axis=1).sort_values(ascending=False)
    
    ax2.plot(measurements.values, label='total_measurements', color='blue')
    ax2.plot(pos_drugs.values, label='Positive', color='red')
    ax2.plot(neg_drugs.values, label='Negative', color='green')
    ax2.set_yscale('log')
    ax2.set_title('Distribution of Positive/Negative Results per Drug')
    ax2.set_xlabel('Drug Index (sorted by count)')
    ax2.set_ylabel('Count (log scale)')
    ax2.legend()
    ax2.grid(True)

def read_data_file(file_path: str, csv_sep: str = ',') -> pd.DataFrame:
    """
    Read data from various file formats (csv, parquet, tar.gz).
    For tar.gz files, specifically looks for 'chembl20.parquet'.
    
    Args:
        file_path (str): Path to the input file
        csv_sep (str): Separator for CSV files (default: ',')
        
    Returns:
        pd.DataFrame: Loaded DataFrame
        
    Raises:
        ValueError: If file format is not supported or chembl20.parquet not found
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Handle regular files
    if file_path.suffix == '.parquet':
        print(f"Reading parquet file: {file_path}")
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        print(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path, sep=csv_sep)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats are: .parquet, .csv, .tar.gz")
    
    print(f"Successfully loaded data with shape: {df.shape}")
    return df

def filter_invivo_compounds(invitro_df: pd.DataFrame, invivo_path: str, invivo_smiles_column: str) -> pd.DataFrame:
    """
    Filter out compounds that are present in invivo dataset from invitro data.
    
    Args:
        invitro_df (pd.DataFrame): Invitro DataFrame with Normalized_SMILES
        invivo_path (str): Path to invivo SMILES file
        invivo_smiles_column (str): Name of the column containing SMILES strings in invivo data
        
    Returns:
        pd.DataFrame: Filtered DataFrame with invivo compounds removed
    """
    print("\nFiltering out invivo compounds from invitro data...")
    
    # Load and process invivo SMILES
    invivo_smiles = pd.read_excel(invivo_path)
    if invivo_smiles_column not in invivo_smiles.columns:
        raise ValueError(f"Invivo SMILES column '{invivo_smiles_column}' not found in data. Available columns: {list(invivo_smiles.columns)}")
    
    print(f"Invivo unique SMILES: {invivo_smiles[invivo_smiles_column].nunique()}")
    
    # Normalize invivo SMILES
    normalized_smiles_list = normalize_smiles_parallel(invivo_smiles[invivo_smiles_column].tolist())
    invivo_smiles['Normalized_SMILES'] = normalized_smiles_list
    invivo_compounds = invivo_smiles.Normalized_SMILES.unique().tolist()
    
    # Filter out invivo compounds
    filtered_data = invitro_df[~invitro_df['Normalized_SMILES'].isin(invivo_compounds)]
    
    # Print statistics
    print(f"Original invitro dataset size: {len(invitro_df)}")
    print(f"Number of invivo SMILES to remove: {len(invivo_compounds)}")
    print(f"Filtered invitro dataset size: {len(filtered_data)}")
    print(f"Number of rows removed: {len(invitro_df) - len(filtered_data)}")
    
    return filtered_data

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process invitro data for ToxBERT pretraining')
    
    parser.add_argument(
        '--invitro_input_path',
        type=str,
        required=True,
        help='Path to invitro input file (supported formats: .parquet, .csv, .tar.gz)'
    )
    
    parser.add_argument(
        '--invivo_input_path',
        type=str,
        required=True,
        help='Path to invivo input file (supported formats: .xlsx)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save processed data (will be saved as parquet)'
    )
    
    parser.add_argument(
        '--smiles_column',
        type=str,
        default='smiles',
        help='Name of the column containing SMILES strings (default: smiles)'
    )
    
    parser.add_argument(
        '--min_pos_neg_per_assay',
        type=int,
        default=10,
        help='Minimum number of positive/negative samples required per assay (default: 10)'
    )
    
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Save distribution plots to output directory'
    )
    
    parser.add_argument(
        '--csv_sep',
        type=str,
        default=',',
        help='Separator for CSV files (default: ,)'
    )
    
    parser.add_argument(
        '--plot_path',
        type=str,
        default=None,
        help='Path to save the distribution plots (default: same directory as output file)'
    )
    
    parser.add_argument(
        '--invivo_smiles_column',
        type=str,
        default='SMILES',
        help='Name of the column containing SMILES strings in invivo data (default: SMILES)'
    )
    
    parser.add_argument(
        '--invitro_smiles_column',
        type=str,
        default='smiles',
        help='Name of the column containing SMILES strings in invitro data (default: smiles)'
    )
    
    return parser.parse_args()

def main():
    """Main function to process invitro data."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load invitro data
    print(f"Loading invitro data from {args.invitro_input_path}")
    try:
        invitro_data = read_data_file(args.invitro_input_path, csv_sep=args.csv_sep)
    except Exception as e:
        print(f"Error loading invitro data: {str(e)}")
        return
    
    # Ensure required columns exist
    if args.invitro_smiles_column not in invitro_data.columns:
        print(f"Error: Invitro SMILES column '{args.invitro_smiles_column}' not found in data. Available columns: {list(invitro_data.columns)}")
        return
    
    # Rename SMILES column to standard name for processing
    invitro_data = invitro_data.rename(columns={args.invitro_smiles_column: 'smiles'})
    
    # Normalize SMILES
    print(f"Original unique SMILES: {invitro_data.smiles.nunique()}")
    normalized_smiles_list = normalize_smiles_parallel(invitro_data.smiles.tolist())
    invitro_data['Normalized_SMILES'] = normalized_smiles_list
    print(f"Normalized unique SMILES: {invitro_data.Normalized_SMILES.nunique()}")
    
    # Remove duplicates
    invitro_data = invitro_data.drop_duplicates(subset=["Normalized_SMILES"]).reset_index(drop=True)
    
    # Filter out invivo compounds
    before_filtering = invitro_data.Normalized_SMILES.nunique()
    invitro_data = filter_invivo_compounds(invitro_data, args.invivo_input_path, args.invivo_smiles_column)
    after_filtering = invitro_data.Normalized_SMILES.nunique()
    print(f"Number of invivo compounds removed: {before_filtering - after_filtering}")

    
    # Filter assays and create visualizations
    filtered_df = filter_assays(invitro_data, min_pos_neg_per_assay=args.min_pos_neg_per_assay)
    pos_counts_assays = (invitro_data.iloc[:, 1:] == 1).sum(axis=0)
    neg_counts_assays = (invitro_data.iloc[:, 1:] == 0).sum(axis=0)
    valid_assays = (pos_counts_assays >= args.min_pos_neg_per_assay) & (neg_counts_assays >= args.min_pos_neg_per_assay)
    
    # Print statistics
    print(f"Original number of assays: {invitro_data.shape[1]-2}")
    print(f"Number of assays with ≥{args.min_pos_neg_per_assay} pos and neg: {len(filtered_df.columns)-2}")
    print(f"Shape of filtered dataset: {filtered_df.shape}")
    
    # Create and optionally save visualizations
    plot_distributions(filtered_df, pos_counts_assays, neg_counts_assays, valid_assays)
    if args.save_plots:
        # Determine plot save location
        if args.plot_path:
            plot_path = Path(args.plot_path)
            # Create directory if it doesn't exist
            plot_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            plot_path = output_dir / "distribution_plots.png"
        
        plt.savefig(plot_path)
        print(f"Saved distribution plots to {plot_path}")
    
    # Save processed data
    print(f"Saving processed data to {args.output_path}")
    filtered_df.to_parquet(args.output_path)

if __name__ == "__main__":
    main()