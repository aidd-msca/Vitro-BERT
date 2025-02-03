import pandas as pd
import os

def load_data(base_folder):
    """Load and merge TG-GATES data with SMILES"""
    # Load labels from Excel
    tggate_inhands = pd.read_excel(
        base_folder + "tx2c00378_si_001.xlsx", 
        sheet_name="ALL_data"
    )
    # Load SMILES data
    tggate_smiles = pd.read_csv(base_folder + "TG_GATES_SMILES.csv")
    
    # Filter and merge data
    tggate_smiles = tggate_smiles[tggate_smiles.COMPOUND_NAME.isin(tggate_inhands.COMPOUND_NAME)]
    merged_data = pd.merge(tggate_smiles, tggate_inhands, how='left', on='COMPOUND_NAME')
    
    return merged_data

def preprocess_data(df):
    """Preprocess the merged dataset"""
    # Rename and fill missing values
    df.rename(columns={"Finding: Final INHANDS nomenclature": "Findings"}, inplace=True)
    df["Findings"].fillna('NonToxic', inplace=True)
    
    # Select relevant columns
    selected_columns = ['COMPOUND_NAME', 'SMILES', 'Dose_Level', 'Time', 'Findings']
    df = df[selected_columns]
    
    # Create DILI labels
    df.loc[df['Findings'].notna(), 'DILI_labels'] = 1
    df.loc[df['Findings'].isna(), 'DILI_labels'] = 0
    
    return df

def create_binary_matrix(df):
    """Create binary matrix of findings"""
    # Group by compound and create binary indicators
    grouped_data = df.groupby(['COMPOUND_NAME', 'SMILES', 'Findings']).DILI_labels.sum().reset_index()
    grouped_data['DILI_labels'] = grouped_data['DILI_labels'].astype(bool)
    grouped_data.loc[grouped_data['Findings'] == "NonToxic", 'DILI_labels'] = False
    
    # Pivot and create binary matrix
    binary_matrix = grouped_data.pivot(
        index=['COMPOUND_NAME', 'SMILES'],
        columns='Findings',
        values="DILI_labels"
    ).rename_axis(None, axis=1).reset_index()
    
    binary_matrix = binary_matrix.fillna(0) * 1
    return binary_matrix

def filter_frequent_findings(df, min_frequency=6):
    """Filter findings based on minimum frequency"""
    findings_freq = df.iloc[:, 2:].sum(axis=0).reset_index()
    findings_freq.columns = ["Finding", "Frequency"]
    
    selected_findings = findings_freq[
        findings_freq.Frequency > min_frequency
    ].sort_values(by="Frequency").reset_index(drop=True)
    
    selected_columns = ["COMPOUND_NAME", "SMILES"] + selected_findings.Finding.tolist()
    return df[selected_columns]

def main(input_folder, output_path):
    # Process data
    raw_data = load_data(input_folder)
    processed_data = preprocess_data(raw_data)
    binary_matrix = create_binary_matrix(processed_data)
    final_data = filter_frequent_findings(binary_matrix)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    final_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process histopathology data')
    parser.add_argument('--input', type=str, required=True,
                      help='Input folder path containing raw data')
    parser.add_argument('--output', type=str, required=True,
                      help='Output path for the processed CSV file')
    
    args = parser.parse_args()
    main(args.input, args.output)