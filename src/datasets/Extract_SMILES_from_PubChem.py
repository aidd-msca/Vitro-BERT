#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tqdm import tqdm
import pubchempy as pcp
import argparse
import sys
from pathlib import Path

def get_canonical_smiles(drug_name):
    """
    Retrieve canonical SMILES for a given drug name using PubChemPy.
    
    Args:
        drug_name (str): Name of the drug compound
    
    Returns:
        str or np.nan: Canonical SMILES string if found, np.nan otherwise
    """
    try:
        results = pcp.get_compounds(drug_name, 'name')
        if results:
            return results[0].canonical_smiles
        else:
            return np.nan
    except Exception as e:
        print(f"Error retrieving canonical SMILES for {drug_name}: {str(e)}")
        return np.nan

def main(input_file, output_file, sheet_name="Sheet1", compound_limit=None):
    """
    Main function to process compound names and generate SMILES.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output CSV file
        sheet_name (str): Name of the sheet in Excel file
        compound_limit (int, optional): Limit number of compounds to process
    """
    try:
        # Read input file
        print(f"Reading input file: {input_file}")
        tggateINHANDS = pd.read_excel(input_file, sheet_name=sheet_name)
        
        # Get unique compound names
        compound_list = tggateINHANDS.COMPOUND_NAME.unique().tolist()
        if compound_limit:
            compound_list = compound_list[:compound_limit]
        
        print(f"Processing {len(compound_list)} compounds...")
        
        # Create dictionary to store SMILES
        compound_smiles_dict = {}
        
        # Retrieve SMILES for each drug name
        for name in tqdm(compound_list):
            smiles = get_canonical_smiles(name)
            compound_smiles_dict[name] = smiles
        
        # Convert to DataFrame
        SMILES = pd.DataFrame(list(compound_smiles_dict.items()), 
                            columns=['COMPOUND_NAME', 'SMILES'])
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        print(f"Saving results to: {output_file}")
        SMILES.to_csv(output_file, index=False)
        
        # Print summary
        print(f"Successfully processed {len(SMILES)} compounds")
        print(f"Found SMILES for {SMILES['SMILES'].notna().sum()} compounds")
        print(f"Missing SMILES for {SMILES['SMILES'].isna().sum()} compounds")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate SMILES strings for compounds')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input Excel file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output CSV file')
    parser.add_argument('--sheet', type=str, default="Sheet1",
                      help='Sheet name in Excel file')
    parser.add_argument('--limit', type=int,
                      help='Limit number of compounds to process')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function
    main(args.input, args.output, args.sheet, args.limit)