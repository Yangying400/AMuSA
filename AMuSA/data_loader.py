"""
data_loader.py
Module for loading and preprocessing mutation and signature data for the signature classifier.
Handles data alignment, transformation, and standardization.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import sys

def load_data(mutation_file, exposure_file, signature_file, scaler=None, train=True):
    # Load mutation data
    try:
        mutation_df = pd.read_csv(mutation_file, index_col=0)
        mutation_df = mutation_df.T
    except Exception as e:
        print(f"Error loading mutation file: {e}")
        sys.exit(1)
    
    # Load exposure data
    exposure_df = None
    if exposure_file is not None:
        try:
            exposure_df = pd.read_csv(exposure_file, index_col=0)
            exposure_df = exposure_df.T
        except Exception as e:
            print(f"Error loading exposure file: {e}")
            sys.exit(1)
    
    # Load signature matrix
    signature_df = None
    signature_names = None
    
    if signature_file is not None:
        try:
            signature_df = pd.read_csv(signature_file, index_col=0)
            signature_names = signature_df.columns.tolist()
        except Exception as e:
            print(f"Error loading signature file: {e}")
            sys.exit(1)
    
    # Ensure data alignment
    if exposure_df is not None and not mutation_df.index.equals(exposure_df.index):
        common_indices = mutation_df.index.intersection(exposure_df.index)
        if len(common_indices) > 0:
            print(f"Using {len(common_indices)} common samples.")
            mutation_df = mutation_df.loc[common_indices]
            exposure_df = exposure_df.loc[common_indices]
        else:
            print("Error: No common samples between mutation and exposure data.")
            sys.exit(1)
    
    # Standardize mutation data
    if train:
        # Apply log transformation
        mutation_df_transformed = np.log1p(mutation_df)
        
        # Remove outliers
        q_low = mutation_df_transformed.quantile(0.01)
        q_high = mutation_df_transformed.quantile(0.99)
        mutation_df_filtered = mutation_df_transformed.clip(lower=q_low, upper=q_high, axis=1)
        
        # Apply power transformation
        scaler = PowerTransformer(method='yeo-johnson')
        mutation_scaled = scaler.fit_transform(mutation_df_filtered)
    else:
        if scaler is None:
            print("Error: Scaler must be provided for test data.")
            sys.exit(1)
        
        # Apply same preprocessing to test data
        mutation_df_transformed = np.log1p(mutation_df)
        mutation_scaled = scaler.transform(mutation_df_transformed)
    
    # Convert exposure values to binary (active/inactive)
    exposure_binary = None
    if exposure_df is not None:
        exposure_values = exposure_df.values
        exposure_threshold = 0.05
        exposure_binary = (exposure_values > exposure_threshold).astype(float)
    
    # Prepare return values
    return mutation_scaled, exposure_binary, signature_names, mutation_df.index, scaler