#!/usr/bin/env python3
"""
Data acquisition script for Heart Disease UCI Dataset
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset() -> pd.DataFrame:
    """
    Download and preprocess the Heart Disease dataset from UCI repository
    
    Returns:
        pd.DataFrame: Cleaned dataset with proper column names
    """
    # URLs for the dataset - use only Cleveland dataset for consistency
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names as per UCI documentation
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        logger.info(f"Downloading dataset from {url}")
        df = pd.read_csv(
            url,
            names=columns,
            na_values='?',
            engine='python'
        )
        
        # Add source information
        df['source'] = 'cleveland'
        
        logger.info(f"Downloaded {len(df)} samples from Cleveland dataset")
        return df
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        # Return empty dataframe with correct columns
        return pd.DataFrame(columns=columns + ['source'])

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    logger.info("Cleaning dataset...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Convert target to binary (0 = no disease, 1 = disease)
    df_clean['target'] = df_clean['target'].apply(
        lambda x: 1 if x > 0 else 0 if pd.notna(x) else np.nan
    )
    
    # Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    
    # Define categorical and numerical columns
    categorical_cols = ['ca', 'thal', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # First, convert problematic columns to numeric, forcing errors to NaN
    for col in ['ca', 'thal']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # For categorical columns, fill NaN with mode
    for col in categorical_cols:
        if col in df_clean.columns:
            # Check if column has any non-NaN values
            if df_clean[col].notna().any():
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                df_clean[col] = df_clean[col].fillna(mode_val)
            else:
                # If all values are NaN, fill with 0
                df_clean[col] = df_clean[col].fillna(0)
    
    # For numerical columns, fill NaN with median
    for col in numerical_cols:
        if col in df_clean.columns:
            if df_clean[col].notna().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                df_clean[col] = df_clean[col].fillna(0)
    
    # Now safely convert to integer types
    for col in categorical_cols:
        if col in df_clean.columns:
            # Check if all values are finite before converting
            if df_clean[col].notna().all():
                df_clean[col] = df_clean[col].astype(int)
            else:
                logger.warning(f"Column {col} still has NaN values after filling")
                # Fill any remaining NaN with 0 and convert
                df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    # Convert numerical columns to float (some may have decimal values)
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(float)
    
    # Drop rows where target is still NaN
    df_clean = df_clean.dropna(subset=['target'])
    df_clean['target'] = df_clean['target'].astype(int)
    
    missing_after = df_clean.isnull().sum().sum()
    logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
    
    logger.info(f"Final dataset shape: {df_clean.shape}")
    logger.info(f"Class distribution:\n{df_clean['target'].value_counts()}")
    
    # Verify no NaN values remain
    if df_clean.isnull().sum().sum() > 0:
        logger.error(f"Dataset still has {df_clean.isnull().sum().sum()} NaN values")
        logger.error(f"Columns with NaN: {df_clean.columns[df_clean.isnull().any()].tolist()}")
    
    return df_clean

def save_data(df: pd.DataFrame, raw_path: str, processed_path: str):
    """
    Save raw and processed data
    
    Args:
        df (pd.DataFrame): Dataset to save
        raw_path (str): Path to save raw data
        processed_path (str): Path to save processed data
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    # Save raw data
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to {raw_path}")
    
    # Clean and save processed data
    try:
        cleaned_df = clean_data(df)
        
        # Final check for NaN values
        if cleaned_df.isnull().sum().sum() > 0:
            logger.error(f"Processed data still has NaN values. Filling with 0.")
            cleaned_df = cleaned_df.fillna(0)
        
        cleaned_df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
        
        # Print dataset statistics
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total samples: {len(cleaned_df)}")
        print(f"Features: {len(cleaned_df.columns) - 1}")  # Excluding target
        print(f"Positive cases (disease): {cleaned_df['target'].sum()}")
        print(f"Negative cases (no disease): {len(cleaned_df) - cleaned_df['target'].sum()}")
        print(f"Class ratio (positive/negative): {cleaned_df['target'].mean():.2%}")
        print(f"Missing values in processed data: {cleaned_df.isnull().sum().sum()}")
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        # Save raw data as fallback
        df.to_csv(processed_path, index=False)
        logger.info(f"Saved raw data to {processed_path} as fallback")
        return df

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'heart_disease_raw.csv')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'heart_disease_processed.csv')
    
    # Download and process data
    logger.info("Starting data download and processing...")
    raw_df = download_dataset()
    
    if not raw_df.empty:
        cleaned_df = save_data(raw_df, RAW_DATA_PATH, PROCESSED_DATA_PATH)
        
        if cleaned_df is not None:
            # Generate basic statistics
            print("\n" + "="*50)
            print("FEATURE STATISTICS")
            print("="*50)
            
            # Only show numeric columns in describe
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(cleaned_df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']].round(2))
            else:
                print("No numeric columns found")
            
            # Check for any remaining missing values
            missing_count = cleaned_df.isnull().sum().sum()
            if missing_count == 0:
                logger.info("✓ Dataset successfully processed with no missing values")
            else:
                logger.warning(f"Dataset still has {missing_count} missing values")
                # Show which columns have missing values
                missing_cols = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
                logger.warning(f"Columns with missing values: {missing_cols}")
    else:
        logger.error("Failed to download dataset. Please check your internet connection.")
        # Create empty processed file to avoid downstream errors
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        pd.DataFrame().to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info("Created empty processed file to allow pipeline to continue")
