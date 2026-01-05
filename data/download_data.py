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
    # URLs for the dataset
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    ]
    
    # Column names as per UCI documentation
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    datasets = []
    
    for i, url in enumerate(urls):
        try:
            logger.info(f"Downloading dataset from {url}")
            df = pd.read_csv(
                url,
                names=columns,
                na_values='?',
                engine='python'
            )
            
            # Add source information
            df['source'] = ['cleveland', 'hungarian', 'switzerland', 'va'][i]
            datasets.append(df)
            
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    
    # Combine all datasets
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
    else:
        # Fallback to local file if download fails
        logger.warning("All downloads failed, using local data if available")
        combined_df = pd.DataFrame(columns=columns + ['source'])
    
    return combined_df

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
    
    # For categorical columns with missing values, use mode
    categorical_cols = ['ca', 'thal']
    for col in categorical_cols:
        if col in df_clean.columns:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            df_clean[col].fillna(mode_val, inplace=True)
    
    # For numerical columns, use median
    numerical_cols = ['trestbps', 'chol', 'thalach']
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    missing_after = df_clean.isnull().sum().sum()
    logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
    
    # Convert data types
    df_clean['sex'] = df_clean['sex'].astype(int)
    df_clean['cp'] = df_clean['cp'].astype(int)
    df_clean['fbs'] = df_clean['fbs'].astype(int)
    df_clean['restecg'] = df_clean['restecg'].astype(int)
    df_clean['exang'] = df_clean['exang'].astype(int)
    df_clean['slope'] = df_clean['slope'].astype(int)
    df_clean['ca'] = df_clean['ca'].astype(int)
    df_clean['thal'] = df_clean['thal'].astype(int)
    
    # Drop rows where target is still NaN
    df_clean = df_clean.dropna(subset=['target'])
    df_clean['target'] = df_clean['target'].astype(int)
    
    logger.info(f"Final dataset shape: {df_clean.shape}")
    logger.info(f"Class distribution:\n{df_clean['target'].value_counts()}")
    
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
    cleaned_df = clean_data(df)
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
    
    return cleaned_df

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
        
        # Generate basic statistics
        print("\n" + "="*50)
        print("FEATURE STATISTICS")
        print("="*50)
        print(cleaned_df.describe().T[['mean', 'std', 'min', 'max']])
        
        # Check for any remaining missing values
        if cleaned_df.isnull().sum().sum() == 0:
            logger.info("✓ Dataset successfully processed with no missing values")
        else:
            logger.warning(f"Dataset still has {cleaned_df.isnull().sum().sum()} missing values")
    else:
        logger.error("Failed to download dataset. Please check your internet connection.")
