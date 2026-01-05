#!/usr/bin/env python3
"""
Simple training script for CI/CD
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    print("Starting training process...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Create or load data
    data_file = 'data/processed/heart_disease_processed.csv'
    
    if os.path.exists(data_file):
        print(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
    else:
        print("Creating sample data...")
        np.random.seed(42)
        n_samples = 100
        data = {
            'age': np.random.normal(54, 9, n_samples).astype(int),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([0, 1, 2, 3], n_samples),
            'trestbps': np.random.normal(131, 18, n_samples).astype(int),
            'chol': np.random.normal(246, 52, n_samples).astype(int),
            'fbs': np.random.choice([0, 1], n_samples),
            'restecg': np.random.choice([0, 1, 2], n_samples),
            'thalach': np.random.normal(149, 23, n_samples).astype(int),
            'exang': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.exponential(1.0, n_samples).round(1),
            'slope': np.random.choice([0, 1, 2], n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
        }
        
        df = pd.DataFrame(data)
        df.to_csv(data_file, index=False)
        print(f"Created sample data with {n_samples} samples")
    
    print(f"Data shape: {df.shape}")
    
    # Train model
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Simple model training
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nTraining Results:")
    print(f"  Training accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    
    # Save model
    model_path = 'models/best_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Create a simple report
    report = f"""
# Model Training Report

## Dataset
- Total samples: {len(df)}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Features: {X.shape[1]}

## Model
- Algorithm: Random Forest
- Parameters: n_estimators=100, max_depth=10
- Training accuracy: {train_score:.4f}
- Test accuracy: {test_score:.4f}

## Files Generated
- Data: {data_file}
- Model: {model_path}
"""
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/training_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
