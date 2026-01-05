#!/usr/bin/env python3
"""
Standalone training script
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def setup_directories():
    """Create all necessary directories"""
    dirs = ['logs', 'models', 'reports', 'reports/figures', 
            'data/raw', 'data/processed', 'mlruns']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Created all directories")

def load_or_create_data():
    """Load existing data or create sample data"""
    data_path = 'data/processed/heart_disease_processed.csv'
    
    if os.path.exists(data_path):
        print(f"✓ Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("⚠ Creating sample data...")
        np.random.seed(42)
        n_samples = 200
        
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
        df.to_csv(data_path, index=False)
        print(f"✓ Created sample data with {n_samples} samples")
    
    return df

def train_models(df):
    """Train multiple models and select the best one"""
    print("\n" + "="*50)
    print("Training Models")
    print("="*50)
    
    # Prepare features and target
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope']
    X = df[features]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'cv_score': cv_scores.mean(),
            'test_accuracy': accuracy,
            'features': features
        }
        
        print(f"  CV Score: {cv_scores.mean():.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"  Precision: {report['weighted avg']['precision']:.4f}")
        print(f"  Recall: {report['weighted avg']['recall']:.4f}")
        print(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_result = results[best_model_name]
    
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"{'='*50}")
    
    return best_result, results

def save_results(best_result, all_results):
    """Save the trained model and results"""
    print("\nSaving results...")
    
    # Save best model
    model_data = {
        'model': best_result['model'],
        'scaler': best_result['scaler'],
        'features': best_result['features'],
        'test_accuracy': best_result['test_accuracy']
    }
    
    joblib.dump(model_data, 'models/best_model.pkl')
    print("✓ Saved best model to models/best_model.pkl")
    
    # Save all results
    import json
    
    results_dict = {}
    for name, result in all_results.items():
        results_dict[name] = {
            'cv_score': float(result['cv_score']),
            'test_accuracy': float(result['test_accuracy'])
        }
    
    with open('reports/model_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("✓ Saved results to reports/model_results.json")
    
    # Create training report
    report = f"""
# Heart Disease Prediction Model - Training Report

## Dataset Information
- Total Samples: {len(df)}
- Training Samples: {int(len(df) * 0.8)}
- Test Samples: {int(len(df) * 0.2)}
- Features: {len(best_result['features'])}
- Target Classes: 2 (0 = No Disease, 1 = Disease)

## Model Performance

### All Models
"""
    
    for name, result in all_results.items():
        report += f"""
**{name.replace('_', ' ').title()}**
- Cross-validation Score: {result['cv_score']:.4f}
- Test Accuracy: {result['test_accuracy']:.4f}
"""
    
    report += f"""
## Best Model
- **Algorithm**: {best_model_name.replace('_', ' ').title()}
- **Test Accuracy**: {best_result['test_accuracy']:.4f}
- **Cross-validation Score**: {best_result['cv_score']:.4f}

## Files Generated
1. `models/best_model.pkl` - Trained model with scaler
2. `reports/model_results.json` - Detailed results
3. `data/processed/heart_disease_processed.csv` - Dataset

## Usage
```python
import joblib

# Load the model
model_data = joblib.load('models/best_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# Prepare new data (same features as training)
new_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0]]
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)
