"""
Unit tests for data processing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = pd.DataFrame(
            {
                "age": [63, 37, 41, 50, 56],
                "sex": [1, 1, 0, 1, 0],
                "cp": [3, 2, 1, 0, 3],
                "trestbps": [145, 130, 130, 120, 120],
                "chol": [233, 250, 204, 236, 256],
                "fbs": [1, 0, 0, 0, 1],
                "restecg": [0, 1, 0, 1, 1],
                "thalach": [150, 187, 172, 160, 142],
                "exang": [0, 0, 0, 1, 1],
                "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.0],
                "slope": [0, 0, 2, 1, 2],
                "target": [1, 0, 0, 1, 0],
            }
        )
        return data
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values for testing"""
        data = pd.DataFrame(
            {
                "age": [63, 37, np.nan, 50, 56],
                "sex": [1, 1, 0, 1, 0],
                "cp": [3, 2, 1, 0, 3],
                "trestbps": [145, 130, 130, np.nan, 120],
                "chol": [233, 250, 204, 236, 256],
                "fbs": [1, 0, 0, 0, 1],
                "restecg": [0, 1, 0, 1, 1],
                "thalach": [150, 187, 172, 160, 142],
                "exang": [0, 0, 0, 1, 1],
                "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.0],
                "slope": [0, 0, 2, 1, 2],
                "target": [1, 0, 0, 1, 0],
            }
        )
        return data
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        return DataPreprocessor()
    
    def test_handle_missing_values(self, preprocessor, sample_data_with_missing):
        """Test missing value handling"""
        processed_data = preprocessor.handle_missing_values(sample_data_with_missing)
        
        # Check no NaN values remain
        assert processed_data.isnull().sum().sum() == 0
        
        # Check numerical columns filled with median
        assert processed_data["age"].isnull().sum() == 0
        assert processed_data["trestbps"].isnull().sum() == 0
        
        # Verify values were filled
        assert processed_data["age"][2] == processed_data["age"].median()
    
    def test_convert_target_to_binary(self, preprocessor, sample_data):
        """Test target conversion"""
        # Create data with non-binary target
        data = sample_data.copy()
        data["target"] = [2, 0, 0, 3, 1]  # Some values > 1
        
        processed_data = preprocessor.convert_target_to_binary(data)
        
        # Check target values are binary
        assert set(processed_data["target"].unique()).issubset({0, 1})
        
        # Check conversion logic
        assert processed_data.loc[0, "target"] == 1  # 2 -> 1
        assert processed_data.loc[3, "target"] == 1  # 3 -> 1
    
    def test_feature_scaling(self, preprocessor, sample_data):
        """Test feature scaling"""
        numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        processed_data = preprocessor.scale_features(sample_data, numerical_features)
        
        # Check that scaled columns were added
        for feature in numerical_features:
            scaled_col = f"{feature}_scaled"
            assert scaled_col in processed_data.columns
            
            # Check that scaling was applied (values are different from original)
            if len(sample_data[feature].unique()) > 1:  # Only check if not constant
                assert not processed_data[scaled_col].equals(sample_data[feature])
    
    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Test categorical feature encoding"""
        categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
        processed_data = preprocessor.encode_categorical(sample_data, categorical_features)
        
        # Check that encoded columns were added
        # Note: With drop='first', we get n_categories-1 columns per feature
        for feature in categorical_features:
            # Count how many columns start with this feature name
            encoded_cols = [col for col in processed_data.columns if col.startswith(f"{feature}_")]
            
            # For binary features (sex, fbs, exang), we should get 1 column
            if feature in ["sex", "fbs", "exang"]:
                assert len(encoded_cols) >= 1
            # For multi-category features, we should get at least 1
            else:
                assert len(encoded_cols) >= 1
    
    def test_full_preprocessing_pipeline(self, preprocessor, sample_data_with_missing):
        """Test complete preprocessing pipeline"""
        config = {
            "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope"],
            "target": "target",
        }
        
        X_processed, y_processed = preprocessor.preprocess(sample_data_with_missing, config)
        
        # Check output shapes
        assert X_processed.shape[0] == sample_data_with_missing.shape[0]
        assert len(y_processed) == sample_data_with_missing.shape[0]
        
        # Check no NaN values
        assert not X_processed.isnull().any().any()
        assert not y_processed.isnull().any()
        
        # Check that preprocessor was fitted
        assert preprocessor._is_fitted
        
        # Check feature names
        assert len(X_processed.columns) > 0
        
        # Verify we can transform new data
        new_data = sample_data_with_missing.iloc[:2].copy()
        X_new, y_new = preprocessor.preprocess(new_data, config)
        assert X_new.shape[0] == 2
    
    def test_preprocess_with_minimal_features(self, preprocessor, sample_data):
        """Test preprocessing with minimal feature set"""
        config = {
            "numerical_features": ["age"],  # Only one numerical feature
            "categorical_features": ["sex"],  # Only one categorical feature
            "target": "target",
        }
        
        X_processed, y_processed = preprocessor.preprocess(sample_data, config)
        
        # Check output
        assert X_processed.shape[0] == sample_data.shape[0]
        assert len(y_processed) == sample_data.shape[0]
        assert X_processed.shape[1] >= 2  # age + encoded sex columns
    
    def test_preprocess_with_no_features(self, preprocessor, sample_data):
        """Test preprocessing with no features specified"""
        config = {
            "numerical_features": [],  # No numerical features
            "categorical_features": [],  # No categorical features
            "target": "target",
        }
        
        X_processed, y_processed = preprocessor.preprocess(sample_data, config)
        
        # Should return all columns except target
        assert X_processed.shape[0] == sample_data.shape[0]
        assert len(y_processed) == sample_data.shape[0]
        assert X_processed.shape[1] == len(sample_data.columns) - 1
