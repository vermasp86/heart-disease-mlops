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
                "age": [63, 37, 41, np.nan, 56],
                "sex": [1, 1, 0, 1, 0],
                "cp": [3, 2, 1, 0, 3],
                "trestbps": [145, 130, 130, np.nan, 120],
                "chol": [233, 250, 204, 236, 256],
                "fbs": [1, 0, 0, 0, 1],
                "restecg": [0, 1, 0, 1, 1],
                "thalach": [150, 187, 172, np.nan, 142],
                "exang": [0, 0, 0, 1, 1],
                "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.0],
                "slope": [0, 0, 2, 1, 2],
                "ca": [0, 0, 0, 0, 1],
                "thal": [1, 2, 2, 3, 2],
                "target": [1, 0, 0, 1, 0],
            }
        )
        return data

    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        return DataPreprocessor()

    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        processed_data = preprocessor.handle_missing_values(sample_data)

        # Check no NaN values remain
        assert processed_data.isnull().sum().sum() == 0

        # Check numerical columns filled with median
        assert processed_data["age"].isnull().sum() == 0
        assert processed_data["trestbps"].isnull().sum() == 0

    def test_convert_target_to_binary(self, preprocessor, sample_data):
        """Test target conversion"""
        processed_data = preprocessor.convert_target_to_binary(sample_data)

        # Check target values are binary
        assert set(processed_data["target"].unique()).issubset({0, 1})

    def test_feature_scaling(self, preprocessor, sample_data):
        """Test feature scaling"""
        numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        processed_data = preprocessor.scale_features(sample_data, numerical_features)

        # Check scaling (mean ~0, std ~1 for scaled columns)
        for feature in numerical_features:
            if f"{feature}_scaled" in processed_data.columns:
                scaled_mean = processed_data[f"{feature}_scaled"].mean()
                scaled_std = processed_data[f"{feature}_scaled"].std()
                assert abs(scaled_mean) < 0.1  # Should be close to 0
                assert abs(scaled_std - 1) < 0.1  # Should be close to 1

    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Test categorical feature encoding"""
        categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        processed_data = preprocessor.encode_categorical(sample_data, categorical_features)

        # Check one-hot encoding
        for feature in categorical_features:
            encoded_cols = [col for col in processed_data.columns if col.startswith(f"{feature}_")]
            if encoded_cols:
                # Sum of one-hot columns should be 1 for each row
                assert all(processed_data[encoded_cols].sum(axis=1) == 1)

    def test_full_preprocessing_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline"""
        config = {
            "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            "target": "target",
        }

        X_processed, y_processed = preprocessor.preprocess(sample_data, config)

        # Check output shapes
        assert X_processed.shape[0] == sample_data.shape[0]
        assert len(y_processed) == sample_data.shape[0]

        # Check no NaN values
        assert not X_processed.isnull().any().any()
        assert not y_processed.isnull().any()
