"""
Unit tests for models
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.train import HeartDiseaseModelTrainer


class TestModelTraining:
    """Test cases for model training"""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic data for testing"""
        np.random.seed(42)

        n_samples = 100
        X = pd.DataFrame(
            {
                "age": np.random.normal(50, 10, n_samples),
                "sex": np.random.randint(0, 2, n_samples),
                "cp": np.random.randint(0, 4, n_samples),
                "trestbps": np.random.normal(130, 20, n_samples),
                "chol": np.random.normal(240, 50, n_samples),
                "fbs": np.random.randint(0, 2, n_samples),
                "restecg": np.random.randint(0, 3, n_samples),
                "thalach": np.random.normal(150, 20, n_samples),
                "exang": np.random.randint(0, 2, n_samples),
                "oldpeak": np.random.exponential(1, n_samples),
                "slope": np.random.randint(0, 3, n_samples),
                "ca": np.random.randint(0, 4, n_samples),
                "thal": np.random.randint(0, 4, n_samples),
            }
        )

        # Create target with some relationship to features
        y = ((X["age"] > 55) & (X["chol"] > 240)).astype(int)
        y = y | (X["thalach"] < 120).astype(int)

        return X, y

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        class MockConfig:
            data = type(
                "obj",
                (object,),
                {
                    "processed_path": "test_data.csv",
                    "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
                    "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
                    "target": "target",
                    "test_size": 0.2,
                    "random_state": 42,
                },
            )()

            model = type(
                "obj",
                (object,),
                {
                    "random_state": 42,
                    "test_size": 0.2,
                    "cv_folds": 3,
                    "scoring_metric": "accuracy",
                    "logistic_regression_params": {"C": [0.1, 1.0], "penalty": ["l2"]},
                    "random_forest_params": {"n_estimators": [10, 20], "max_depth": [5, 10]},
                },
            )()

            mlflow = type(
                "obj",
                (object,),
                {
                    "tracking_uri": "mlruns",
                    "experiment_name": "test_experiment",
                    "registered_model_name": "TestModel",
                },
            )()

        return MockConfig()

    def test_data_loading(self, mock_config, sample_data, tmp_path):
        """Test data loading functionality"""
        # Save sample data
        X, y = sample_data
        data = pd.concat([X, y.rename("target")], axis=1)
        data_path = tmp_path / "test_data.csv"
        data.to_csv(data_path, index=False)

        mock_config.data.processed_path = str(data_path)

        trainer = HeartDiseaseModelTrainer(mock_config)
        loaded_data = trainer.load_data()

        assert loaded_data is not None
        assert len(loaded_data) == len(data)

    def test_model_initialization(self, mock_config):
        """Test model trainer initialization"""
        trainer = HeartDiseaseModelTrainer(mock_config)

        assert trainer.config == mock_config
        assert trainer.data is None
        assert trainer.best_model is None

    def test_metrics_calculation(self, mock_config):
        """Test metrics calculation"""
        trainer = HeartDiseaseModelTrainer(mock_config)

        # Create test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.6, 0.3])

        metrics = trainer.calculate_metrics(y_true, y_pred, y_pred_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

        # Check accuracy calculation
        expected_accuracy = 0.8  # 4 out of 5 correct
        assert abs(metrics["accuracy"] - expected_accuracy) < 0.01

    def test_preprocessor_creation(self, mock_config, sample_data):
        """Test preprocessor creation"""
        X, y = sample_data
        trainer = HeartDiseaseModelTrainer(mock_config)

        # Mock data loading
        trainer.X_train = X
        trainer.y_train = y

        preprocessor = trainer.create_preprocessor()

        assert preprocessor is not None
        assert hasattr(preprocessor, "transform")

    @pytest.mark.skip(reason="MLflow integration test")
    def test_mlflow_integration(self, mock_config):
        """Test MLflow integration"""
        # This would test MLflow logging in a real scenario
        pass

    def test_best_model_selection(self, mock_config):
        """Test best model selection logic"""
        trainer = HeartDiseaseModelTrainer(mock_config)

        # Mock results
        trainer.results = {
            "logistic_regression": {"metrics": {"roc_auc": 0.85, "accuracy": 0.82}},
            "random_forest": {"metrics": {"roc_auc": 0.90, "accuracy": 0.88}},
        }

        trainer.select_best_model()

        assert trainer.best_model_name == "random_forest"
        assert trainer.best_model is None  # Model not actually trained in this test
