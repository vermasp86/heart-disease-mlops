"""
Pytest configuration file
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create necessary directories before tests run
@pytest.fixture(scope="session", autouse=True)
def create_test_directories():
    """Create directories needed for tests"""
    directories = ['logs', 'models', 'reports/figures']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    yield
    
    # Cleanup (optional)
    pass

# Create a mock config for testing
@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    class MockConfig:
        data = type(
            "obj",
            (object,),
            {
                "processed_path": "test_data.csv",
                "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
                "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope"],
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
