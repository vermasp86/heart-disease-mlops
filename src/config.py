"""
Configuration management for the MLOps pipeline
"""

import yaml
import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration"""

    raw_path: str
    processed_path: str
    test_size: float
    random_state: int
    categorical_features: List[str]
    numerical_features: List[str]
    target: str


@dataclass
class ModelConfig:
    """Model configuration"""

    random_state: int
    test_size: float
    cv_folds: int
    scoring_metric: str
    random_forest_params: Dict[str, Any]
    logistic_regression_params: Dict[str, Any]


@dataclass
class MLflowConfig:
    """MLflow configuration"""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str


@dataclass
class APIConfig:
    """API configuration"""

    host: str
    port: int
    reload: bool
    workers: int
    log_level: str


@dataclass
class DeploymentConfig:
    """Deployment configuration"""

    docker: Dict[str, Any]
    kubernetes: Dict[str, Any]


@dataclass
class Config:
    """Main configuration class"""

    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
    api: APIConfig
    deployment: DeploymentConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            # Fallback to default config
            return cls.get_default_config()

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Create nested config objects
        data_config = DataConfig(
            raw_path=config_dict.get("data", {}).get("raw_path", "data/raw/heart_disease_raw.csv"),
            processed_path=config_dict.get("data", {}).get("processed_path", "data/processed/heart_disease_processed.csv"),
            test_size=config_dict.get("data", {}).get("test_size", 0.2),
            random_state=config_dict.get("data", {}).get("random_state", 42),
            categorical_features=config_dict.get("features", {}).get("categorical", ["sex", "cp", "fbs", "restecg", "exang", "slope"]),
            numerical_features=config_dict.get("features", {}).get("numerical", ["age", "trestbps", "chol", "thalach", "oldpeak"]),
            target=config_dict.get("features", {}).get("target", "target"),
        )

        model_config = ModelConfig(
            random_state=config_dict.get("model", {}).get("random_state", 42),
            test_size=config_dict.get("model", {}).get("test_size", 0.2),
            cv_folds=config_dict.get("model", {}).get("cv_folds", 5),
            scoring_metric=config_dict.get("model", {}).get("scoring_metric", "roc_auc"),
            random_forest_params=config_dict.get("model", {}).get("random_forest", {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }),
            logistic_regression_params=config_dict.get("model", {}).get("logistic_regression", {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"]
            }),
        )

        mlflow_config = MLflowConfig(
            tracking_uri=config_dict.get("mlflow", {}).get("tracking_uri", "mlruns"),
            experiment_name=config_dict.get("mlflow", {}).get("experiment_name", "heart_disease_prediction"),
            registered_model_name=config_dict.get("mlflow", {}).get("registered_model_name", "HeartDiseaseClassifier"),
        )

        api_config = APIConfig(
            host=config_dict.get("api", {}).get("host", "0.0.0.0"),
            port=config_dict.get("api", {}).get("port", 8000),
            reload=config_dict.get("api", {}).get("reload", False),
            workers=config_dict.get("api", {}).get("workers", 2),
            log_level=config_dict.get("api", {}).get("log_level", "info"),
        )

        deployment_config = DeploymentConfig(
            docker=config_dict.get("deployment", {}).get("docker", {
                "image_name": "heart-disease-api",
                "tag": "latest",
                "port": 8000
            }),
            kubernetes=config_dict.get("deployment", {}).get("kubernetes", {
                "namespace": "heart-disease",
                "replicas": 2,
                "service_type": "LoadBalancer",
                "port": 80,
                "target_port": 8000
            }),
        )

        return cls(
            data=data_config,
            model=model_config,
            mlflow=mlflow_config,
            api=api_config,
            deployment=deployment_config,
        )

    @classmethod
    def get_default_config(cls) -> "Config":
        """Get default configuration"""
        return cls(
            data=DataConfig(
                raw_path="data/raw/heart_disease_raw.csv",
                processed_path="data/processed/heart_disease_processed.csv",
                test_size=0.2,
                random_state=42,
                categorical_features=["sex", "cp", "fbs", "restecg", "exang", "slope"],
                numerical_features=["age", "trestbps", "chol", "thalach", "oldpeak"],
                target="target"
            ),
            model=ModelConfig(
                random_state=42,
                test_size=0.2,
                cv_folds=5,
                scoring_metric="roc_auc",
                random_forest_params={
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },
                logistic_regression_params={
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                }
            ),
            mlflow=MLflowConfig(
                tracking_uri="mlruns",
                experiment_name="heart_disease_prediction",
                registered_model_name="HeartDiseaseClassifier"
            ),
            api=APIConfig(
                host="0.0.0.0",
                port=8000,
                reload=False,
                workers=2,
                log_level="info"
            ),
            deployment=DeploymentConfig(
                docker={
                    "image_name": "heart-disease-api",
                    "tag": "latest",
                    "port": 8000
                },
                kubernetes={
                    "namespace": "heart-disease",
                    "replicas": 2,
                    "service_type": "LoadBalancer",
                    "port": 80,
                    "target_port": 8000
                }
            )
        )


# Global configuration instance - try to load from YAML, fallback to default
try:
    CONFIG = Config.from_yaml("config/config.yaml")
except Exception as e:
    print(f"Warning: Could not load config from YAML: {e}. Using default configuration.")
    CONFIG = Config.get_default_config()
