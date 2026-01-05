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
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Create nested config objects
        data_config = DataConfig(
            raw_path=config_dict["data"]["raw_path"],
            processed_path=config_dict["data"]["processed_path"],
            test_size=config_dict["data"]["test_size"],
            random_state=config_dict["data"]["random_state"],
            categorical_features=config_dict["features"]["categorical"],
            numerical_features=config_dict["features"]["numerical"],
            target=config_dict["features"]["target"],
        )

        model_config = ModelConfig(
            random_state=config_dict["model"]["random_state"],
            test_size=config_dict["model"]["test_size"],
            cv_folds=config_dict["model"]["cv_folds"],
            scoring_metric=config_dict["model"]["scoring_metric"],
            random_forest_params=config_dict["model"]["random_forest"],
            logistic_regression_params=config_dict["model"]["logistic_regression"],
        )

        mlflow_config = MLflowConfig(
            tracking_uri=config_dict["mlflow"]["tracking_uri"],
            experiment_name=config_dict["mlflow"]["experiment_name"],
            registered_model_name=config_dict["mlflow"]["registered_model_name"],
        )

        api_config = APIConfig(
            host=config_dict["api"]["host"],
            port=config_dict["api"]["port"],
            reload=config_dict["api"]["reload"],
            workers=config_dict["api"]["workers"],
            log_level=config_dict["api"]["log_level"],
        )

        deployment_config = DeploymentConfig(
            docker=config_dict["deployment"]["docker"],
            kubernetes=config_dict["deployment"]["kubernetes"],
        )

        return cls(
            data=data_config,
            model=model_config,
            mlflow=mlflow_config,
            api=api_config,
            deployment=deployment_config,
        )


# Global configuration instance
CONFIG = Config.from_yaml("config/config.yaml")
