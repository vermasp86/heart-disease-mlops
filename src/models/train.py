"""
Model training script with MLflow tracking
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class HeartDiseaseModelTrainer:
    """Model trainer for heart disease prediction"""

    def __init__(self, config: Any):
        """
        Initialize model trainer

        Args:
            config: Configuration object
        """
        self.config = config
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.results = {}

        # Create directories
        self.create_directories()

        # Setup MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)

    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'models',
            'logs',
            'reports',
            'reports/figures',
            'mlruns'  # Add MLflow directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare data

        Returns:
            pd.DataFrame: Processed data
        """
        logger.info("Loading data...")

        try:
            self.data = pd.read_csv(self.config.data.processed_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")

            # Separate features and target
            X = self.data[self.config.data.numerical_features + self.config.data.categorical_features]
            y = self.data[self.config.data.target]

            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=self.config.model.test_size,
                random_state=self.config.model.random_state,
                stratify=y,
            )

            logger.info(f"Train set: {self.X_train.shape}")
            logger.info(f"Test set: {self.X_test.shape}")
            logger.info(f"Class distribution in train: {self.y_train.value_counts().to_dict()}")
            logger.info(f"Class distribution in test: {self.y_test.value_counts().to_dict()}")

            return self.data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create data preprocessor

        Returns:
            ColumnTransformer: Fitted preprocessor
        """
        logger.info("Creating preprocessor...")

        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config.data.numerical_features),
                ("cat", categorical_transformer, self.config.data.categorical_features),
            ]
        )

        # Fit preprocessor on training data
        self.preprocessor.fit(self.X_train)

        # Save preprocessor
        preprocessor_path = "models/preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.preprocessor, f)

        logger.info(f"Preprocessor saved to {preprocessor_path}")

        return self.preprocessor

    def train_models(self) -> Dict[str, Any]:
        """
        Train multiple models and select the best one

        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Training models...")

        # Transform data
        X_train_transformed = self.preprocessor.transform(self.X_train)
        X_test_transformed = self.preprocessor.transform(self.X_test)

        models = {
            "logistic_regression": {
                "model": LogisticRegression(random_state=self.config.model.random_state),
                "params": self.config.model.logistic_regression_params,
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=self.config.model.random_state),
                "params": self.config.model.random_forest_params,
            },
        }

        self.results = {}

        with mlflow.start_run(run_name="model_comparison"):
            # Log data info
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("test_size", len(self.X_test))
            mlflow.log_param("n_features", X_train_transformed.shape[1])

            for model_name, model_info in models.items():
                logger.info(f"Training {model_name}...")

                with mlflow.start_run(run_name=model_name, nested=True):
                    # Hyperparameter tuning
                    grid_search = GridSearchCV(
                        estimator=model_info["model"],
                        param_grid=model_info["params"],
                        cv=self.config.model.cv_folds,
                        scoring=self.config.model.scoring_metric,
                        n_jobs=-1,
                        verbose=1,
                    )

                    grid_search.fit(X_train_transformed, self.y_train)

                    # Get best model
                    best_model = grid_search.best_estimator_

                    # Make predictions
                    y_pred = best_model.predict(X_test_transformed)
                    y_pred_proba = best_model.predict_proba(X_test_transformed)[:, 1]

                    # Calculate metrics
                    metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)

                    # Store results
                    self.results[model_name] = {
                        "model": best_model,
                        "params": grid_search.best_params_,
                        "metrics": metrics,
                        "cv_score": grid_search.best_score_,
                    }

                    # Log to MLflow
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric("cv_score", grid_search.best_score_)

                    # Log model
                    signature = infer_signature(
                        X_train_transformed, best_model.predict(X_train_transformed)
                    )
                    mlflow.sklearn.log_model(
                        best_model,
                        artifact_path=f"models/{model_name}",
                        signature=signature,
                        registered_model_name=f"{self.config.mlflow.registered_model_name}_{model_name}",
                    )

                    # Log artifacts
                    self.plot_roc_curve(self.y_test, y_pred_proba, model_name)
                    self.plot_confusion_matrix(self.y_test, y_pred, model_name)

                    logger.info(f"{model_name} - Best params: {grid_search.best_params_}")
                    logger.info(f"{model_name} - CV Score: {grid_search.best_score_:.4f}")
                    logger.info(f"{model_name} - Test Accuracy: {metrics['accuracy']:.4f}")

            # Select best model
            self.select_best_model()

            # Log comparison
            mlflow.log_metric("best_model_score", self.results[self.best_model_name]["metrics"]["roc_auc"])
            mlflow.log_param("best_model", self.best_model_name)

        return self.results

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str):
        """
        Plot and save ROC curve

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = f"reports/figures/roc_curve_{model_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Log to MLflow
        mlflow.log_artifact(plot_path)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """
        Plot and save confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")

        # Save plot
        plot_path = f"reports/figures/confusion_matrix_{model_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Log to MLflow
        mlflow.log_artifact(plot_path)

    def select_best_model(self):
        """Select the best model based on ROC-AUC score"""
        best_score = -1
        best_model_name = None

        for model_name, result in self.results.items():
            if result["metrics"]["roc_auc"] > best_score:
                best_score = result["metrics"]["roc_auc"]
                best_model_name = model_name

        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]["model"]

        logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_score:.4f}")

    def save_best_model(self):
        """Save the best model and results"""
        logger.info("Saving best model...")

        # Save best model
        model_path = "models/best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.best_model,
                    "preprocessor": self.preprocessor,
                    "config": self.config,
                    "results": self.results,
                },
                f,
            )

        # Save results as JSON
        results_dict = {}
        for model_name, result in self.results.items():
            results_dict[model_name] = {
                "params": result["params"],
                "metrics": result["metrics"],
                "cv_score": result["cv_score"],
            }

        with open("reports/model_results.json", "w") as f:
            json.dump(results_dict, f, indent=4)

        # Create model card
        self.create_model_card()

        logger.info(f"Best model saved to {model_path}")
        logger.info(f"Results saved to reports/model_results.json")

    def create_model_card(self):
        """Create model card for documentation"""
        model_card = f"""
# Model Card: Heart Disease Prediction

## Model Details
- **Model Name**: {self.best_model_name}
- **Version**: 1.0.0
- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
- **Framework**: Scikit-learn

## Intended Use
- **Primary Use**: Predict the risk of heart disease based on patient health metrics
- **Target Users**: Healthcare professionals, researchers
- **Limitations**: Should not be used as sole diagnostic tool

## Training Data
- **Source**: UCI Heart Disease Dataset
- **Size**: {len(self.data)} samples
- **Features**: {len(self.config.data.numerical_features + self.config.data.categorical_features)}
- **Class Distribution**: {dict(self.data[self.config.data.target].value_counts())}

## Performance
"""

        for model_name, result in self.results.items():
            metrics = result["metrics"]
            model_card += f"\n### {model_name.title()}\n"
            model_card += f"- Accuracy: {metrics['accuracy']:.4f}\n"
            model_card += f"- Precision: {metrics['precision']:.4f}\n"
            model_card += f"- Recall: {metrics['recall']:.4f}\n"
            model_card += f"- F1-Score: {metrics['f1']:.4f}\n"
            model_card += f"- ROC-AUC: {metrics['roc_auc']:.4f}\n"
            model_card += f"- Cross-validation Score: {result['cv_score']:.4f}\n"

        model_card += f"""
## Best Model Details
- **Selected Model**: {self.best_model_name}
- **Hyperparameters**: {self.results[self.best_model_name]['params']}

## Ethical Considerations
- The model may have biases based on the training data demographics
- Should be validated on diverse populations before clinical use
- Predictions should be interpreted by medical professionals

## Maintenance
- Regular retraining recommended with new data
- Monitor for concept drift
- Update when performance degrades below acceptable thresholds
"""

        with open("reports/model_card.md", "w") as f:
            f.write(model_card)

        logger.info("Model card created at reports/model_card.md")

    def print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING SUMMARY")
        print("=" * 60)

        print(f"\nDataset:")
        print(f"  - Total samples: {len(self.data)}")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Test samples: {len(self.X_test)}")
        print(f"  - Features: {len(self.config.data.numerical_features + self.config.data.categorical_features)}")

        print(f"\nModels Trained: {list(self.results.keys())}")

        print(f"\nBest Model: {self.best_model_name}")
        print(f"  - ROC-AUC Score: {self.results[self.best_model_name]['metrics']['roc_auc']:.4f}")
        print(f"  - Accuracy: {self.results[self.best_model_name]['metrics']['accuracy']:.4f}")
        print(f"  - Hyperparameters: {self.results[self.best_model_name]['params']}")

        print(f"\nDetailed Metrics:")
        for model_name, result in self.results.items():
            print(f"\n  {model_name.upper()}:")
            for metric_name, metric_value in result["metrics"].items():
                print(f"    - {metric_name}: {metric_value:.4f}")

        print(f"\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)


def main():
    """Main training function"""
    try:
        logger.info("Starting model training pipeline...")

        # Initialize trainer
        trainer = HeartDiseaseModelTrainer(CONFIG)

        # Load data
        trainer.load_data()

        # Create preprocessor
        trainer.create_preprocessor()

        # Train models
        results = trainer.train_models()

        # Save best model
        trainer.save_best_model()

        # Print summary
        trainer.print_summary()

        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
