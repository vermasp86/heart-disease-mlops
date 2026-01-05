from setuptools import setup, find_packages

setup(
    name="heart-disease-mlops",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "mlflow>=2.8.1",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
    ],
    author="MLOps Assignment",
    description="Heart Disease Prediction MLOps Pipeline",
    python_requires=">=3.9",
)
