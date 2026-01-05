"""
FastAPI application for heart disease prediction API
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import logging
import time
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime

# Monitoring imports
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import prometheus_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk based on patient health metrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made", ["prediction"])

# Load model and preprocessor
MODEL = None
PREPROCESSOR = None


class PatientData(BaseModel):
    """Patient data model for prediction"""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (1-3)")

    @validator("*")
    def check_nan(cls, v):
        """Check for NaN values"""
        if pd.isna(v):
            raise ValueError("NaN values are not allowed")
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model"""

    prediction: int = Field(..., description="0 = No Disease, 1 = Disease")
    probability: float = Field(..., ge=0, le=1, description="Probability of having heart disease")
    risk_level: str = Field(..., description="Risk level category")
    features: Dict[str, Any] = Field(..., description="Input features")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version")


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    model_loaded: bool
    timestamp: str
    uptime: float


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""

    patients: List[PatientData]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""

    predictions: List[Dict[str, Any]]
    total_patients: int
    positive_cases: int
    negative_cases: int
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global MODEL, PREPROCESSOR

    try:
        logger.info("Loading model and preprocessor...")

        # Load model package
        model_data = joblib.load("models/best_model.pkl")
        MODEL = model_data["model"]
        PREPROCESSOR = model_data["preprocessor"]

        logger.info(f"Model loaded successfully: {type(MODEL).__name__}")
        logger.info(f"Preprocessor loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        MODEL = None
        PREPROCESSOR = None


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware for monitoring requests"""
    start_time = time.time()

    try:
        response = await call_next(request)

        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start_time)

        # Log request
        logger.info(f"{request.method} {request.url.path} - {response.status_code}")

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    start_time = getattr(app, "start_time", time.time())

    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        timestamp=datetime.now().isoformat(),
        uptime=time.time() - start_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict heart disease risk for a single patient

    Args:
        patient (PatientData): Patient health metrics

    Returns:
        PredictionResponse: Prediction results
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert patient data to DataFrame
        patient_dict = patient.dict()
        features_df = pd.DataFrame([patient_dict])

        # Preprocess features
        features_processed = PREPROCESSOR.transform(features_df)

        # Make prediction
        prediction = MODEL.predict(features_processed)[0]
        probability = MODEL.predict_proba(features_processed)[0][1]

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Record prediction metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        # Log prediction
        logger.info(f"Prediction made - Risk: {risk_level}, Probability: {probability:.3f}")

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            features=patient_dict,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0",
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Predict heart disease risk for multiple patients

    Args:
        batch_request (BatchPredictionRequest): List of patients

    Returns:
        BatchPredictionResponse: Batch prediction results
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        patients_data = [patient.dict() for patient in batch_request.patients]
        features_df = pd.DataFrame(patients_data)

        # Preprocess features
        features_processed = PREPROCESSOR.transform(features_df)

        # Make predictions
        predictions = MODEL.predict(features_processed)
        probabilities = MODEL.predict_proba(features_processed)[:, 1]

        # Prepare results
        results = []
        positive_count = 0
        negative_count = 0

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Determine risk level
            if prob < 0.3:
                risk_level = "Low"
            elif prob < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            if pred == 1:
                positive_count += 1
            else:
                negative_count += 1

            results.append(
                {
                    "patient_id": i + 1,
                    "prediction": int(pred),
                    "probability": float(prob),
                    "risk_level": risk_level,
                    "features": patients_data[i],
                }
            )

        logger.info(f"Batch prediction completed - Total: {len(results)}, Positive: {positive_count}")

        return BatchPredictionResponse(
            predictions=results,
            total_patients=len(results),
            positive_cases=positive_count,
            negative_cases=negative_count,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/features")
async def get_features_info():
    """
    Get information about the features used by the model
    """
    feature_info = {
        "numerical_features": [
            {"name": "age", "description": "Age in years", "range": [0, 120]},
            {"name": "trestbps", "description": "Resting blood pressure (mm Hg)", "range": [0, 300]},
            {"name": "chol", "description": "Serum cholesterol (mg/dl)", "range": [0, 600]},
            {"name": "thalach", "description": "Maximum heart rate achieved", "range": [0, 250]},
            {"name": "oldpeak", "description": "ST depression induced by exercise", "range": [0, 10]},
        ],
        "categorical_features": [
            {"name": "sex", "description": "Sex (1=male, 0=female)", "values": [0, 1]},
            {"name": "cp", "description": "Chest pain type", "values": [0, 1, 2, 3]},
            {"name": "fbs", "description": "Fasting blood sugar > 120 mg/dl", "values": [0, 1]},
            {"name": "restecg", "description": "Resting electrocardiographic results", "values": [0, 1, 2]},
            {"name": "exang", "description": "Exercise induced angina", "values": [0, 1]},
            {"name": "slope", "description": "Slope of peak exercise ST segment", "values": [0, 1, 2]},
            {"name": "ca", "description": "Number of major vessels", "values": [0, 1, 2, 3]},
            {"name": "thal", "description": "Thalassemia", "values": [0, 1, 2, 3]},
        ],
    }

    return feature_info


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_info = {
        "model_type": type(MODEL).__name__,
        "model_version": "1.0.0",
        "loaded": True,
        "features_count": PREPROCESSOR.transformers_[0][2].shape[0] + PREPROCESSOR.transformers_[1][2].shape[0]
        if PREPROCESSOR
        else 0,
        "training_date": "2024-01-01",  # This should be loaded from model metadata
        "performance_metrics": {"accuracy": 0.85, "roc_auc": 0.92},  # Load from saved metrics
    }

    return model_info


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled error: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Set start time for uptime calculation
    app.start_time = time.time()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
