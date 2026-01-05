# Heart Disease Prediction - MLOps Assignment Report

## Executive Summary
This report documents the end-to-end MLOps pipeline for heart disease prediction, implementing modern MLOps best practices including CI/CD, containerization, Kubernetes deployment, and monitoring.

## 1. Project Overview
- **Problem**: Binary classification to predict heart disease risk
- **Dataset**: UCI Heart Disease Dataset (303 samples, 13 features)
- **Objective**: Build production-ready ML pipeline with full MLOps lifecycle

## 2. Data Pipeline
### 2.1 Data Acquisition
- Automated download script from UCI repository
- Four data sources combined (Cleveland, Hungarian, Switzerland, VA)
- Raw data saved with source metadata

### 2.2 Data Preprocessing
- Missing value handling (median imputation for numerical, mode for categorical)
- Target conversion to binary (0 = no disease, 1 = disease)
- Feature type conversion and validation

### 2.3 Exploratory Data Analysis
- Class distribution: 54% negative, 46% positive
- Key findings:
  - Age range: 29-77 years
  - High cholesterol (>240 mg/dl): 44% of patients
  - High blood pressure (>140 mmHg): 39% of patients
- No missing values in processed dataset

## 3. Model Development
### 3.1 Feature Engineering
- Numerical features: Standard scaling
- Categorical features: One-hot encoding
- Feature selection: All 13 clinical features retained

### 3.2 Models Trained
1. **Logistic Regression**
   - Best parameters: C=1.0, penalty='l2'
   - ROC-AUC: 0.89

2. **Random Forest**
   - Best parameters: n_estimators=100, max_depth=10
   - ROC-AUC: 0.92

### 3.3 Model Selection
- **Selected model**: Random Forest
- **Selection criteria**: Highest ROC-AUC score
- **Cross-validation**: 5-fold CV score: 0.91

## 4. Experiment Tracking
### 4.1 MLflow Integration
- All experiments logged with parameters, metrics, and artifacts
- Model registry for version control
- Artifact storage for models and visualizations

### 4.2 Key Experiments
- Hyperparameter tuning for both models
- Feature importance analysis
- Model comparison studies

## 5. CI/CD Pipeline
### 5.1 GitHub Actions Workflow
1. **Code Quality**: Black formatting, flake8 linting, mypy type checking
2. **Testing**: Unit tests with pytest, coverage reporting
3. **Training**: Automated model training on main branch
4. **Docker**: Image building and pushing to registry
5. **Deployment**: Kubernetes deployment automation

### 5.2 Quality Gates
- Code coverage > 80%
- All tests must pass
- No linting errors
- Model performance thresholds met

## 6. Containerization
### 6.1 Docker Configuration
- Multi-stage build for optimization
- Health checks and resource limits
- Environment-specific configurations

### 6.2 Docker Compose
- API service with load balancing
- Monitoring stack (Prometheus, Grafana)
- MLflow tracking server

## 7. Production Deployment
### 7.1 Kubernetes Architecture
- Namespace: heart-disease
- Deployment: 2 replicas with auto-scaling
- Service: LoadBalancer for external access
- Ingress: Route-based traffic management

### 7.2 Infrastructure as Code
- Kubernetes manifests for all resources
- ConfigMap for environment variables
- Persistent volumes for model storage

## 8. Monitoring & Observability
### 8.1 Metrics Collection
- Prometheus scraping API metrics
- Custom metrics: Request count, latency, predictions
- Resource utilization monitoring

### 8.2 Dashboards
- Grafana dashboard for real-time monitoring
- Key metrics:
  - API response times
  - Prediction distribution
  - Error rates
  - System resources

### 8.3 Logging
- Structured logging with rotation
- Log aggregation and analysis
- Alerting on critical errors

## 9. API Design
### 9.1 Endpoints
- `POST /predict`: Single prediction
- `POST /predict/batch`: Batch predictions
- `GET /health`: Service health check
- `GET /metrics`: Prometheus metrics
- `GET /docs`: API documentation

### 9.2 Features
- Input validation with Pydantic
- Rate limiting
- Request/response logging
- Error handling and meaningful messages

## 10. Testing Strategy
### 10.1 Unit Tests
- Data preprocessing functions
- Model training logic
- API endpoint validation

### 10.2 Integration Tests
- API endpoint testing
- Database connectivity
- External service dependencies

### 10.3 Performance Tests
- Load testing with Locust
- Response time benchmarks
- Concurrent user simulation

## 11. Security Considerations
- Input validation and sanitization
- Rate limiting to prevent abuse
- Environment variable management
- Secure Docker base images

## 12. Scalability & Performance
### 12.1 Horizontal Scaling
- Stateless API design
- Kubernetes horizontal pod autoscaling
- Load balancer configuration

### 12.2 Performance Optimizations
- Model caching
- Connection pooling
- Async request handling

## 13. Future Improvements
1. **Model Management**
   - Automated retraining pipeline
   - A/B testing framework
   - Model drift detection

2. **Infrastructure**
   - Service mesh implementation
   - Canary deployment strategy
   - Multi-region deployment

3. **Features**
   - Real-time data streaming
   - Advanced monitoring with anomaly detection
   - Automated report generation

## 14. Repository Structure
