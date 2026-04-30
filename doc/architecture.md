# Project Overview

## Goal
Build a production-ready churn prediction system with a complete ML pipeline, FastAPI backend, and Streamlit UI. The goal is to showcase end-to-end ML engineering and MLOps skills.

## Business Value
Predict which telecom customers are likely to leave. This enables targeted retention campaigns, reducing churn and increasing revenue.

## Data Source
IBM/Kaggle Telco Customer Churn dataset (free, ~7,000 rows).

## Tech Stack

| Component          | Technology                | Purpose                                          |
|--------------------|---------------------------|--------------------------------------------------|
| ML Pipeline        | scikit-learn, XGBoost     | Model training and evaluation                    |
| Experiment Tracking| MLflow                    | Log parameters, metrics, and model artifacts     |
| API                | FastAPI (Python)          | Serve predictions via REST endpoint              |
| Frontend           | Streamlit (Python)        | Interactive dashboard for data upload & results  |
| Containerization   | Docker, Docker Compose    | Portable, reproducible deployment                |
| CI/CD              | GitHub Actions            | Automated testing, building, and deployment      |

## Data Flow

```
Kaggle CSV
    |
    v
Data Cleaning & Feature Engineering (src/features.py)
    |
    v
Train/Test Split & Model Training (src/train.py)
    |
    v
MLflow Logging (metrics, params, model artifact)
    |
    v
Export Trained Model (pickle / joblib)
    |
    v
FastAPI Service (api/main.py)
    |
    v
Streamlit Dashboard (dashboard/app.py)
```

## Target Metrics
- Accuracy: ~80-85%
- ROC AUC: >0.85
- F1 Score: >0.75

## Deliverables
- Fully functional ML pipeline
- REST API for predictions
- Interactive web dashboard
- Docker setup for one-command deployment
- CI/CD pipeline with GitHub Actions
- Comprehensive documentation
