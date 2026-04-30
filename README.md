# Churn Prediction MLOps

A production-ready end-to-end churn prediction system demonstrating full-stack ML engineering and MLOps skills.

## Overview

This project predicts customer churn based on historical telecom data. It features a complete ML pipeline, a FastAPI backend, a Streamlit dashboard, Docker containerization, and a GitHub Actions CI/CD workflow.

## Architecture

- **ML Pipeline:** Python, scikit-learn, XGBoost, MLflow
- **Backend:** FastAPI for model serving
- **Frontend:** Streamlit for interactive visualization
- **Deployment:** Docker + Docker Compose
- **CI/CD:** GitHub Actions

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-mlops.git
cd churn-prediction-mlops
```

### 2. Build and run with Docker Compose
```bash
docker-compose up --build
```

This will start:
- FastAPI on `http://localhost:8000`
- Streamlit on `http://localhost:8501`
- MLflow UI on `http://localhost:5000`

### 3. Train the model locally
```bash
pip install -r requirements.txt
python src/train.py
```

## Project Structure

```
churn-prediction-mlops/
├── doc/           # Documentation
├── data/          # Data files
├── src/           # ML pipeline scripts
├── api/           # FastAPI application
├── dashboard/     # Streamlit dashboard
├── tests/         # Unit and integration tests
├── notebooks/     # Jupyter notebooks for EDA
├── .github/workflows/  # CI/CD pipelines
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## License

MIT
