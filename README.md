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
git clone https://github.com/GucciPants/churn-prediction-mlops.git
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
├── doc/                  # Documentation
│   ├── architecture.md
│   └── decisions.md
├── data/                 # Data files
│   └── raw/
├── src/                  # ML pipeline scripts
│   ├── config.py
│   ├── features.py
│   └── train.py
├── api/                  # FastAPI application
│   └── main.py
├── dashboard/            # Streamlit dashboard
│   └── app.py
├── tests/                # Unit and integration tests
│   ├── test_features.py
│   └── test_api.py
├── notebooks/            # Jupyter notebooks for EDA
│   └── 01_eda.ipynb
├── models/               # Trained model artifacts
├── .github/workflows/    # CI/CD pipelines
│   └── ci.yml
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch predictions |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, XGBoost |
| Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Container | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

## License

MIT
