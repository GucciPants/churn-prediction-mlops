# Churn Prediction MLOps

[![CI](https://github.com/GucciPants/churn-prediction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/GucciPants/churn-prediction-mlops/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready end-to-end churn prediction system demonstrating full-stack ML engineering and MLOps skills.

![Architecture](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)

## Overview

This project predicts customer churn based on historical telecom data. It features a complete ML pipeline, a FastAPI backend, a Streamlit dashboard, Docker containerization, and GitHub Actions CI/CD.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Source                              │
│                   (Kaggle Telco Dataset)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML Pipeline                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Training   │  │   SHAP       │  │   MLflow Tracking    │  │
│  │   (XGBoost,  │  │   Explainer  │  │   (Metrics, Models)  │  │
│  │   RF)        │  │              │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   /predict   │  │   /explain   │  │   /analytics         │  │
│  │   /batch     │  │   /compare   │  │   /monitoring        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Single     │  │   Batch      │  │   Analytics          │  │
│  │   Prediction │  │   Prediction │  │   Monitoring         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML Pipeline | scikit-learn, XGBoost | Model training and evaluation |
| Explainability | SHAP | Model interpretability |
| Experiment Tracking | MLflow | Log parameters, metrics, and model artifacts |
| API | FastAPI, Uvicorn | Serve predictions via REST endpoints |
| Frontend | Streamlit | Interactive dashboard for data upload & results |
| Containerization | Docker, Docker Compose | Portable, reproducible deployment |
| CI/CD | GitHub Actions | Automated testing, building, and deployment |
| Testing | pytest | Unit and integration tests |

## Features

- **Multi-model Training**: RandomForest and XGBoost with hyperparameter configuration
- **SHAP Explanations**: Global and local feature importance for model interpretability
- **Model Comparison**: Side-by-side prediction comparison between models
- **Batch Predictions**: Process entire datasets via API or CSV upload
- **Real-time Monitoring**: Data drift detection and high-risk customer alerts
- **Interactive Dashboard**: 5 tabs for different use cases
- **Production Ready**: Docker containerization with health checks

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
- **FastAPI** on `http://localhost:8000` (with Swagger docs at `/docs`)
- **Streamlit** on `http://localhost:8501`
- **MLflow** on `http://localhost:5000`

### 3. Train the model locally
```bash
pip install -r requirements.txt
python src/train.py
```

## Project Structure

```
churn-prediction-mlops/
├── api/                    # FastAPI application
│   └── main.py            # API endpoints and routes
├── dashboard/              # Streamlit dashboard
│   └── app.py             # Interactive UI
├── doc/                    # Documentation
│   ├── architecture.md    # System architecture
│   ├── decisions.md       # Key decisions & rationale
│   └── features.md        # Feature documentation
├── notebooks/              # Jupyter notebooks for EDA
│   └── 01_eda.ipynb       # Exploratory data analysis
├── src/                    # ML pipeline modules
│   ├── config.py          # Configuration and constants
│   ├── explainer.py       # SHAP explanation utilities
│   ├── features.py        # Feature engineering pipeline
│   ├── preprocessing.py   # Shared preprocessing (DRY)
│   └── train.py           # Model training pipeline
├── tests/                  # Test suite
│   ├── test_api.py        # API endpoint tests
│   ├── test_features.py   # Feature engineering tests
│   └── test_preprocessing.py  # Preprocessing tests
├── .github/workflows/      # CI/CD pipelines
│   ├── ci.yml             # Test and build pipeline
│   └── retrain.yml        # Model retraining pipeline
├── Dockerfile             # API container
├── Dockerfile.dashboard   # Dashboard container
├── docker-compose.yml     # Multi-service orchestration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## API Endpoints

### Health & Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check with model status |

### Predictions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single customer prediction |
| `/predict/explain` | POST | Prediction with SHAP explanation |
| `/predict/compare` | POST | Compare RandomForest vs XGBoost |
| `/predict/batch` | POST | Batch predictions (JSON) |
| `/predict/batch/csv` | POST | Batch predictions from CSV upload |
| `/predict/batch/dataset` | POST | Predict on entire dataset |
| `/predict/customer/{id}` | GET | Get specific customer prediction |

### Analytics & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analytics/feature-importance` | GET | Global feature importance |
| `/analytics/churn-profile` | GET | Churn rates by segments |
| `/alerts/check` | GET | High-risk customer alerts |
| `/monitoring/drift` | GET | Data drift detection |

## Usage Examples

### Single Prediction
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
})

print(response.json())
# {"churn_probability": 0.2341, "churn_prediction": 0, "churn_label": "No"}
```

### Model Comparison
```python
response = requests.post("http://localhost:8000/predict/compare", json={...})
# Returns predictions from both RandomForest and XGBoost with agreement status
```

### Check Data Drift
```python
response = requests.get("http://localhost:8000/monitoring/drift")
# Returns drift metrics for numeric features
```

## Dashboard Features

1. **🔮 Single Prediction** - Input customer data and get instant churn prediction with SHAP explanation
2. **📁 Batch Prediction** - Upload CSV or use built-in dataset for bulk predictions
3. **👤 Customer Details** - View detailed breakdown for individual customers
4. **📈 Analytics** - Feature importance and churn rate analysis by segments
5. **⚖️ Model Comparison** - Side-by-side comparison of RandomForest vs XGBoost
6. **🚨 Monitoring** - Data drift detection and high-risk customer alerts

## Development

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### Code Formatting
```bash
# Check formatting
black --check src/ api/ dashboard/ tests/

# Auto-format
black src/ api/ dashboard/ tests/
```

### Training Models
```bash
# Train both RandomForest and XGBoost
python src/train.py

# Models will be saved to models/ directory
# MLflow will log metrics to mlruns/
```

## Key Design Decisions

See [doc/decisions.md](doc/decisions.md) for detailed rationale behind technology choices.

### Why These Choices?

| Choice | Rationale |
|--------|-----------|
| **scikit-learn + XGBoost** | Industry standard for tabular data, good performance |
| **FastAPI** | High performance, automatic API docs, async support |
| **Streamlit** | Rapid prototyping, pure Python, ML-friendly |
| **Docker** | Reproducible environments, easy deployment |
| **MLflow** | Open-source experiment tracking, model registry |
| **SHAP** | Gold standard for model interpretability |
| **GitHub Actions** | Native integration, free for public repos |

## Model Performance

Target metrics (achieved on test set):

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | >80% | Overall correct predictions |
| ROC AUC | >0.85 | Area under ROC curve |
| F1 Score | >0.75 | Balance of precision and recall |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

- GitHub: [@GucciPants](https://github.com/GucciPants)
- Project Link: https://github.com/GucciPants/churn-prediction-mlops
