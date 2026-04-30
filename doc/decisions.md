# Key Decisions & Rationale

## 1. Dataset Choice
**Decision:** Use the IBM/Kaggle Telco Customer Churn dataset.  
**Rationale:** Free, well-known, clean, and contains a clear binary target. It avoids the need to generate synthetic data or manage complex data collection.

## 2. Programming Language
**Decision:** Python.  
**Rationale:** De-facto standard for ML and data science. Excellent libraries (scikit-learn, pandas, FastAPI, Streamlit).

## 3. Model Choice
**Decision:** Start with a `RandomForestClassifier` and compare against `XGBoost`.  
**Rationale:** Both are robust for tabular data, handle non-linear relationships well, and provide feature importance. They are easy to interpret and deploy.

## 4. Experiment Tracking
**Decision:** MLflow.  
**Rationale:** Open-source, integrates seamlessly with scikit-learn/XGBoost, and provides a UI for comparing experiments. It also supports model registry for versioning.

## 5. API Framework
**Decision:** FastAPI.  
**Rationale:** High performance, easy to use, automatic interactive API documentation (Swagger UI), and async support.

## 6. Frontend
**Decision:** Streamlit.  
**Rationale:** Pure Python, extremely fast to build data apps, integrates directly with pandas and ML models. Ideal for ML demos and internal tools.

## 7. Containerization
**Decision:** Docker + Docker Compose.  
**Rationale:** Ensures the environment is reproducible across development and production. Compose orchestrates the API, Dashboard, and MLflow UI.

## 8. CI/CD
**Decision:** GitHub Actions.  
**Rationale:** Native integration with GitHub. Free for public repositories. Automates running tests and building Docker images on every push.

## 9. Project Language
**Decision:** English.  
**Rationale:** Global standard for open-source projects and portfolios. Improves accessibility for recruiters and collaborators worldwide.
