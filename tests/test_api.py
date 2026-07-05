"""Tests for FastAPI endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))


# Sample valid customer data for testing
SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
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
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
}


class TestHealthEndpoint:
    def test_health_check(self):
        from api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "scaler_loaded" in data

    def test_health_contains_required_fields(self):
        from api.main import app

        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        required_fields = ["status", "model_loaded", "scaler_loaded", "explainer_loaded"]
        for field in required_fields:
            assert field in data


class TestRootEndpoint:
    def test_root(self):
        from api.main import app

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert data["docs"] == "/docs"


class TestPredictionValidation:
    """Test input validation for prediction endpoints."""

    def test_missing_required_field(self):
        from api.main import app

        client = TestClient(app)
        incomplete_data = {"gender": "Male"}  # Missing most fields
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_invalid_gender(self):
        from api.main import app

        client = TestClient(app)
        invalid_data = SAMPLE_CUSTOMER.copy()
        invalid_data["gender"] = "Invalid"
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_invalid_contract_type(self):
        from api.main import app

        client = TestClient(app)
        invalid_data = SAMPLE_CUSTOMER.copy()
        invalid_data["Contract"] = "Invalid Contract"
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_invalid_tenure_negative(self):
        from api.main import app

        client = TestClient(app)
        invalid_data = SAMPLE_CUSTOMER.copy()
        invalid_data["tenure"] = -1
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_invalid_monthly_charges_too_high(self):
        from api.main import app

        client = TestClient(app)
        invalid_data = SAMPLE_CUSTOMER.copy()
        invalid_data["MonthlyCharges"] = 300.0  # Max is 200
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422


class TestPredictionEndpoint:
    @pytest.mark.skipif(
        not Path("models/random_forest.pkl").exists(),
        reason="Model not trained yet",
    )
    def test_predict_valid_input(self):
        from api.main import app

        client = TestClient(app)
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "churn_label" in data
        assert data["churn_label"] in ["Yes", "No"]
        assert 0 <= data["churn_probability"] <= 1

    @pytest.mark.skipif(
        not Path("models/random_forest.pkl").exists(),
        reason="Model not trained yet",
    )
    def test_predict_returns_float_probability(self):
        from api.main import app

        client = TestClient(app)
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = response.json()
        assert isinstance(data["churn_probability"], float)
        assert isinstance(data["churn_prediction"], int)


class TestBatchPredictionEndpoint:
    @pytest.mark.skipif(
        not Path("models/random_forest.pkl").exists(),
        reason="Model not trained yet",
    )
    def test_batch_predict(self):
        from api.main import app

        client = TestClient(app)
        batch = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_batch_predict_empty_list(self):
        from api.main import app

        client = TestClient(app)
        batch = {"customers": []}
        response = client.post("/predict/batch", json=batch)
        # Empty list should still return 200 with empty predictions
        assert response.status_code == 200


class TestModelComparisonEndpoint:
    @pytest.mark.skipif(
        not Path("models/xgboost.pkl").exists(),
        reason="XGBoost model not trained yet",
    )
    def test_compare_models(self):
        from api.main import app

        client = TestClient(app)
        response = client.post("/predict/compare", json=SAMPLE_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert "random_forest" in data
        assert "xgboost" in data
        assert "comparison" in data
        assert "models_agree" in data["comparison"]


class TestFeatureImportanceEndpoint:
    @pytest.mark.skipif(
        not Path("models/random_forest.pkl").exists(),
        reason="Model not trained yet",
    )
    def test_get_feature_importance(self):
        from api.main import app

        client = TestClient(app)
        response = client.get("/analytics/feature-importance")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "model" in data
        assert len(data["features"]) > 0


class TestChurnProfileEndpoint:
    def test_get_churn_profile(self):
        from api.main import app

        client = TestClient(app)
        response = client.get("/analytics/churn-profile")
        assert response.status_code == 200
        data = response.json()
        assert "by_contract" in data
        assert "by_tenure_bucket" in data
        assert "by_payment_method" in data
        assert "by_internet_service" in data


class TestDriftMonitoring:
    def test_check_drift(self):
        from api.main import app

        client = TestClient(app)
        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "drift_metrics" in data
        assert "check_timestamp" in data
