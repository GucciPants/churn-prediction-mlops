"""FastAPI backend for churn prediction."""

import io
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Load model and scaler
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Global artifacts
model = None
scaler = None


def load_artifacts():
    """Load trained model and scaler."""
    global model, scaler
    model_path = MODELS_DIR / "random_forest.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not model_path.exists():
        print(
            f"WARNING: Model not found at {model_path}. Predictions will not work until training is run."
        )
        return
    if not scaler_path.exists():
        print(
            f"WARNING: Scaler not found at {scaler_path}. Predictions will not work until training is run."
        )
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    load_artifacts()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn based on telecom features",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerData(BaseModel):
    """Input schema for single prediction."""

    tenure: int = Field(..., ge=0, le=100, description="Number of months as a customer")
    MonthlyCharges: float = Field(..., ge=0, le=200, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, le=10000, description="Total charges")
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(..., pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No|No internet service)$")
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(
        ...,
        pattern=r"^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$",
    )


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""

    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    """Output schema for predictions."""

    churn_probability: float
    churn_prediction: int
    churn_label: str


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""

    predictions: List[PredictionResponse]


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    df = pd.DataFrame([data])

    # Encode binary
    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_mappings.items():
        df[col] = df[col].map(mapping)

    # One-hot encode categoricals
    cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align columns with training data
    expected_cols = list(model.feature_names_in_)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Scale numeric
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerData):
    """Predict churn for a single customer."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_df = preprocess_input(customer.dict())
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        return PredictionResponse(
            churn_probability=round(float(prob), 4),
            churn_prediction=int(pred),
            churn_label="Yes" if pred == 1 else "No",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"]
)
async def predict_batch(batch: BatchPredictionInput):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    predictions = []
    try:
        for customer in batch.customers:
            input_df = preprocess_input(customer.dict())
            prob = model.predict_proba(input_df)[0, 1]
            pred = model.predict(input_df)[0]

            predictions.append(
                PredictionResponse(
                    churn_probability=round(float(prob), 4),
                    churn_prediction=int(pred),
                    churn_label="Yes" if pred == 1 else "No",
                )
            )

        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch/csv", tags=["Predictions"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """Predict churn for multiple customers from CSV file."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate required columns
        required_cols = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}",
            )

        # Clean data (handle TotalCharges spaces, etc.)
        df_clean = df.copy()
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"], errors="coerce"
        )
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0)
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

        # Encode binary features
        binary_mappings = {
            "gender": {"Female": 0, "Male": 1},
            "Partner": {"No": 0, "Yes": 1},
            "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1},
            "PaperlessBilling": {"No": 0, "Yes": 1},
        }
        for col, mapping in binary_mappings.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(mapping)

        # One-hot encode categoricals
        cat_cols = [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ]
        df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

        # Align columns with training data
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_cols]

        # Scale numeric
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

        # Predict for all rows
        probs = model.predict_proba(df_encoded)[:, 1]
        preds = model.predict(df_encoded)

        # Build results
        results = []
        for prob, pred in zip(probs, preds):
            results.append(
                {
                    "churn_probability": round(float(prob), 4),
                    "churn_prediction": int(pred),
                    "churn_label": "Yes" if pred == 1 else "No",
                    "risk_level": (
                        "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low"
                    ),
                }
            )

        # Add results to original dataframe
        results_df = pd.DataFrame(results)
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # Convert to CSV
        output = io.StringIO()
        output_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Run batch predictions on a dataframe."""
    df_clean = df.copy()
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0)
    df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_mappings.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping)

    cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    expected_cols = list(model.feature_names_in_)
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_cols]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    probs = model.predict_proba(df_encoded)[:, 1]
    preds = model.predict(df_encoded)

    results = []
    for prob, pred in zip(probs, preds):
        results.append(
            {
                "churn_probability": round(float(prob), 4),
                "churn_prediction": int(pred),
                "churn_label": "Yes" if pred == 1 else "No",
                "risk_level": (
                    "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low"
                ),
            }
        )

    results_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), results_df], axis=1)


@app.post("/predict/batch/dataset", tags=["Predictions"])
async def predict_batch_dataset():
    """Predict churn for all customers in the built-in dataset."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        dataset_path = Path("/app/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Built-in dataset not found")

        df = pd.read_csv(dataset_path)
        output_df = run_predictions(df)

        # Return as JSON
        records = output_df.to_dict(orient="records")
        return {"total": len(records), "predictions": records}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/customer/{customer_id}", tags=["Predictions"])
async def get_customer_prediction(customer_id: str):
    """Get prediction details for a specific customer from the dataset."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        dataset_path = Path("/app/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Built-in dataset not found")

        df = pd.read_csv(dataset_path)
        customer_df = df[df["customerID"] == customer_id]

        if customer_df.empty:
            raise HTTPException(
                status_code=404, detail=f"Customer {customer_id} not found"
            )

        output_df = run_predictions(customer_df)
        record = output_df.iloc[0].to_dict()

        return record

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }
