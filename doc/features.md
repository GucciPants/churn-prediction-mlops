# New Features Documentation

## Overview

This document describes three major features added to the churn prediction system to enhance model interpretability, automation, and analytics capabilities.

---

## Feature 1: 🔬 SHAP Model Explanations

### Purpose
Provide interpretable explanations for every prediction, answering: **"Why did the model predict this?"**

### Implementation

**Files Added/Modified:**
- `src/explainer.py` - New SHAP explainer module
- `src/train.py` - Modified to generate and save SHAP explainers during training
- `api/main.py` - New endpoint: `POST /predict/explain`
- `dashboard/app.py` - SHAP visualization in Single Prediction tab
- `requirements.txt` - Added `shap==0.45.0`

**How It Works:**
1. During training, a `TreeExplainer` is created for the RandomForest model
2. The explainer is saved to `models/random_forest_explainer.pkl`
3. API endpoint accepts the same input as prediction and returns:
   - Prediction result (probability, label)
   - Top 5 features pushing toward churn (positive contributions)
   - Top 5 features pushing against churn (negative contributions)
   - Full SHAP value breakdown for all features

**API Response Example:**
```json
{
  "prediction": {
    "churn_probability": 0.7266,
    "churn_prediction": 1,
    "churn_label": "Yes"
  },
  "explanation": {
    "base_value": 0.2654,
    "top_positive": [
      {"feature": "tenure", "value": 1.0, "contribution": 0.1523},
      {"feature": "Contract", "value": 0.0, "contribution": 0.0891}
    ],
    "top_negative": [
      {"feature": "OnlineSecurity", "value": 1.0, "contribution": -0.0342}
    ]
  }
}
```

**Dashboard Display:**
- "Why This Prediction?" expandable section below prediction results
- Bar chart showing SHAP contributions for top features
- Lists of positive and negative indicators

---

## Feature 2: 🔄 Automated Retraining

### Purpose
Automate model retraining on a schedule to ensure the model stays up-to-date.

### Implementation

**Files Added:**
- `.github/workflows/retrain.yml` - New GitHub Actions workflow

**Schedule:**
- **Automatic:** Every Sunday at 2:00 AM UTC (`0 2 * * 0`)
- **Manual:** `workflow_dispatch` trigger for on-demand retraining

**Workflow Steps:**
1. Checkout repository
2. Setup Python 3.11
3. Install dependencies
4. Run `src/train.py` to retrain models
5. Execute test suite (`pytest tests/`)
6. Check code formatting (`black --check`)
7. Build Docker images (API + Dashboard)
8. Upload trained model artifacts
9. Notify success/failure

**Artifacts:**
- Trained models are uploaded as GitHub Actions artifacts
- Retention: 30 days
- Includes: `random_forest.pkl`, `xgboost.pkl`, `scaler.pkl`, `*_explainer.pkl`

**Usage:**
```bash
# Manual trigger via GitHub UI:
# Actions → Scheduled Retraining → Run workflow
```

---

## Feature 3: 📊 Churn Analytics Dashboard

### Purpose
Provide visual analytics showing feature importance and churn profiles across different dimensions.

### Implementation

**Files Modified:**
- `api/main.py` - Two new endpoints
- `dashboard/app.py` - New "📈 Analytics" tab

**API Endpoints:**

#### `GET /analytics/feature-importance`
Returns top 15 most important features from the RandomForest model.

```json
{
  "model": "random_forest",
  "features": [
    {"feature": "tenure", "importance": 0.185},
    {"feature": "Contract", "importance": 0.142},
    {"feature": "MonthlyCharges", "importance": 0.112}
  ]
}
```

#### `GET /analytics/churn-profile`
Returns churn rates grouped by different dimensions.

```json
{
  "by_contract": [
    {"group": "Month-to-month", "total": 3875, "churned": 1655, "churn_rate": 42.7},
    {"group": "One year", "total": 1473, "churned": 166, "churn_rate": 11.3},
    {"group": "Two year", "total": 1695, "churned": 48, "churn_rate": 2.8}
  ],
  "by_tenure_bucket": [...],
  "by_payment_method": [...],
  "by_internet_service": [...]
}
```

**Dashboard Tab - "📈 Analytics":**

**Left Column:**
- Feature Importance bar chart (top 10)
- Expandable table with all features

**Right Column:**
- Churn Rate by Contract Type
- Churn Rate by Tenure Bucket

**Bottom Section:**
- Detailed tables for Payment Method and Internet Service breakdowns

**Key Insights Displayed:**
- Month-to-month contracts have ~43% churn rate vs 3% for two-year contracts
- Customers with 0-12 months tenure have the highest churn (~50%)
- Electronic check payment method correlates with higher churn

---

## Testing Notes

### SHAP Feature
- Run training first: `python src/train.py` or `docker-compose exec api python src/train.py`
- This generates the explainer `.pkl` files required for explanations
- If explainer is missing, the dashboard shows a friendly info message

### Retraining Feature
- Workflow triggers automatically every Sunday
- Manual trigger available in GitHub Actions UI
- Artifacts are downloadable from the workflow run page

### Analytics Feature
- Reads from built-in dataset (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- No additional training required
- Updates automatically when dataset changes

---

## Architecture Impact

```
┌─────────────────────────────────────────────────────────────┐
│                    Updated Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Pipeline                                          │
│  ├── src/train.py (generates models + SHAP explainers)     │
│  └── .github/workflows/retrain.yml (weekly automation)      │
│                                                             │
│  API Endpoints                                              │
│  ├── POST /predict (existing)                              │
│  ├── POST /predict/explain (NEW - SHAP explanations)       │
│  ├── POST /predict/batch/csv (existing)                    │
│  ├── GET /predict/customer/{id} (existing)                 │
│  ├── GET /analytics/feature-importance (NEW)               │
│  └── GET /analytics/churn-profile (NEW)                    │
│                                                             │
│  Dashboard Tabs                                             │
│  ├── 🔮 Single Prediction (+ SHAP explanations)            │
│  ├── 📁 Batch Prediction                                    │
│  ├── 👤 Customer Details                                    │
│  └── 📈 Analytics (NEW - Feature importance + profiles)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

---

## Feature 4: ⚖️ Model Comparison Dashboard

### Purpose
Compare predictions between RandomForest and XGBoost models side-by-side to assess model agreement and confidence.

### Implementation

**API Endpoint:** `POST /predict/compare`

**Returns:**
```json
{
  "random_forest": {
    "churn_probability": 0.7266,
    "churn_prediction": 1,
    "churn_label": "Yes"
  },
  "xgboost": {
    "churn_probability": 0.6891,
    "churn_prediction": 1,
    "churn_label": "Yes"
  },
  "comparison": {
    "models_agree": true,
    "confidence_difference": 0.0375,
    "average_probability": 0.7079
  }
}
```

**Dashboard:** "⚖️ Model Comparison" tab
- Input form (same as Single Prediction)
- Side-by-side results display
- Probability comparison bar chart
- Agreement indicator (green/red)

---

## Feature 5: 🚨 Alerts System

### Purpose
Monitor customer base for high churn risk and trigger alerts when thresholds are exceeded.

### Implementation

**API Endpoint:** `GET /alerts/check?threshold=0.5`

**Returns:**
```json
{
  "threshold": 0.5,
  "total_customers": 7043,
  "high_risk_count": 1867,
  "high_risk_percentage": 26.51,
  "top_risk_customers": [...],
  "alert_triggered": true
}
```

**Features:**
- Configurable probability threshold (default: 0.5)
- Returns top 10 highest risk customers
- Shows percentage of at-risk customer base
- Dashboard integration in Model Comparison tab

---

## Feature 6: 📊 Data Drift Detection

### Purpose
Monitor input data distribution changes over time to detect when retraining might be needed.

### Implementation

**API Endpoint:** `GET /monitoring/drift`

**Methodology:**
- Compares current dataset statistics against known baseline
- Tracks: tenure, MonthlyCharges, TotalCharges
- Calculates drift score: |current_mean - baseline_mean| / baseline_std
- Drift detected if score > 0.5

**Returns:**
```json
{
  "drift_detected": false,
  "drift_metrics": {
    "tenure": {
      "current_mean": 32.37,
      "baseline_mean": 32.37,
      "drift_score": 0.0,
      "drift_detected": false
    }
  }
}
```

**Dashboard Integration:**
- Real-time drift check button
- Visual indicators (✅/🚨) per feature
- Current vs baseline comparison display

---

## Updated Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Complete Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  API Endpoints                                                   │
│  ├── POST /predict (single prediction)                          │
│  ├── POST /predict/explain (SHAP explanations)                  │
│  ├── POST /predict/compare (RF vs XGBoost)                      │
│  ├── POST /predict/batch/csv (batch processing)                 │
│  ├── GET /predict/customer/{id} (customer details)              │
│  ├── GET /analytics/feature-importance                          │
│  ├── GET /analytics/churn-profile                               │
│  ├── GET /alerts/check (high risk monitoring)                   │
│  └── GET /monitoring/drift (data drift detection)               │
│                                                                  │
│  Dashboard Tabs                                                  │
│  ├── 🔮 Single Prediction (+ SHAP)                              │
│  ├── 📁 Batch Prediction                                        │
│  ├── 👤 Customer Details                                        │
│  ├── 📈 Analytics (Feature importance + profiles)               │
│  └── ⚖️ Model Comparison (+ Alerts & Drift)                    │
│                                                                  │
│  CI/CD Workflows                                                 │
│  ├── ci.yml (test + build on push)                              │
│  └── retrain.yml (weekly automated retraining)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Future Enhancements

Potential next features:

1. **Prediction History Database** - Store all predictions over time
2. **A/B Testing Framework** - Compare model versions in production
3. **Email/Slack Webhooks** - Automated notifications for alerts
4. **Advanced Drift Detection** - Statistical tests (KS-test, PSI)
5. **Model Performance Monitoring** - Track accuracy degradation over time
