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

## Future Enhancements

Potential next features to consider:

1. **Model Comparison Dashboard** - Side-by-side RandomForest vs XGBoost performance
2. **Prediction History** - Track predictions over time in a database
3. **A/B Testing Framework** - Compare model versions in production
4. **Data Drift Detection** - Monitor for changes in input data distribution
5. **Email/Slack Alerts** - Notifications when churn rate exceeds thresholds
