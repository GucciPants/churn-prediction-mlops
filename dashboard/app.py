"""Streamlit dashboard for churn prediction."""

import io

import pandas as pd
import requests
import streamlit as st

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict whether telecom customers are likely to churn.")

# Session state
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None

# Tabs for single and batch prediction
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔮 Single Prediction",
        "📁 Batch Prediction",
        "👤 Customer Details",
        "📈 Analytics",
        "⚖️ Model Comparison",
    ]
)


with tab1:
    st.header("Single Customer Prediction")

    # Sidebar for single prediction inputs
    st.sidebar.header("Customer Features")

    # Use session state if available
    sf = st.session_state.selected_features

    # Demographics
    gender = st.sidebar.selectbox(
        "Gender",
        ["Male", "Female"],
        index=1 if sf and sf.get("gender") == "Female" else 0,
    )
    senior = st.sidebar.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        index=sf.get("SeniorCitizen", 0) if sf else 0,
    )
    partner = st.sidebar.selectbox(
        "Partner", ["Yes", "No"], index=1 if sf and sf.get("Partner") == "No" else 0
    )
    dependents = st.sidebar.selectbox(
        "Dependents",
        ["Yes", "No"],
        index=1 if sf and sf.get("Dependents") == "No" else 0,
    )

    # Services
    tenure = st.sidebar.slider(
        "Tenure (months)", 0, 100, sf.get("tenure", 12) if sf else 12
    )
    phone_service = st.sidebar.selectbox(
        "Phone Service",
        ["Yes", "No"],
        index=1 if sf and sf.get("PhoneService") == "No" else 0,
    )
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"],
        index=(
            2
            if sf and sf.get("MultipleLines") == "No phone service"
            else (1 if sf and sf.get("MultipleLines") == "No" else 0)
        ),
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        index=(
            2
            if sf and sf.get("InternetService") == "No"
            else (1 if sf and sf.get("InternetService") == "Fiber optic" else 0)
        ),
    )
    online_security = st.sidebar.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"],
        index=(
            2
            if sf and sf.get("OnlineSecurity") == "No internet service"
            else (1 if sf and sf.get("OnlineSecurity") == "No" else 0)
        ),
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"],
        index=(
            2
            if sf and sf.get("OnlineBackup") == "No internet service"
            else (1 if sf and sf.get("OnlineBackup") == "No" else 0)
        ),
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"],
        index=(
            2
            if sf and sf.get("DeviceProtection") == "No internet service"
            else (1 if sf and sf.get("DeviceProtection") == "No" else 0)
        ),
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"],
        index=(
            2
            if sf and sf.get("TechSupport") == "No internet service"
            else (1 if sf and sf.get("TechSupport") == "No" else 0)
        ),
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"],
        index=(
            2
            if sf and sf.get("StreamingTV") == "No internet service"
            else (1 if sf and sf.get("StreamingTV") == "No" else 0)
        ),
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"],
        index=(
            2
            if sf and sf.get("StreamingMovies") == "No internet service"
            else (1 if sf and sf.get("StreamingMovies") == "No" else 0)
        ),
    )

    # Billing
    contract = st.sidebar.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
        index=(
            2
            if sf and sf.get("Contract") == "Two year"
            else (1 if sf and sf.get("Contract") == "One year" else 0)
        ),
    )
    paperless = st.sidebar.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
        index=1 if sf and sf.get("PaperlessBilling") == "No" else 0,
    )
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        index=(
            3
            if sf and sf.get("PaymentMethod") == "Credit card (automatic)"
            else (
                2
                if sf and sf.get("PaymentMethod") == "Bank transfer (automatic)"
                else (1 if sf and sf.get("PaymentMethod") == "Mailed check" else 0)
            )
        ),
    )
    monthly_charges = st.sidebar.slider(
        "Monthly Charges ($)",
        0.0,
        150.0,
        float(sf.get("MonthlyCharges", 70.0)) if sf else 70.0,
    )
    total_charges = st.sidebar.slider(
        "Total Charges ($)",
        0.0,
        9000.0,
        float(sf.get("TotalCharges", 1000.0)) if sf else 1000.0,
    )

    # Prediction button
    if st.sidebar.button("Predict Churn", type="primary"):
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Prediction", result["churn_label"])
            with col2:
                st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            with col3:
                risk = (
                    "High"
                    if result["churn_probability"] > 0.5
                    else "Medium" if result["churn_probability"] > 0.3 else "Low"
                )
                st.metric("Risk Level", risk)

            # Visualization
            st.subheader("Prediction Visualization")
            prob = result["churn_probability"]
            st.progress(prob)

            # SHAP Explanation
            st.subheader("🔍 Why This Prediction?")
            try:
                explain_response = requests.post(
                    f"{API_URL}/predict/explain", json=payload
                )
                if explain_response.status_code == 200:
                    explanation = explain_response.json()["explanation"]

                    with st.expander("View SHAP Explanation", expanded=True):
                        st.markdown("**Features pushing toward churn:**")
                        if explanation["top_positive"]:
                            for feat in explanation["top_positive"]:
                                st.markdown(
                                    f"- ↑ **{feat['feature']}**: {feat['value']:.2f} "
                                    f"(contribution: +{feat['contribution']:.4f})"
                                )
                        else:
                            st.markdown("_No strong churn indicators_")

                        st.markdown("**Features pushing against churn:**")
                        if explanation["top_negative"]:
                            for feat in explanation["top_negative"]:
                                st.markdown(
                                    f"- ↓ **{feat['feature']}**: {feat['value']:.2f} "
                                    f"(contribution: {feat['contribution']:.4f})"
                                )
                        else:
                            st.markdown("_No strong retention indicators_")

                        # Waterfall chart placeholder
                        st.markdown("---")
                        st.markdown("**SHAP Value Breakdown**")
                        chart_data = {
                            "Feature": [
                                f["feature"] for f in explanation["shap_values"]
                            ],
                            "Contribution": [
                                f["contribution"] for f in explanation["shap_values"]
                            ],
                        }
                        st.bar_chart(
                            chart_data,
                            x="Feature",
                            y="Contribution",
                            color="Contribution",
                        )
                else:
                    st.info(
                        "SHAP explanations not available. Run training to generate the explainer."
                    )
            except Exception as e:
                st.warning(f"Could not load explanation: {e}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # Clear selected features after using
    if st.session_state.selected_features:
        st.session_state.selected_features = None


with tab2:
    st.header("Batch Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Built-in Dataset")
        st.markdown("Run predictions on all **7,043 customers** in the dataset.")
        if st.button(
            "🔮 Analyze All Customers", type="primary", use_container_width=True
        ):
            with st.spinner("Processing all customers..."):
                try:
                    response = requests.post(f"{API_URL}/predict/batch/dataset")
                    if response.status_code == 200:
                        data = response.json()
                        results_df = pd.DataFrame(data["predictions"])
                        st.session_state.prediction_results = results_df
                        st.success(
                            f"Predictions complete for {len(results_df)} customers!"
                        )
                    else:
                        st.error(
                            f"API Error: {response.json().get('detail', 'Unknown error')}"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("Upload CSV")
        st.markdown("Or upload your own customer data.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file)
            st.info(f"Total rows: {len(df_preview)}")

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
            missing_cols = [
                col for col in required_cols if col not in df_preview.columns
            ]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button(
                    "🔮 Run Predictions", type="primary", use_container_width=True
                ):
                    with st.spinner("Processing predictions..."):
                        try:
                            uploaded_file.seek(0)
                            files = {
                                "file": ("customers.csv", uploaded_file, "text/csv")
                            }
                            response = requests.post(
                                f"{API_URL}/predict/batch/csv", files=files
                            )

                            if response.status_code == 200:
                                results_df = pd.read_csv(
                                    io.StringIO(response.content.decode("utf-8"))
                                )
                                st.session_state.prediction_results = results_df
                                st.success(
                                    f"Predictions complete for {len(results_df)} customers!"
                                )
                            else:
                                st.error(
                                    f"API Error: {response.json().get('detail', 'Unknown error')}"
                                )
                        except Exception as e:
                            st.error(f"Error processing predictions: {e}")

    # Display results if available
    if st.session_state.prediction_results is not None:
        results_df = st.session_state.prediction_results

        st.markdown("---")

        # Statistics
        st.subheader("Statistics")
        col1, col2, col3, col4 = st.columns(4)

        total = len(results_df)
        churn_yes = len(results_df[results_df["churn_label"] == "Yes"])
        churn_no = total - churn_yes
        churn_rate = (churn_yes / total) * 100

        with col1:
            st.metric("Total Customers", total)
        with col2:
            st.metric("Churn (Yes)", churn_yes)
        with col3:
            st.metric("Retain (No)", churn_no)
        with col4:
            st.metric("Churn Rate", f"{churn_rate:.1f}%")

        # Risk distribution
        st.subheader("Risk Distribution")
        risk_counts = results_df["risk_level"].value_counts()
        st.bar_chart(risk_counts)

        # Results table with selection
        st.subheader("Customer Results")

        # Search/filter
        search_term = st.text_input("Search by customerID", "")
        if search_term:
            filtered_df = results_df[
                results_df["customerID"]
                .astype(str)
                .str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered_df = results_df

        st.dataframe(
            filtered_df,
            column_order=[
                "customerID",
                "churn_label",
                "churn_probability",
                "risk_level",
                "tenure",
                "Contract",
                "MonthlyCharges",
            ],
            use_container_width=True,
            hide_index=True,
        )

        # Customer selection for details
        st.subheader("Select Customer for Details")
        customer_options = filtered_df["customerID"].tolist()
        selected_id = st.selectbox(
            "Choose a customer", customer_options, index=0 if customer_options else None
        )

        if selected_id and st.button("👤 View Details", type="primary"):
            st.session_state.selected_customer = selected_id
            st.rerun()

        # Download button
        st.subheader("Download Results")
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )


with tab3:
    st.header("Customer Details")

    if st.session_state.selected_customer:
        customer_id = st.session_state.selected_customer

        with st.spinner(f"Loading details for {customer_id}..."):
            try:
                response = requests.get(f"{API_URL}/predict/customer/{customer_id}")
                if response.status_code == 200:
                    customer = response.json()

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader(f"Customer: {customer_id}")

                        # Prediction card
                        with st.container():
                            st.markdown("#### 📊 Prediction")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Churn", customer["churn_label"])
                            with c2:
                                st.metric(
                                    "Probability",
                                    f"{customer['churn_probability']:.2%}",
                                )
                            with c3:
                                st.metric("Risk", customer["risk_level"])

                        # Progress bar
                        st.progress(customer["churn_probability"])

                        # Demographics
                        with st.container():
                            st.markdown("#### 👤 Demographics")
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("Gender", customer["gender"])
                            with c2:
                                st.metric(
                                    "Senior",
                                    (
                                        "Yes"
                                        if customer["SeniorCitizen"] in ["1", 1, "Yes"]
                                        else "No"
                                    ),
                                )
                            with c3:
                                st.metric("Partner", customer["Partner"])
                            with c4:
                                st.metric("Dependents", customer["Dependents"])

                        # Services
                        with st.container():
                            st.markdown("#### 📞 Services")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Tenure", f"{customer['tenure']} months")
                            with c2:
                                st.metric("Internet", customer["InternetService"])
                            with c3:
                                st.metric("Contract", customer["Contract"])

                            c4, c5, c6, c7 = st.columns(4)
                            with c4:
                                st.metric("Phone", customer["PhoneService"])
                            with c5:
                                st.metric("Tech Support", customer["TechSupport"])
                            with c6:
                                st.metric("Online Security", customer["OnlineSecurity"])
                            with c7:
                                st.metric("Backup", customer["OnlineBackup"])

                        # Billing
                        with st.container():
                            st.markdown("#### 💳 Billing")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Monthly", f"${customer['MonthlyCharges']}")
                            with c2:
                                st.metric("Total", f"${customer['TotalCharges']}")
                            with c3:
                                st.metric("Payment", customer["PaymentMethod"])

                    with col2:
                        st.subheader("Actions")

                        if st.button(
                            "🔮 Predict in Single Mode",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.selected_features = customer
                            st.session_state.selected_customer = None
                            st.rerun()

                        if st.button(
                            "⬅️ Back to Batch Results", use_container_width=True
                        ):
                            st.session_state.selected_customer = None
                            st.rerun()

                else:
                    st.error(
                        f"API Error: {response.json().get('detail', 'Unknown error')}"
                    )
            except Exception as e:
                st.error(f"Error loading customer details: {e}")
    else:
        st.info("Select a customer from the Batch Prediction tab to view details here.")


with tab4:
    st.header("📈 Churn Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")
        try:
            response = requests.get(f"{API_URL}/analytics/feature-importance")
            if response.status_code == 200:
                data = response.json()
                features_df = pd.DataFrame(data["features"])

                # Display as bar chart
                st.bar_chart(
                    features_df.head(10),
                    x="feature",
                    y="importance",
                    use_container_width=True,
                )

                # Show table
                with st.expander("View all features"):
                    st.dataframe(features_df, use_container_width=True)
            else:
                st.error("Failed to load feature importance")
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("Churn Profiles")
        try:
            response = requests.get(f"{API_URL}/analytics/churn-profile")
            if response.status_code == 200:
                data = response.json()

                # Contract type
                st.markdown("**By Contract Type**")
                contract_df = pd.DataFrame(data["by_contract"])
                st.bar_chart(
                    contract_df,
                    x="group",
                    y="churn_rate",
                    use_container_width=True,
                )

                # Tenure
                st.markdown("**By Tenure**")
                tenure_df = pd.DataFrame(data["by_tenure_bucket"])
                st.bar_chart(
                    tenure_df,
                    x="group",
                    y="churn_rate",
                    use_container_width=True,
                )
            else:
                st.error("Failed to load churn profiles")
        except Exception as e:
            st.error(f"Error: {e}")

    # Detailed tables
    st.markdown("---")
    st.subheader("Detailed Breakdown")

    try:
        response = requests.get(f"{API_URL}/analytics/churn-profile")
        if response.status_code == 200:
            data = response.json()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Payment Method**")
                payment_df = pd.DataFrame(data["by_payment_method"])
                st.dataframe(payment_df, use_container_width=True)

            with c2:
                st.markdown("**Internet Service**")
                internet_df = pd.DataFrame(data["by_internet_service"])
                st.dataframe(internet_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading details: {e}")

with tab5:
    st.header("⚖️ Model Comparison")
    st.markdown("Compare predictions between RandomForest and XGBoost models.")

    # Use same inputs as single prediction
    st.subheader("Enter Customer Data")

    col1, col2 = st.columns(2)
    with col1:
        cmp_gender = st.selectbox("Gender", ["Male", "Female"], key="cmp_gender")
        cmp_senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="cmp_senior",
        )
        cmp_partner = st.selectbox("Partner", ["Yes", "No"], key="cmp_partner")
        cmp_dependents = st.selectbox("Dependents", ["Yes", "No"], key="cmp_dependents")
        cmp_tenure = st.slider("Tenure (months)", 0, 100, 12, key="cmp_tenure")
        cmp_phone = st.selectbox("Phone Service", ["Yes", "No"], key="cmp_phone")
        cmp_multiple = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"], key="cmp_multiple"
        )
        cmp_internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"], key="cmp_internet"
        )
        cmp_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"], key="cmp_security"
        )
        cmp_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"], key="cmp_backup"
        )

    with col2:
        cmp_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
            key="cmp_protection",
        )
        cmp_tech = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"], key="cmp_tech"
        )
        cmp_streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"], key="cmp_streaming_tv"
        )
        cmp_streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
            key="cmp_streaming_movies",
        )
        cmp_contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"], key="cmp_contract"
        )
        cmp_paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"], key="cmp_paperless"
        )
        cmp_payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="cmp_payment",
        )
        cmp_monthly = st.slider(
            "Monthly Charges ($)", 0.0, 150.0, 70.0, key="cmp_monthly"
        )
        cmp_total = st.slider("Total Charges ($)", 0.0, 9000.0, 1000.0, key="cmp_total")

    if st.button("🔮 Compare Models", type="primary"):
        payload = {
            "gender": cmp_gender,
            "SeniorCitizen": cmp_senior,
            "Partner": cmp_partner,
            "Dependents": cmp_dependents,
            "tenure": cmp_tenure,
            "PhoneService": cmp_phone,
            "MultipleLines": cmp_multiple,
            "InternetService": cmp_internet,
            "OnlineSecurity": cmp_security,
            "OnlineBackup": cmp_backup,
            "DeviceProtection": cmp_protection,
            "TechSupport": cmp_tech,
            "StreamingTV": cmp_streaming_tv,
            "StreamingMovies": cmp_streaming_movies,
            "Contract": cmp_contract,
            "PaperlessBilling": cmp_paperless,
            "PaymentMethod": cmp_payment,
            "MonthlyCharges": cmp_monthly,
            "TotalCharges": cmp_total,
        }

        try:
            response = requests.post(f"{API_URL}/predict/compare", json=payload)
            if response.status_code == 200:
                result = response.json()

                # Display comparison
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("🌲 RandomForest")
                    rf = result["random_forest"]
                    st.metric("Prediction", rf["churn_label"])
                    st.metric("Probability", f"{rf['churn_probability']:.2%}")

                with col2:
                    st.subheader("⚡ XGBoost")
                    xgb = result["xgboost"]
                    st.metric("Prediction", xgb["churn_label"])
                    st.metric("Probability", f"{xgb['churn_probability']:.2%}")

                with col3:
                    st.subheader("📊 Comparison")
                    comp = result["comparison"]
                    st.metric("Models Agree", "Yes" if comp["models_agree"] else "No")
                    st.metric("Confidence Diff", f"{comp['confidence_difference']:.4f}")
                    st.metric("Avg Probability", f"{comp['average_probability']:.2%}")

                # Visual comparison
                st.subheader("Probability Comparison")
                comparison_data = {
                    "Model": ["RandomForest", "XGBoost", "Average"],
                    "Probability": [
                        rf["churn_probability"],
                        xgb["churn_probability"],
                        comp["average_probability"],
                    ],
                }
                st.bar_chart(
                    comparison_data,
                    x="Model",
                    y="Probability",
                    use_container_width=True,
                )

                # Agreement indicator
                if comp["models_agree"]:
                    st.success("✅ Both models agree on the prediction!")
                else:
                    st.warning("⚠️ Models disagree. Consider reviewing the prediction.")
            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error comparing models: {e}")

    # Monitoring & Alerts Section
    st.markdown("---")
    st.header("🚨 Monitoring & Alerts")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("High Risk Alert Check")
        threshold = st.slider(
            "Churn Probability Threshold", 0.0, 1.0, 0.5, key="alert_threshold"
        )
        if st.button("🔍 Check Alerts", key="check_alerts"):
            try:
                response = requests.get(
                    f"{API_URL}/alerts/check", params={"threshold": threshold}
                )
                if response.status_code == 200:
                    alert_data = response.json()

                    if alert_data["alert_triggered"]:
                        st.error(
                            f"🚨 ALERT: {alert_data['high_risk_count']} customers ({alert_data['high_risk_percentage']}%) "
                            f"have churn probability >= {threshold}"
                        )

                        st.subheader("Top 10 Highest Risk Customers")
                        risk_df = pd.DataFrame(alert_data["top_risk_customers"])
                        st.dataframe(risk_df, use_container_width=True)
                    else:
                        st.success(
                            f"✅ No alerts. All customers have churn probability < {threshold}"
                        )
                else:
                    st.error("Failed to check alerts")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("Data Drift Monitoring")
        if st.button("📊 Check Data Drift", key="check_drift"):
            try:
                response = requests.get(f"{API_URL}/monitoring/drift")
                if response.status_code == 200:
                    drift_data = response.json()

                    if drift_data["drift_detected"]:
                        st.error("🚨 DATA DRIFT DETECTED!")
                    else:
                        st.success("✅ No significant data drift detected")

                    # Show drift metrics
                    st.markdown("**Drift Metrics:**")
                    for feature, metrics in drift_data["drift_metrics"].items():
                        drift_status = "🚨" if metrics["drift_detected"] else "✅"
                        st.markdown(
                            f"{drift_status} **{feature}**: drift_score={metrics['drift_score']}, "
                            f"current_mean={metrics['current_mean']:.2f}, "
                            f"baseline_mean={metrics['baseline_mean']:.2f}"
                        )
                else:
                    st.error("Failed to check drift")
            except Exception as e:
                st.error(f"Error: {e}")

# Health check in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{API_URL}/health")
        if response.json()["status"] == "healthy":
            st.sidebar.success("API is healthy")
        else:
            st.sidebar.warning("API is not healthy")
    except Exception:
        st.sidebar.error("Cannot connect to API")
