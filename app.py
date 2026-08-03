import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Telco Churn Predictor",  layout="centered")

st.title(" Live Telco Customer Churn Dashboard")
st.write("Adjust customer attributes on the left panel to test live churn probabilities.")

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    
    missing_mask = df['TotalCharges'].isnull()
    df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'tenure'] * df.loc[missing_mask, 'MonthlyCharges']
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    df['AvgMonthlySpendRatio'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1e-5)
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

model = load_and_train_model()
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=4)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0)
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

total_charges = tenure * monthly_charges
avg_spend_ratio = 1.0
is_new = 1 if tenure <= 6 else 0

input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': internet,
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': tech_support,
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'AvgMonthlySpendRatio': avg_spend_ratio,
    'IsNewCustomer': is_new
}])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

st.subheader("Live Prediction Output")

col1, col2 = st.columns(2)
with col1:
    st.metric("Churn Probability", f"{probability:.1%}")

with col2:
    if prediction == 1:
        st.error("⚠️ HIGH CHURN RISK")
    else:
        st.success("✅ RETAINED CUSTOMER")

st.progress(float(probability))