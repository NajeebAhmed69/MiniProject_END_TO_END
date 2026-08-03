import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# =============================================================
# 1. LOAD & CLEAN DATASET
# =============================================================
# Reads dataset file
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop non-predictive customer identifier
df = df.drop(columns=['customerID'])

# Convert TotalCharges from string/object to float and handle missing spaces
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')

# Fill missing TotalCharges values using Tenure * MonthlyCharges
missing_mask = df['TotalCharges'].isnull()
df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'tenure'] * df.loc[missing_mask, 'MonthlyCharges']

# Encode binary target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# =============================================================
# 2. FEATURE ENGINEERING
# =============================================================
# Create spend ratio feature
df['AvgMonthlySpendRatio'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1e-5)
df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)

X = df.drop(columns=['Churn'])
y = df['Churn']

# Define numeric and categorical feature columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# =============================================================
# 3. PREPROCESSING PIPELINE & MODEL TRAINING
# =============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# 80/20 Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build Production Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
])

# Train model
model_pipeline.fit(X_train, y_train)

# =============================================================
# 4. EVALUATION
# =============================================================
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("=" * 50)
print(" MODEL EVALUATION RESULTS ")
print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}\n")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# =============================================================
# 5. PREDICT ON NEW UNSEEN CUSTOMER
# =============================================================
new_customer_data = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.50,
    'TotalCharges': 171.00,
    'AvgMonthlySpendRatio': 1.0,
    'IsNewCustomer': 1
}])

pred_class = model_pipeline.predict(new_customer_data)[0]
pred_prob = model_pipeline.predict_proba(new_customer_data)[0][1]

print("=" * 50)
print(" LIVE PREDICTION FOR NEW CUSTOMER ")
print("=" * 50)
print(f"Prediction       : {'HIGH CHURN RISK (1)' if pred_class == 1 else 'LOW CHURN RISK (0)'}")
print(f"Churn Probability: {pred_prob:.2%}")