# 📶 Telco Customer Churn Prediction & Live Web Dashboard

An end-to-end Machine Learning pipeline and interactive local web dashboard designed to predict customer churn risk using telecom subscriber data (`WA_Fn-UseC_-Telco-Customer-Churn_2.csv`). 

This project covers everything from dirty data cleaning, EDA, and domain-driven feature engineering to production model training, evaluation, and live real-time inference via a **Streamlit** dashboard.

---

## 🛠️ Project Features

- **Data Cleaning & Imputation:** Handles missing spaces in `TotalCharges`, dynamically computes missing charges via $TotalCharges = Tenure \times MonthlyCharges$, and removes non-predictive identifiers.
- **Feature Engineering:** Calculates `AvgMonthlySpendRatio` and `IsNewCustomer` flags to capture early subscription risk and spending patterns.
- **Production Preprocessing Pipeline:** Uses Scikit-Learn `Pipeline` and `ColumnTransformer` with `StandardScaler` for numeric scaling and `OneHotEncoder` for categorical features.
- **Model Training & Comparison:** Trains and benchmarks tree-based ensemble models (`GradientBoostingClassifier`, `RandomForestClassifier`, and `LogisticRegression`).
- **Live Local Dashboard:** Interactive web UI built with Streamlit allowing users to adjust customer parameters (tenure, contract type, internet service) and view live churn probabilities instantly.

---

## 📂 Project Structure

```text
├── WA_Fn-UseC_-Telco-Customer-Churn_2.csv   # Telco Churn Dataset
├── app.py                                  # Streamlit Live Web Application
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation