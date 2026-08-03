```markdown
# 📶 Telco Customer Churn Prediction & Live Web Dashboard

An end-to-end Machine Learning pipeline and interactive local web application designed to predict customer churn risk using telecom subscriber data (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). 

This project covers the full data science lifecycle: dirty data cleaning, exploratory data analysis (EDA), domain-driven feature engineering, model training/evaluation, and live real-time inference via an interactive **Streamlit** dashboard.

---

## 🛠️ Key Features

- **Data Cleaning & Imputation:** Automatically handles missing values in `TotalCharges` using dynamic calculation ($TotalCharges = Tenure \times MonthlyCharges$) and removes uninformative identifiers.
- **Feature Engineering:** Computes custom features such as `AvgMonthlySpendRatio` and `IsNewCustomer` flags to capture subscriber risk patterns.
- **Production Preprocessing Pipeline:** Built with Scikit-Learn `Pipeline` and `ColumnTransformer` featuring `StandardScaler` for numeric scaling and `OneHotEncoder` for categorical features.
- **Model Training & Comparison:** Trains and benchmarks tree-based ensemble models (`GradientBoostingClassifier`, `RandomForestClassifier`, and `LogisticRegression`).
- **Live Interactive Web App:** Built with **Streamlit** to allow users to adjust customer inputs via sliders and dropdowns to view live churn predictions instantly.

---

## 📂 Repository Structure

```text
├── WA_Fn-UseC_-Telco-Customer-Churn_2.csv   # Telco Churn Dataset
├── app.py                                  # Live Interactive Streamlit Web Dashboard
├── requirements.txt                        # Required Python packages
└── README.md                               # Project documentation

```

---

## 🚀 How to Clone & Run This Project Locally

Follow these step-by-step instructions to clone the repository and run the project on your local machine.

### Prerequisites

Make sure you have **Python 3.8+** and **Git** installed on your system.

---

### Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone [[https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/NajeebAhmed69/MiniProject_END_TO_END.git)](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd REPOSITORY_NAME

```

---

### Step 2: Set Up a Virtual Environment (Recommended)

Creating a virtual environment ensures dependencies remain isolated.

* **On Windows (PowerShell / CMD):**
```bash
python -m venv venv
.\venv\Scripts\activate

```


* **On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate

```



---

### Step 3: Install Required Dependencies

Install all required libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt

```

---

### Step 4: Run the Application

You can run the project in two different ways depending on your use case:

#### Option A: Run the Machine Learning Pipeline (Terminal Script)

To train the model, view evaluation metrics, and run sample inference directly in your command line:

```bash
python Project-END-TO-END.py

```

#### Option B: Launch the Live Web App (Streamlit Dashboard)

To open the interactive web interface where you can test live predictions using sliders and inputs:

```bash
streamlit run app.py

```

Once executed, Streamlit will automatically launch the dashboard in your web browser at:
👉 **`http://localhost:8501`**

---

## 💻 How to Use the Live Web Dashboard

1. Adjust the sliders and dropdown menus on the **left sidebar** (e.g., set `Tenure`, change `Contract Type` to *Month-to-month*, or select `Internet Service`).
2. The main screen will display the **real-time Churn Probability percentage** and a **Risk Assessment Banner** (`HIGH CHURN RISK` vs `RETAINED CUSTOMER`) instantly.

---

## 📊 Dataset Overview

The dataset contains **7,043 customer records** with **21 attributes**:

* **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Subscribed Services:** `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* **Account & Billing:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Target Variable:** `Churn` (`1` = High Churn Risk, `0` = Retained)

---

## 📄 License

Distributed under the MIT License. Feel free to use and modify this project for learning or portfolio purposes.

```

```