# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Project Overview

This project focuses on analyzing and predicting customer churn for a European bank using exploratory data analysis, machine learning, data visualization, and interactive predictive analytics.

The project was completed as part of a **Data Analyst Internship/Training Program at Unified Mentor**.

The objective is to identify important factors associated with customer churn, develop machine learning models to predict churn risk, and provide interactive tools that can help understand and evaluate customer churn behavior.

---

## Objectives

The main objectives of this project are:

- Analyze customer data to identify churn patterns.
- Perform exploratory data analysis (EDA).
- Identify important customer characteristics associated with churn.
- Engineer additional features for predictive modeling.
- Train and compare multiple machine learning classification models.
- Select the best-performing model based on evaluation metrics.
- Interpret model predictions using feature importance and SHAP analysis.
- Develop an interactive Streamlit application for customer churn prediction.
- Build a What-If Scenario Simulator to evaluate changes in customer churn risk.
- Create an interactive Tableau dashboard for business-level analysis.

---

## Dataset

The project uses a European bank customer dataset containing:

- **10,000 customer records**
- **14 variables**

### Main Features

| Feature | Description |
|---|---|
| Year | Year associated with the customer record |
| CustomerId | Unique customer identifier |
| Surname | Customer surname |
| CreditScore | Customer credit score |
| Geography | Customer's country |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Customer account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Estimated customer salary |
| Exited | Customer churn indicator |

`Exited` is the target variable:

- `0` = Customer did not churn
- `1` = Customer churned

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Checked the dataset for missing values.
2. Checked for duplicate records.
3. Removed `CustomerId` and `Surname` from the modeling features.
4. Separated the target variable `Exited` from the input features.
5. Applied one-hot encoding to categorical variables:
   - Geography
   - Gender
6. Performed an 80/20 stratified train-test split.
7. Used `random_state = 42`.
8. Applied StandardScaler to the numerical features.

---

## Exploratory Data Analysis

Exploratory analysis was performed to understand customer churn patterns across:

- Geography
- Age groups
- Gender
- Active membership
- Account balance
- Number of products

The analysis showed that churn is strongly associated with customer age, product usage, activity level, geography, and balance.

---

## Feature Engineering

Additional features were created to provide the models with meaningful behavioral and relationship-related information.

The engineered features include:

- `BalanceSalaryRatio`
- `ProductDensity`
- `EngagementProduct`
- `AgeTenure`

These features were designed to represent relationships between customer financial characteristics, product utilization, engagement, age, and tenure.

---

## Machine Learning Models

Five classification models were developed and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.65% | 57.35% | 19.16% | 28.73% | 77.34% |
| Decision Tree | 85.80% | 80.30% | 40.05% | 53.44% | 84.52% |
| Random Forest | 86.90% | 82.22% | 45.45% | 58.54% | 86.45% |
| Gradient Boosting | **86.90%** | 78.66% | **48.89%** | **60.30%** | **86.82%** |
| XGBoost | 86.85% | 79.75% | 47.42% | 59.48% | 86.79% |

### Selected Model

**Gradient Boosting** was selected as the final model because it achieved:

- Accuracy: **86.90%**
- Recall: **48.89%**
- F1-Score: **60.30%**
- ROC-AUC: **86.82%**

The primary prediction threshold used in the application is **0.50**.

---

## Feature Importance

The Gradient Boosting model identified the following features as the strongest contributors to its predictions:

| Feature | Importance |
|---|---:|
| Age | 0.388299 |
| NumOfProducts | 0.299886 |
| IsActiveMember | 0.113911 |
| Balance | 0.089054 |
| Geography_Germany | 0.055604 |
| CreditScore | 0.018647 |
| EstimatedSalary | 0.016514 |
| Gender_Male | 0.013169 |
| Tenure | 0.003754 |
| HasCrCard | 0.000665 |

The results indicate that **Age** and **Number of Products** are the most influential features in the Gradient Boosting model.

---

## Model Interpretability

Model interpretation was performed using:

- Gradient Boosting feature importance
- SHAP analysis
- Partial Dependence analysis

These techniques were used to better understand how customer characteristics contribute to churn predictions.

---

## Streamlit Application

An interactive Streamlit application was developed to provide practical access to the trained model.

### Main Application Modules

#### 1. Customer Risk Prediction

Users can enter customer characteristics and obtain:

- Predicted churn probability
- Customer churn risk classification

#### 2. Model Insights

The application provides:

- Churn probability distribution
- Model prediction threshold
- Feature importance visualization

#### 3. What-If Scenario Simulator

Users can modify customer characteristics and compare:

- Original churn probability
- What-If churn probability
- Change in predicted churn risk

This allows users to explore how changes in customer characteristics may affect the model's predicted churn probability.

#### 4. Tableau Dashboard

The Streamlit application also integrates the interactive Tableau dashboard for business-level customer churn analysis.

---

## Tableau Dashboard

The Tableau dashboard provides interactive visual analysis of customer churn across:

- Geography
- Age Group
- Gender
- Active Member Status
- Balance
- Number of Products

The dashboard includes KPI cards for:

- Total Customers
- Churned Customers
- Churn Rate
- Retention Rate

The dashboard also includes interactive filters for exploring different customer segments.

---

## Key Business Insights

The analysis identified the following key findings:

- Customers aged **43–62** show the highest churn levels, with the **53–57** age group reaching **58.96%** churn.
- **Germany** has the highest country-level churn rate at **32.44%**.
- **Non-active members** have higher churn (**26.85%**) than active members (**14.27%**).
- Customers with **200K–250K balances** have a churn rate of **55.88%**.
- Customers with **one product** have higher churn (**27.71%**) than those with **two products (7.58%)**.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Tableau
- Matplotlib
- Joblib

---

## Project Structure

```text
Predictive Modeling and Risk Scoring for Bank Customer Churn/
│
├── app.py
├── README.md
├── European_Bank.csv
├── dashboard_background.jpg
│
├── gb_model.pkl
├── scaler.pkl
├── test_probabilities.pkl
├── feature_importance.pkl
│
├── Research_Paper.docx
└── Executive_Summary.docx


##How to Run the Application
1. Install the required libraries
pip install pandas numpy scikit-learn xgboost shap streamlit matplotlib joblib

2. Navigate to the project directory
cd "Predictive Modeling and Risk Scoring for Bank Customer Churn"

3. Run the Streamlit application
streamlit run app.py

4. Open the application
Streamlit will provide a local URL, typically:
http://localhost:8501
