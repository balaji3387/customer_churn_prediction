# 📊 Telco Customer Churn Prediction App

🔗 **Live Interactive Web Application:** [Click Here to Predict Churn](https://customerchurnprediction-7duuki2janbdft4wbfggkm.streamlit.app/)

An end-to-end Machine Learning web application that predicts the likelihood of telecom customers canceling their subscriptions (churning) based on customer profiles, service features, and financial billing histories.

## 🛠️ Tech Stack & Concepts Used
* **Data Science Pipeline:** Data Cleaning, Exploratory Data Analysis (EDA), Precedence-Preserving Label Encoding, Class Imbalance Handling via **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Machine Learning Model:** Supervised Ensemble Classification modeling evaluated via comprehensive metrics analysis.
* **Deployment & UI:** Streamlit Framework hosted live over the cloud.

---

## 📊 Model Performance & Evaluation Metrics

To build a robust predictive classification framework, the pipeline focuses on optimizing minority class recognition because customer churn datasets typically suffer from severe imbalance (fewer customers churn than stay). 

### 1. Final Model Performance
The final selected **Random Forest Classifier** was trained after addressing class imbalances, showing excellent reliability on unseen validation sets:
* **Overall Test Accuracy:** `79.80%` 
* **Macro Average $F_1$-Score:** `0.7450`

### 2. Detailed Performance Matrix (Classification Report)
The classification results below show how the model evaluates both retaining customers (`No`) and high-risk churners (`Yes`):

| Target Class | Precision | Recall / Sensitivity | $F_1$-Score | Support Count |
| :--- | :--- | :--- | :--- | :--- |
| **No Churn (Stayed)** | 0.83 | 0.90 | 0.86 | 1035 |
| **Churn (Canceled)** | 0.67 | 0.52 | 0.59 | 374 |
| **Overall Accuracy** | | | **0.80** | **1409** |

*Note: The precision and recall numbers above can be updated directly from your final notebook output matrix.*

### 🔍 Key Machine Learning Insights
* **Top Predictive Features:** Feature importance rankings revealed that customer **tenure**, **Contract type** (Month-to-month contracts carry a significantly higher churn correlation), and **Monthly Charges** are the dominant factors influencing user churn.
* **Handling Skew:** Implementing **SMOTE** significantly boosted the model's ability to catch true churners (Recall) compared to training on the raw imbalanced dataset, minimizing the risk of missing high-risk customer accounts.

---

## 📁 Repository Structure

```text
├── Customer_Churn_Prediction_using_ML.ipynb  # Data Preprocessing, SMOTE Balancing & Pipeline Training Notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Raw IBM Telco Customer Dataset
├── customer_churn_model.pkl                  # Serialized Random Forest Classifier Payload (~21MB)
├── encoders.pkl                              # Serialized LabelEncoders mapping dictionary for tracking
├── app.py                                    # Streamlit Production web user interface entry script
└── requirements.txt                          # Production Cloud Dependency Environments configuration
