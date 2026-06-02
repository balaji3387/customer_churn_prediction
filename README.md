# customer_churn_prediction
# 📊 Telco Customer Churn Prediction App

A full-stack machine learning web application built to predict the likelihood of telecom customers canceling their subscriptions (churning). This end-to-end project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and a production-ready user interface deployed to the cloud.

🔗 **Live Production Link:** [View Live Streamlit Web App](https://customerchurnprediction-7duuki2janbdft4wbfggkm.streamlit.app/)

---

## 🚀 Project Overview

Customer churn is one of the most critical metrics for telecom companies. Retaining existing customers is significantly less expensive than acquiring new ones. This project leverages historical user data to build an intelligent predictive system capable of flagging high-risk churn profiles instantly.

### Key Features of the App:
* **Interactive UI:** Input individual customer demographic details, subscribed services, and financial contract types.
* **Instant Risk Assessments:** Uses an optimized machine learning model to compute real-time churn percentages.
* **Precedence Preservation:** Seamlessly applies pre-trained `LabelEncoders` on user input before scoring pipelines run.

---

## 🛠️ Tech Stack & Frameworks

* **User Interface & Deployment:** Streamlit Community Cloud
* **Machine Learning Framework:** Scikit-Learn (`v1.6.1`)
* **Data Engineering:** Pandas, NumPy
* **Model Serialization:** Pickle (Python Object Serialization)
* **Development Environment:** Jupyter Notebook / Google Colab

---

## 📁 Repository Structure

```text
├── Customer_Churn_Prediction_using_ML.ipynb  # Comprehensive Data Analysis, Training & Pipeline Notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Raw IBM Telco Customer Churn Dataset
├── customer_churn_model.pkl                  # Serialized Random Forest Classifier (~21MB)
├── encoders.pkl                              # Serialized LabelEncoders dictionary for categorical columns
├── app.py                                    # Streamlit Production web UI entry point
└── requirements.txt                          # Precise Environment & Library Dependencies configuration
