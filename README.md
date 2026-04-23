## Project 9
# 🚀 Project #9 / 23 — E-commerce Customer Churn Prediction

## 📌 Overview

Most businesses have customer data, but many do not know **which customers are likely to stop purchasing**.

This project builds a **Customer Churn Prediction System** using real-world e-commerce transaction data to identify at-risk customers based on behavioral patterns and provide proactive retention insights.

Unlike traditional churn datasets, this project uses raw transactional data and defines churn through customer inactivity behavior.

---

## 🎯 Business Problem

Customer churn leads to:

- Lost revenue  
- Lower repeat purchase rates  
- Higher customer acquisition costs  
- Reduced long-term growth  

The goal of this project is to help businesses detect churn risk early and take action before losing customers.

---

## 📊 Dataset

**Online Retail Dataset (UCI Machine Learning Repository)**

This dataset contains real transactional records from a UK-based online retailer.
This dataset is from kaggle.
a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.
https://www.kaggle.com/datasets/jihyeseo/online-retail-data-set-from-uci-ml-repo?utm_source=chatgpt.com

### Key Raw Columns:

- InvoiceNo  
- StockCode  
- Description  
- Quantity  
- InvoiceDate  
- UnitPrice  
- CustomerID  
- Country  

---

## 🧹 Data Preparation

The raw dataset was cleaned by:

- Removing missing CustomerID values  
- Removing cancelled / negative transactions  
- Removing zero or invalid prices  
- Converting InvoiceDate to datetime format  
- Creating transaction revenue (`TotalPrice`)  

---

## 🧠 Feature Engineering

The dataset was transformed from transaction-level to customer-level using behavioral analytics.

### Core Features:

- **Recency** – days since last purchase  
- **Frequency** – number of purchases  
- **Monetary** – total customer spend  

### Additional Features:

- **TotalQuantity** – total items purchased  
- **AvgOrderValue** – average spend per order  
- **PurchaseIntensity** – engagement rate  
- **CustomerLifetime** – days since first purchase  

---

## 🎯 Churn Definition

Since the dataset does not contain a churn label, churn was defined using customer inactivity behavior.

Customers with high inactivity (based on recency threshold) were labeled as churned.

This simulates real-world churn logic used in many businesses.

---

## 🤖 Models Used

### 1. Logistic Regression
Used as an interpretable baseline model.

### 2. Random Forest Classifier
Used to capture nonlinear customer behavior patterns.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|------|----------|----------|--------|----------|
| Logistic Regression | 0.87 | 0.78 | 0.71 | 0.74 |
| Random Forest | 0.99 | 0.99 | 0.98 | 0.99 |

---

## 🔍 Key Insights

- Customer inactivity is a strong indicator of churn risk  
- Low-frequency customers are more likely to churn  
- High-engagement customers are more loyal  
- Behavioral features significantly improve churn prediction performance  

---

## 💻 Streamlit Application

An interactive Streamlit app was developed to:

- View churn metrics  
- Explore customer behavior  
- Visualize feature importance  
- Predict churn risk for new customers  
- Recommend business actions based on risk level  

### Risk Levels:

- 🟢 Low Risk → loyalty / upsell opportunities  
- 🟡 Medium Risk → engagement campaigns  
- 🔴 High Risk → urgent retention action  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## 📂 Project Structure

project_9_churn/

├── app.py  
├── data/  
├── models/  
├── notebooks/  
│   ├── NB1_EDA.ipynb  
│   ├── NB2_Churn_Definition_and_Feature_Engineering.ipynb  
│   └── NB3_Churn_Modeling_and_Evaluation.ipynb  
├── requirements.txt  
└── README.md

---

## 💡 Business Value

This system helps businesses:

- Reduce customer churn  
- Improve retention strategy  
- Increase repeat purchase revenue  
- Focus resources on high-risk customers  

---

## 🚀 Future Improvements

- XGBoost / LightGBM models  
- SHAP explainability  
- Real-time churn scoring API  
- Marketing campaign simulation  
- Customer segmentation + churn combined dashboard  

---

## 👩‍💻 Author

**Shwe Yamin Oo**  
Data Science Graduate | Machine Learning & AI Enthusiast

---

## 🌟 Project Series

This project is part of my **23 Projects at 23** portfolio challenge.

