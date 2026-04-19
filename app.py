import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# -----------------------------
# Load Data and Model
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Projects/09_Customer_churn_predictor/data/final_churn_dataset.csv", index_col=0)
    return df

@st.cache_resource
def load_model():
    model = joblib.load("C:/Projects/09_Customer_churn_predictor/notebooks/models/random_forest_model.pkl")
    return model

df = load_data()
model = load_model()

# -----------------------------
# Title and Intro
# -----------------------------
st.title("🛒 E-commerce Customer Churn Prediction")
st.write(
    """
    This application analyzes customer behavior and predicts churn risk using a machine learning model.
    Churn in this project is defined based on customer inactivity patterns derived from transaction data.
    """
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer Insights", "Feature Importance", "Predict Churn"]
)

# -----------------------------
# Overview Section
# -----------------------------
if section == "Overview":
    st.subheader("Project Overview")

    total_customers = len(df)
    churned_customers = int(df["Churn"].sum())
    active_customers = total_customers - churned_customers
    churn_rate = df["Churn"].mean() * 100
    avg_monetary = df["Monetary"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Active Customers", f"{active_customers:,}")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        st.metric("Average Monetary Value", f"{avg_monetary:.2f}")

    with col6:
        avg_frequency = df["Frequency"].mean()
        st.metric("Average Frequency", f"{avg_frequency:.2f}")

    st.markdown("---")
    st.subheader("Churn Distribution")

    churn_counts = df["Churn"].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(["Active (0)", "Churned (1)"], churn_counts.values)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.write(
        """
        This chart shows the class balance between active and churned customers in the dataset.
        """
    )

# -----------------------------
# Customer Insights Section
# -----------------------------
elif section == "Customer Insights":
    st.subheader("Customer Behavior Insights")

    feature = st.selectbox(
        "Select a feature to compare against churn",
        [
            "Frequency",
            "TotalQuantity",
            "Monetary",
            "AvgOrderValue",
            "PurchaseIntensity",
            "CustomerLifetime"
        ]
    )

    fig, ax = plt.subplots()
    data_to_plot = [
        df[df["Churn"] == 0][feature].dropna(),
        df[df["Churn"] == 1][feature].dropna()
    ]
    ax.boxplot(data_to_plot, labels=["Active (0)", "Churned (1)"])
    ax.set_title(f"{feature} vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel(feature)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Summary Statistics by Churn Group")

    summary_df = df.groupby("Churn")[
        [
            "Frequency",
            "TotalQuantity",
            "Monetary",
            "AvgOrderValue",
            "PurchaseIntensity",
            "CustomerLifetime"
        ]
    ].mean()

    st.dataframe(summary_df)

# -----------------------------
# Feature Importance Section
# -----------------------------
elif section == "Feature Importance":
    st.subheader("Feature Importance")

    feature_names = [
        "Frequency",
        "TotalQuantity",
        "Monetary",
        "AvgOrderValue",
        "PurchaseIntensity",
        "CustomerLifetime"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.dataframe(importance_df)

    st.write(
        """
        Feature importance shows which behavioral variables contribute most to churn prediction.
        Higher importance means the model relied more on that feature.
        """
    )

# -----------------------------
# Prediction Section
# -----------------------------
elif section == "Predict Churn":
    st.subheader("Predict Customer Churn")

    st.write("Enter customer behavior details below:")

    frequency = st.number_input("Frequency", min_value=1, value=5)
    total_quantity = st.number_input("Total Quantity", min_value=1, value=20)
    monetary = st.number_input("Monetary", min_value=0.0, value=200.0)
    avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=40.0)
    purchase_intensity = st.number_input("Purchase Intensity", min_value=0.0, value=0.10, format="%.4f")
    customer_lifetime = st.number_input("Customer Lifetime", min_value=1, value=100)

    input_df = pd.DataFrame({
        "Frequency": [frequency],
        "TotalQuantity": [total_quantity],
        "Monetary": [monetary],
        "AvgOrderValue": [avg_order_value],
        "PurchaseIntensity": [purchase_intensity],
        "CustomerLifetime": [customer_lifetime]
    })

    st.markdown("---")
    st.write("### Input Preview")
    st.dataframe(input_df)

    if st.button("Predict Churn"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Churn Risk — Probability: {probability:.2%}")
        else:
            st.success(f"✅ Low Churn Risk — Probability: {probability:.2%}")