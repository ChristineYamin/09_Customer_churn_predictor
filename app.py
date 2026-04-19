import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# -----------------------------
# Load Data and Model
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_churn_dataset.csv", index_col=0)

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.pkl")

df = load_data()
model = load_model()

# -----------------------------
# Risk Level Function
# -----------------------------
def get_risk_level(p):
    if p < 0.3:
        return "Low Risk"
    elif p < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# -----------------------------
# Title
# -----------------------------
st.title("🛒 E-commerce Customer Churn Prediction")
st.write("Predict customer churn and get actionable business insights.")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer Insights", "Feature Importance", "Predict Churn"]
)

# -----------------------------
# Overview
# -----------------------------
if section == "Overview":
    st.subheader("Overview")

    total = len(df)
    churned = int(df["Churn"].sum())
    active = total - churned
    churn_rate = df["Churn"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", total)
    col2.metric("Churned", churned)
    col3.metric("Active", active)
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    fig, ax = plt.subplots()
    counts = df["Churn"].value_counts().sort_index()
    ax.bar(["Active", "Churned"], counts.values)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

# -----------------------------
# Customer Insights
# -----------------------------
elif section == "Customer Insights":
    st.subheader("Customer Insights")

    feature = st.selectbox(
        "Select Feature",
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
    data_plot = [
        df[df["Churn"] == 0][feature],
        df[df["Churn"] == 1][feature]
    ]

    ax.boxplot(data_plot, labels=["Active", "Churned"])
    ax.set_title(f"{feature} vs Churn")

    st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------
elif section == "Feature Importance":
    st.subheader("Feature Importance")

    features = [
        "Frequency",
        "TotalQuantity",
        "Monetary",
        "AvgOrderValue",
        "PurchaseIntensity",
        "CustomerLifetime"
    ]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)
    st.dataframe(importance_df)

# -----------------------------
# Predict Churn
# -----------------------------
elif section == "Predict Churn":

    st.subheader("Predict Customer Churn")

    frequency = st.number_input("Frequency", 1, 1000, 5)
    total_quantity = st.number_input("Total Quantity", 1, 10000, 20)
    monetary = st.number_input("Monetary", 0.0, 100000.0, 200.0)
    avg_order_value = st.number_input("Avg Order Value", 0.0, 10000.0, 40.0)
    purchase_intensity = st.number_input("Purchase Intensity", 0.0, 10.0, 0.1)
    customer_lifetime = st.number_input("Customer Lifetime", 1, 2000, 100)

    input_df = pd.DataFrame({
        "Frequency": [frequency],
        "TotalQuantity": [total_quantity],
        "Monetary": [monetary],
        "AvgOrderValue": [avg_order_value],
        "PurchaseIntensity": [purchase_intensity],
        "CustomerLifetime": [customer_lifetime]
    })

    st.write("### Input Preview")
    st.dataframe(input_df)

    if st.button("Predict"):

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        risk = get_risk_level(prob)

        st.markdown("---")
        st.subheader("Prediction Result")

        st.write(f"Churn Probability: {prob:.2%}")
        st.write(f"Risk Level: {risk}")

        # 🔥 ACTION LAYER
        if risk == "High Risk":
            st.error("🔴 High Risk Customer")

            st.markdown("""
            **Insight:**
            - Low engagement behavior  
            - Likely to churn soon  

            **Action:**
            - Offer discount  
            - Personal follow-up  
            - Re-engagement campaign  
            """)

        elif risk == "Medium Risk":
            st.warning("🟡 Medium Risk Customer")

            st.markdown("""
            **Insight:**
            - Engagement decreasing  

            **Action:**
            - Send reminders  
            - Offer incentives  
            """)

        else:
            st.success("🟢 Low Risk Customer")

            st.markdown("""
            **Insight:**
            - Active customer  

            **Action:**
            - Upsell products  
            - Loyalty rewards  
            """)