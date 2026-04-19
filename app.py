import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Page Config (Minimalist & Wide)
# -----------------------------
st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #f0f0f0; }
    </style>
    """, unsafe_index=True)

# -----------------------------
# Load Data and Model
# -----------------------------
@st.cache_data
def load_data():
    # Update these to relative paths for easier deployment later!
    return pd.read_csv("data/final_churn_dataset.csv", index_col=0)

@st.cache_resource
def load_model():
    return joblib.load("notebooks/models/random_forest_model.pkl")

df = load_data()
model = load_model()

# -----------------------------
# Logic Functions
# -----------------------------
def get_risk_level(p):
    if p < 0.3: return "Low Risk", "🟢"
    elif p < 0.7: return "Medium Risk", "🟡"
    else: return "High Risk", "🔴"

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation 🧸🎀")
section = st.sidebar.radio("Go to", ["Dashboard Overview", "Risk Predictor & Strategy"])

# -----------------------------
# Section 1: Dashboard Overview
# -----------------------------
if section == "Dashboard Overview":
    st.title("🛒 E-Commerce Health Dashboard")
    st.write("Overview of customer retention and churn patterns.")

    total = len(df)
    churned = int(df["Churn"].sum())
    churn_rate = (churned / total) * 100

    # Layout Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", f"{total:,}")
    m2.metric("Churned", f"{churned:,}")
    m3.metric("Churn Rate", f"{churn_rate:.2f}%", delta="-1.2%" if churn_rate < 20 else "High", delta_color="inverse")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#EAEAEA', '#D4AF37'] # Minimalist Gold/Grey
        df["Churn"].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors, startangle=90)
        ax.set_ylabel("")
        st.pyplot(fig)
    
    with col_b:
        st.subheader("Top Factors Driving Churn")
        importance_df = pd.DataFrame({
            "Feature": ["Frequency", "TotalQuantity", "Monetary", "AvgOrderValue", "PurchaseIntensity", "CustomerLifetime"],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="#D4AF37")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

# -----------------------------
# Section 2: Risk Predictor & Strategy
# -----------------------------
elif section == "Risk Predictor & Strategy":
    st.title("🎯 Customer Risk Assessment")
    st.write("Adjust customer behavior metrics to see real-time churn probability and business recommendations.")

    # Input Area in a Container
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            frequency = st.number_input("Purchase Frequency", 1, 1000, 5)
            avg_order = st.number_input("Avg Order Value ($)", 0.0, 10000.0, 50.0)
        with c2:
            monetary = st.number_input("Total Spending ($)", 0.0, 100000.0, 250.0)
            intensity = st.number_input("Purchase Intensity", 0.0, 10.0, 0.5)
        with c3:
            lifetime = st.number_input("Customer Lifetime (Days)", 1, 3000, 150)
            quantity = st.number_input("Total Quantity", 1, 10000, 30)

    # Data Preparation
    input_data = pd.DataFrame({
        "Frequency": [frequency],
        "TotalQuantity": [quantity],
        "Monetary": [monetary],
        "AvgOrderValue": [avg_order],
        "PurchaseIntensity": [intensity],
        "CustomerLifetime": [lifetime]
    })

    # Prediction Logic (REACTIVE - NO BUTTON)
    prob = model.predict_proba(input_data)[0][1]
    risk_label, icon = get_risk_level(prob)
    revenue_at_risk = prob * monetary

    st.markdown("---")

    # Results Display
    res1, res2, res3 = st.columns(3)
    res1.metric("Churn Probability", f"{prob:.1%}")
    res2.metric("Risk Status", f"{icon} {risk_label}")
    res3.metric("Revenue at Risk", f"${revenue_at_risk:.2f}", help="Probability * Total Spending")

    # Actionable Insights Expansion
    st.subheader("💡 Business Strategy")
    
    if risk_label == "High Risk":
        st.error("Priority 1: Immediate Outreach Required")
        st.write("""
        - **Why:** This customer shows patterns of detachment (low intensity/lifetime).
        - **Action:** Send a high-value 'We Miss You' coupon (20-30% off).
        - **Personalize:** Have a support agent check if there were issues with their last order.
        """)
    elif risk_label == "Medium Risk":
        st.warning("Priority 2: Re-engagement Campaign")
        st.write("""
        - **Why:** Engagement is slipping but they haven't left yet.
        - **Action:** Send personalized product recommendations based on their history.
        - **Incentive:** Offer free shipping on their next order.
        """)
    else:
        st.success("Priority 3: Retention & Growth")
        st.write("""
        - **Why:** Highly active and loyal customer.
        - **Action:** Enroll them in the VIP/Loyalty program.
        - **Strategy:** Focus on Upselling higher-margin products.
        """)