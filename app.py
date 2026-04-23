import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

#-------------------------------
# 1. PAGE CONFIG AND STYLING
#-------------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

#---------------------------
# 2. DATA AND MODEL LOADING
#----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("C:/Projects/09_Customer_churn_predictor/notebooks/models/random_forest_model.pkl")
    df = pd.read_csv("C:/Projects/09_Customer_churn_predictor/data/final_churn_dataset.csv", index_col=0)
    return model, df

try:
    model,df = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

#---------------------------------------
# 3. SIDEBAR NAVIGATION
#--------------------------------------
st.sidebar.title("Project #9: Customer Churn")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Global Overview", "Individual Risk Predictor"])

#---------------------
# FUNCTION: GET RISK DATA
#----------------------------------
def get_risk_analysis(prob):
    if prob < 0.3:
        return "Low Risk", "🟢", "Retention and Upsell", "Keep providing value through loyalty rewards."
    elif prob < 0.7:
        return "Medium Risk", "🟡", "Engagement Campaign", "Send personalized recommendations to bring them back."
    else:
        return "High Risk", "🔴", "Crisis Intervention", "Immediate high-value discount (25%+) required."
    
#---------------------------------------------
# SECTION: GLOBAL OVERVIEW
#--------------------------------------------
if menu == "Global Overview":
    st.title("📊 Business Health Overview")

    avg_churn = df["Churn"].mean()
    total_rev_at_risk = (df["Churn"] * df["Monetary"]).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Market Churn Rate", f"{avg_churn: .1%}")
    col3.metric("Curreny Lost Revenue", f"{total_rev_at_risk:,.2f}")

    st.markdown("---")
    st.subheader("What Drives Churn Globally?")

    # Feature Importance Insight
    importances = model.feature_importances_
    features = ["Frequency", "TotalQuantity", "Monetary", "AvgOrderValue", "PurchaseIntensity", "CustomerLifetime"]
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#d4af37")
    ax.set_title("Key Drivers if Customer Loss")
    st.pyplot(fig)

#-------------------------------------------------
# SECTION: INDIVIDUAL PREDICTOR (The 4 Insights)
#---------------------------------------------------
else:
    st.title("🎯 Individual Customer Intelligence")
    st.write("Calculate exactly who is leaving and why.")

    # Input Section
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            freq = st.number_input("Frequency", 1,500,10)
            aov = st.number_input("Avg Order Value ($)", 0.0, 5000.0, 100.0)
        with c2:
            monetary = st.number_input("Total Spending ($)", 0.0, 50000.0, 1000.0)
            intensity = st.number_input("Purchase Intensity", 0.0, 5.0, 0.5)
        with c3:
            lifetime = st.number_input("Customer Lifetime (Days)", 1, 2000, 365)
            quantity = st.number_input("Total Quantity", 1,5000, 50)
    
    # Prepare Data for model

    data_row = [freq, quantity, monetary, aov, intensity, lifetime]

    input_df = pd.DataFrame([data_row], columns=["Frequency", "TotalQuantity", "Monetary", 
                                 "AvgOrderValue", "PurchaseIntensity", "CustomerLifetime"])

    # Calculations
    prob = model.predict_proba(input_df)[0][1]
    risk_label, icon, strategy, action = get_risk_analysis(prob)
    rev_at_risk = prob * monetary

    st.markdown("---")

    # The four insights display
    st.subheader("Prediction Results")
    res1, res2, res3 = st.columns(3)

    #Insight 1: Probability Score
    res1.metric("Churn Probability", f"{prob:.1%}")

    # Insight 2: Risk Label
    res2.metric("Risk Level", f"{icon} {risk_label}")

    # Insight 3: Revenue at Risk
    res3.metric("Potential Loss", f"{rev_at_risk:.2f}", delta_color="inverse")

    st.markdown("---")

    # Insight 4: Prescriptive Action and "Why"
    st.subheader("Prescriptive Strategy")

    col_left, col_right = st.columns([1,2])

    with col_left:
        st.info(f"**Strategy:** {strategy}")

    with col_right:
        st.write(f"**Targeted Action:** {action}")

        # simple "Why" logic (Heuristic based on inputs)
        if intensity < 0.2:
            st.write("- Critical drop in purchase intensity detected.")
        if lifetime > 500 and prob > 0.5:
            st.write("- Long-term customer showing signs of 'fatigue'.")
        if freq < 3:
            st.write("- Customer has not yet established a habitual buying pattern.")

#---------------------------------------
# SECTION : THE "HUMAN-FRIENDLY" REPORT 
#---------------------------------------------
st.markdown("---")
st.header("📋 Customer Diagnostic Report")

# Top Row: The Big Picture
col_a, col_b = st.columns(2)

with col_a:
    # Insight 1: Visual Status
    if prob > 0.7:
        st.error(f"### 🔴 Status: High Risk of Leaving")
    elif prob > 0.3:
        st.warning(f"### 🟡 Status: Showing Signs of Fading")
    else:
        st.success(f"### 🟢 Status: Healthy & Active")

with col_b:
    # Insight 2: Money Impact
    st.metric("Estimated Revenue at Risk", f"${rev_at_risk:,.2f}", 
              help="The potential financial loss if this customer stops buying.")

# Bottom Row: The "Why" and "What Next"
diag_col, action_col = st.columns(2)

with diag_col:
    st.subheader("🧐 Why this rating?")
    if intensity < 0.3:
        st.write("• **Activity Drop:** They are shopping much slower than usual.")
    if freq < 3:
        st.write("• **New User:** They haven't established a strong loyalty pattern yet.")
    if prob < 0.3:
        st.write("• **Consistent:** Their shopping habits are very stable.")

with action_col:
    st.subheader("🛠️ Suggested Next Step")
    # This matches the Prescriptive Action insight
    st.info(f"**Recommendation:** {action}")
