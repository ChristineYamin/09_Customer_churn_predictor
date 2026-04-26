import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

#-------------------------------
# 1. PAGE CONFIG AND STYLING
#-------------------------------
st.set_page_config(page_title="Customer Intelligence Hub", layout="wide")

#---------------------------
# 2. DATA AND MODEL LOADING
#----------------------------
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    
    # Use relative paths for deployment flexibility
    model_path = os.path.join(base_path, "notebooks/models/random_forest_model.pkl")
    data_path = os.path.join(base_path, "data/final_churn_dataset.csv")
    
    # Fallback if folders are different on your local machine
    if not os.path.exists(model_path): model_path = "random_forest_model.pkl"
    if not os.path.exists(data_path): data_path = "final_churn_dataset.csv"
        
    model = joblib.load(model_path)
    df = pd.read_csv(data_path, index_col=0)
    return model, df

try:
    model, df = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

#---------------------
# HELPER FUNCTIONS
#----------------------------------
def get_risk_analysis(prob):
    if prob < 0.3:
        return "Low Risk", "🟢", "Retention and Upsell", "Keep providing value through loyalty rewards."
    elif prob < 0.7:
        return "Medium Risk", "🟡", "Engagement Campaign", "Send personalized recommendations to bring them back."
    else:
        return "High Risk", "🔴", "Crisis Intervention", "Immediate high-value discount (25%+) required."

#---------------------------------------
# 3. SIDEBAR NAVIGATION
#--------------------------------------
st.sidebar.title("Project #9: CRM Intelligence")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Global Overview", "Search Customer by ID", "Manual Simulator"])

st.sidebar.markdown("---")
st.sidebar.info("**Project 9 of 23**\n\nFocus: ML & Customer Intelligence\n\nBuilt by Shwe Yamin")

#---------------------------------------------
# SECTION 1: GLOBAL OVERVIEW
#--------------------------------------------
if menu == "Global Overview":
    st.title("📊 Business Health Overview")
    
    avg_churn = df["Churn"].mean()
    total_rev_at_risk = (df["Churn"] * df["Monetary"]).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Market Churn Rate", f"{avg_churn: .1%}")
    col3.metric("Current Revenue at Risk", f"${total_rev_at_risk:,.2f}")

    st.markdown("---")
    st.subheader("Key Drivers of Customer Loss")
    
    importances = model.feature_importances_
    features = ["Frequency", "TotalQuantity", "Monetary", "AvgOrderValue", "PurchaseIntensity", "CustomerLifetime"]
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#d4af37")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_alpha(0)
    st.pyplot(fig)

#-------------------------------------------------
# SECTION 2: SEARCH BY CUSTOMER ID
#---------------------------------------------------
#-------------------------------------------------
# UPDATED SECTION 2: SEARCH BY CUSTOMER ID (Dynamic Logic)
#---------------------------------------------------
elif menu == "Search Customer by ID":
    st.title("🔍 Individual Customer Churn")
    search_id = st.text_input("Enter Customer ID:", placeholder="e.g. 17850")

    if search_id:
        try:
            target_id = int(search_id)
            if target_id in df.index:
                cust = df.loc[target_id]
                
                # Global Averages for comparison
                avg_intensity = df['PurchaseIntensity'].mean()
                avg_monetary = df['Monetary'].mean()
                avg_freq = df['Frequency'].mean()
                
                # Prediction
                feature_cols = ["Frequency", "TotalQuantity", "Monetary", "AvgOrderValue", "PurchaseIntensity", "CustomerLifetime"]
                input_df = pd.DataFrame([cust[feature_cols].values], columns=feature_cols)
                prob = model.predict_proba(input_df)[0][1]
                risk_label, icon, strategy, action = get_risk_analysis(prob)
                
                st.markdown(f"### Results for Customer #{target_id} {icon}")
                
                # Metrics Row
                c1, c2, c3 = st.columns(3)
                c1.metric("Churn Probability", f"{prob:.1%}")
                c2.metric("Risk Level", risk_label)
                c3.metric("Total Spend", f"${cust['Monetary']:,.2f}")

                st.markdown("---")
                
                diag_col, action_col = st.columns(2)
                with diag_col:
                    st.subheader(" Why this rating?")
                    
                    # DYNAMIC COMPARISON LOGIC
                    reasons = []
                    
                    # Check Intensity
                    if cust['PurchaseIntensity'] < (avg_intensity * 0.5):
                        reasons.append(f"• **Momentum Loss:** Their purchase intensity ({cust['PurchaseIntensity']:.2f}) is 50% lower than your average customer ({avg_intensity:.2f}).")
                    
                    # Check Frequency
                    if cust['Frequency'] < (avg_freq * 0.3):
                        reasons.append(f"• **Low Habit:** They have only shopped {int(cust['Frequency'])} times, failing to establish a strong loyalty bond.")
                    
                    # Check Lifetime fatigue
                    if prob > 0.5 and cust['CustomerLifetime'] > df['CustomerLifetime'].median():
                        reasons.append(f"• **Legacy Fatigue:** This is a long-term customer (Days: {int(cust['CustomerLifetime'])}) whose activity is suddenly cooling off.")

                    # Check spending
                    if cust['Monetary'] > (avg_monetary * 2) and prob > 0.4:
                        reasons.append("• **High-Value Danger:** This is a VIP spender! Losing them would impact revenue significantly.")

                    if not reasons:
                        st.success("This customer's behavior is currently better than the business average.")
                    else:
                        for r in reasons: st.write(r)
                
                with action_col:
                    st.subheader(" Suggested Next Step")
                    st.info(f"**Action:** {action}")
                    st.write(f"**Business Goal:** {strategy}")
            else:
                st.error(f"Customer ID {target_id} not found.")
        except ValueError:
            st.warning("Please enter a numeric ID.")

#-------------------------------------------------
# SECTION 3: MANUAL SIMULATOR
#---------------------------------------------------
else:
    st.title("🎯 Simulation Predictor")
    st.write("Test 'What-If' scenarios by manually entering values.")

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            freq = st.number_input("Frequency", 1, 500, 10)
            aov = st.number_input("Avg Order Value ($)", 0.0, 5000.0, 100.0)
        with c2:
            monetary = st.number_input("Total Spending ($)", 0.0, 50000.0, 1000.0)
            intensity = st.number_input("Purchase Intensity", 0.0, 5.0, 0.5)
        with c3:
            lifetime = st.number_input("Customer Lifetime (Days)", 1, 2000, 365)
            quantity = st.number_input("Total Quantity", 1, 5000, 50)

    data_row = [freq, quantity, monetary, aov, intensity, lifetime]
    input_df = pd.DataFrame([data_row], columns=["Frequency", "TotalQuantity", "Monetary", 
                                                 "AvgOrderValue", "PurchaseIntensity", "CustomerLifetime"])

    prob = model.predict_proba(input_df)[0][1]
    risk_label, icon, strategy, action = get_risk_analysis(prob)

    st.markdown("---")
    st.subheader(f"Simulation Result: {icon} {risk_label}")
    st.metric("Probability Score", f"{prob:.1%}")
    st.progress(prob)
    st.info(f"**Strategy Recommendation:** {action}")