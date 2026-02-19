import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("rf_model.pkl")

st.set_page_config(page_title="Predictive Order Risk Brain", layout="centered")

st.title("📦 Predictive Order Risk Brain")
st.markdown("AI-powered Delivery Risk Prediction & Simulation")

st.divider()

# ----------------------------
# Input Section
# ----------------------------
st.subheader("Enter Order Details")

quantity = st.number_input("Quantity", min_value=0, value=1000)
ship_quantity = st.number_input("Ship Quantity", min_value=0, value=0)
order_value = st.number_input("Order Value", min_value=0.0, value=10000.0)
days_left = st.number_input("Days Left Until Delivery", value=10)
order_age = st.number_input("Order Age (Days)", min_value=0, value=30)
pr_delay = st.number_input("PR Delay Days", min_value=0, value=5)
plant_load = st.number_input("Plant Load (%)", min_value=0.0, value=90.0)
not_shipped = st.selectbox("Not Shipped?", [0, 1])

# ----------------------------
# Risk Classification Function
# ----------------------------
def classify_risk(prob):
    if prob < 0.3:
        return "🟢 Green (Low Risk)"
    elif prob < 0.8:
        return "🟡 Yellow (Medium Risk)"
    else:
        return "🔴 Red (High Risk)"

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict Risk"):

    new_order = pd.DataFrame([{
        "Quantity": quantity,
        "Ship Quantity": ship_quantity,
        "Order_Value": order_value,
        "Days_Left": days_left,
        "Order_Age": order_age,
        "PR_Delay_Days": pr_delay,
        "Plant_Load_%": plant_load,
        "Not_Shipped_Flag": not_shipped
    }])

    prob = model.predict_proba(new_order)[0][1]
    risk = classify_risk(prob)

    st.success("Prediction Complete")

    st.write("### 📊 Delay Probability:", round(prob, 2))
    st.write("### 🚦 Risk Level:", risk)

    # ----------------------------
    # Simple Recommendation Logic
    # ----------------------------
    st.divider()
    st.subheader("Recommended Action")

    if prob > 0.8:
        st.error("Immediate intervention required: Approve PR and reduce plant load.")
    elif prob > 0.4:
        st.warning("Monitor closely: Review PR status and plant capacity.")
    else:
        st.info("Low risk: No immediate action required.")

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption("Built using Random Forest • AI Signal Layer • POC Version")
