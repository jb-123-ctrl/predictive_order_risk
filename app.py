import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predictive Order Risk Brain")
st.title("📦 Predictive Order Risk Brain")
st.markdown("AI-powered Delivery Risk Prediction")

# -----------------------------------------
# Train Model
# -----------------------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("fixed_dataset.csv")
    df.columns = df.columns.str.strip()

    # Rename safe
    df = df.rename(columns={
        "Order Value": "Order_Value",
        "Plant_Load %": "Plant_Load_%"
    })

    # -----------------------------
    # 🔥 FIXED BUSINESS LABEL LOGIC
    # -----------------------------
    df["Delayed"] = np.where(
        (df["PR_Delay_Days"] > 15) |
        (df["Plant_Load_%"] > 110) |
        ((df["Not_Shipped_Flag"] == 1) & (df["Days_Left"] < 0)),
        1,
        0
    )

    features = [
        "Quantity",
        "Ship Quantity",
        "Order_Value",
        "Days_Left",
        "Order_Age",
        "PR_Delay_Days",
        "Plant_Load_%",
        "Not_Shipped_Flag"
    ]

    X = df[features]
    y = df["Delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


model = train_model()

st.divider()
st.subheader("Enter Order Details")

# -----------------------------------------
# User Inputs
# -----------------------------------------
quantity = st.number_input("Quantity", min_value=0, value=1000)
ship_quantity = st.number_input("Ship Quantity", min_value=0, value=0)
order_value = st.number_input("Order Value", min_value=0.0, value=10000.0)
days_left = st.number_input("Days Left Until Delivery", value=10)
order_age = st.number_input("Order Age (Days)", min_value=0, value=30)
pr_delay = st.number_input("PR Delay Days", min_value=0, value=5)
plant_load = st.number_input("Plant Load (%)", min_value=0.0, value=90.0)
not_shipped = st.selectbox("Not Shipped?", [0, 1])

# -----------------------------------------
# Risk Classification
# -----------------------------------------
def classify_risk(prob):
    if prob < 0.3:
        return "🟢 Green (Low Risk)"
    elif prob < 0.7:
        return "🟡 Yellow (Medium Risk)"
    else:
        return "🔴 Red (High Risk)"


# -----------------------------------------
# Prediction
# -----------------------------------------
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

    proba = model.predict_proba(new_order)[0]

    if 1 in model.classes_:
        class_index = list(model.classes_).index(1)
        prob = proba[class_index]
    else:
        prob = 0.0

    prob = float(np.clip(prob, 0.01, 0.99))

    risk = classify_risk(prob)

    st.success("Prediction Complete")

    st.write("### 📊 Delay Probability:", round(prob, 3))
    st.write("### 🚦 Risk Level:", risk)

    st.divider()
    st.subheader("Recommended Action")

    if prob > 0.7:
        st.error("Immediate intervention required: Approve PR and reduce plant load.")
    elif prob > 0.4:
        st.warning("Monitor closely: Review PR status and plant capacity.")
    else:
        st.info("Low risk: No immediate action required.")

st.divider()
st.caption("AI Signal Layer • Business-Aligned ML Version")

