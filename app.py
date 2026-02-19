import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Predictive Order Risk Brain")

# -----------------------------
# Load Dataset and Train Model
# -----------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("fixed_dataset.csv")

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

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X, y)
    return model

model = train_model()
