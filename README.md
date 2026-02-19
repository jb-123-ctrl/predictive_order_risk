# 📦 Predictive Order Risk Brain

An AI-powered decision support system that predicts delivery risk in the order fulfillment lifecycle using Machine Learning.

This project builds a Random Forest model to monitor and simulate delivery risks across 3 critical KPIs.

---

## 🚀 Project Overview

The goal is to shift from:

Manual, static review of order pendency  
➡️ To  
Proactive, explainable AI-driven early-warning signals.

This system predicts whether an order is at risk of delay and classifies it into:

- 🟢 Green – Low Risk  
- 🟡 Yellow – Medium Risk  
- 🔴 Red – High Risk  

---

## 🧠 3 KPI Coverage

### ✅ KPI 1 – Order Fulfillment Rate (OTIF Signal)

- Detect delayed orders  
- Calculate fulfillment rate  
- Classify orders by risk level  
- Rank highest-risk orders  

---

### ✅ KPI 2 – 90-Day Forward Delivery Risk

- Predict delay probability  
- Classify risk based on model confidence  
- Simulate future scenarios (30/60/90 days)  
- Generate plant-level risk heatmap  

---

### 🟡 KPI 3 – Upstream PR Approval Risk (POC Level)

- Model procurement delay impact  
- Simulate PR approval intervention  
- Link PR delay to fulfillment risk  

(Note: PR data is simulated for POC demonstration.)

---


---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## 📊 Features

✔ Delay Probability Prediction  
✔ Risk Classification Engine  
✔ Feature Importance (Explainability)  
✔ Top Risk Order Ranking  
✔ Plant Risk Heatmap  
✔ Scenario Simulation Mode  
✔ Streamlit Web Deployment  

---

---

## 🌍 Deployment

The application is deployed using:

- GitHub Repository
- Streamlit Cloud

Users can interact with the AI model via a web interface.

---

## 📈 Model Details

- Algorithm: Random Forest Classifier
- Balanced class weighting
- Probability-based risk scoring
- Simulation-driven forward risk analysis

---

## 🔮 Future Enhancements

- Integration with real ERP PR data
- Supplier performance modeling
- SHAP explainability
- Real-time database integration
- Advanced decision recommendation engine

---

## 👩‍💻 Author

BTech – ARTIFICAL INTELLIGENCE AND DATA ANALYTICS 
AI-based Supply Chain Risk Modeling Project  

---

## 📌 Disclaimer

This project is a Proof of Concept (POC) built for demonstration and learning purposes.  
PR data and delay signals are simulated for modeling experimentation.




