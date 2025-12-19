import streamlit as st
import pandas as pd
import joblib

# Load saved model and feature columns
model = joblib.load("lung_cancer_model.pkl")
features = joblib.load("features.pkl")  # X.columns after get_dummies

st.set_page_config(page_title="Cancer Prediction App", layout="centered")

st.title("🫁 Cancer Prediction System")
st.write("Predict whether a person has cancer based on health and lifestyle information.")

st.markdown("---")

# Mapping categorical values to numeric
yes_no = {"No": 0, "Yes": 1}

# User Inputs (NO loops, NO lambda)
AGE = st.number_input("Age", min_value=1, max_value=100, value=40)

GENDER = st.selectbox("Gender", ["Female", "Male"])
SMOKING = st.selectbox("Smoking", ["No", "Yes"])
YELLOW_FINGERS = st.selectbox("Yellow Fingers", ["No", "Yes"])
ANXIETY = st.selectbox("Anxiety", ["No", "Yes"])
PEER_PRESSURE = st.selectbox("Peer Pressure", ["No", "Yes"])
CHRONIC_DISEASE = st.selectbox("Chronic Disease", ["No", "Yes"])
FATIGUE = st.selectbox("Fatigue", ["No", "Yes"])
ALLERGY = st.selectbox("Allerg
