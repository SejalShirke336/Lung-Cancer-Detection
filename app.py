import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & feature order
model = joblib.load("lung_cancer_model.pkl")
features = joblib.load("features.pkl")   # list of column names in order

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")

st.title("🫁 Lung Cancer Prediction")
st.write("Classification model to predict whether a person has cancer")

yes_no = {"No": 0, "Yes": 1}

# Inputs
AGE = st.number_input("Age", 1, 100, 40)
GENDER = st.selectbox("Gender", ["Female", "Male"])
SMOKING = st.selectbox("Smoking", ["No", "Yes"])
YELLOW_FINGERS = st.selectbox("Yellow Fingers", ["No", "Yes"])
ANXIETY = st.selectbox("Anxiety", ["No", "Yes"])
PEER_PRESSURE = st.selectbox("Peer Pressure", ["No", "Yes"])
CHRONIC_DISEASE = st.selectbox("Chronic Disease", ["No", "Yes"])
FATIGUE = st.selectbox("Fatigue", ["No", "Yes"])
ALLERGY = st.selectbox("Allergy", ["No", "Yes"])
WHEEZING = st.selectbox("Wheezing", ["No", "Yes"])
ALCOHOL_CONSUMING = st.selectbox("Alcohol Consuming", ["No", "Yes"])
COUGHING = st.selectbox("Coughing", ["No", "Yes"])
SHORTNESS_OF_BREATH = st.selectbox("Shortness of Breath", ["No", "Yes"])
SWALLOWING_DIFFICULTY = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
CHEST_PAIN = st.selectbox("Chest Pain", ["No", "Yes"])

if st.button("🔍 Predict Cancer"):

    # Create zero-filled dataframe
    input_df = pd.DataFrame(0, index=[0], columns=features)

    # Assign values (no loops, no lambda)
    input_df["AGE"] = AGE
    if "GENDER_M" in input_df.columns:
        input_df["GE]()_
