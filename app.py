import streamlit as st
import pandas as pd
import joblib

# Load model & columns
model = joblib.load("lung_cancer_model.pkl")
features = joblib.load("features.pkl")   # saved X.columns

st.set_page_config(page_title="Cancer Prediction", layout="centered")

st.title("🫁 Lung Cancer Prediction App")
st.write("Predict whether a person has lung cancer using medical indicators")

# Mapping
yes_no = {"No": 0, "Yes": 1}

# Inputs
AGE = st.number_input("Age", min_value=1, max_value=100, value=40)
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

if st.button("🔍 Predict"):

    # Step 1: Create empty dataframe with training columns
    input_df = pd.DataFrame(0, index=[0], columns=features)

    # Step 2: Assign values (NO loops)
    input_df["AGE"] = AGE

    if "GENDER_M" in input_df.columns:
        input_df["GENDER_M"] = 1 if GENDER == "Male" else 0

    input_df["SMOKING"] = yes_no[SMOKING]
    input_df["YELLOW_FINGERS"] = yes_no[YELLOW_FINGERS]
    input_df["ANXIETY"] = yes_no[ANXIETY]
    input_df["PEER_PRESSURE"] = yes_no[PEER_PRESSURE]
    input_df["CHRONIC_DISEASE"] = yes_no[CHRONIC_DISEASE]
    input_df["FATIGUE"] = yes_no[FATIGUE]
    input_df["ALLERGY"] = yes_no[ALLERGY]
    input_df["WHEEZING"] = yes_no[WHEEZING]
    input_df["ALCOHOL_CONSUMING"] = yes_no[ALCOHOL_CONSUMING]
    input_df["COUGHING"] = yes_no[COUGHING]
    input_df["SHORTNESS_OF_BREATH"] = yes_no[SHORTNESS_OF_BREATH]
    input_df["SWALLOWING_DIFFICULTY"] = yes_no[SWALLOWING_DIFFICULTY]
    input_df["CHEST_PAIN"] = yes_no[CHEST_PAIN]

    # Step 3: Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Cancer Detected (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Cancer Detected (Probability: {probability*100:.2f}%)")

st.caption("⚠️ Educational purpose only")
