import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("lung_cancer_model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Cancer Prediction App", layout="centered")

st.title("🫁 Cancer Prediction System")
st.write("Predict whether a person has cancer based on health and lifestyle information.")

st.markdown("---")

yes_no = {"No": 0, "Yes": 1}

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

st.markdown("---")

if st.button("🔍 Predict Cancer"):

    input_df = pd.DataFrame(columns=features)

    input_df.loc[0, "AGE"] = AGE
    input_df.loc[0, "GENDER_M"] = 1 if GENDER == "Male" else 0
    input_df.loc[0, "SMOKING"] = yes_no[SMOKING]
    input_df.loc[0, "YELLOW_FINGERS"] = yes_no[YELLOW_FINGERS]
    input_df.loc[0, "ANXIETY"] = yes_no[ANXIETY]
    input_df.loc[0, "PEER_PRESSURE"] = yes_no[PEER_PRESSURE]
    input_df.loc[0, "CHRONIC_DISEASE"] = yes_no[CHRONIC_DISEASE]
    input_df.loc[0, "FATIGUE"] = yes_no[FATIGUE]
    input_df.loc[0, "ALLERGY"] = yes_no[ALLERGY]
    input_df.loc[0, "WHEEZING"] = yes_no[WHEEZING]
    input_df.loc[0, "ALCOHOL_CONSUMING"] = yes_no[ALCOHOL_CONSUMING]
    input_df.loc[0, "COUGHING"] = yes_no[COUGHING]
    input_df.loc[0, "SHORTNESS_OF_BREATH"] = yes_no[SHORTNESS_OF_BREATH]
    input_df.loc[0, "SWALLOWING_DIFFICULTY"] = yes_no[SWALLOWING_DIFFICULTY]
    input_df.loc[0, "CHEST_PAIN"] = yes_no[CHEST_PAIN]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Cancer Detected (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Cancer Detected (Probability: {probability*100:.2f}%)")

st.markdown("---")
st.caption("⚠️ This application is for educational purposes only.")
