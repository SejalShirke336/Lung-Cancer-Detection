
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and feature columns
model = joblib.load("lung_cancer_model.pkl")
model_columns = joblib.load("features.pkl")

st.title("🫁 Lung Cancer Prediction")
st.write(
    "This application predicts whether a person has **Lung Cancer** "
    "based on health and lifestyle details."
)



AGE = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=40,
    help="Age is an important factor as cancer risk increases with age."
)

GENDER = st.selectbox(
    "Gender",
    ["Female", "Male"],
    help="Gender is considered as lung cancer risk differs across genders."
)

SMOKING = st.selectbox(
    "Smoking",
    ["No", "Yes"],
    help="Smoking is one of the strongest risk factors for lung cancer."
)

YELLOW_FINGERS = st.selectbox(
    "Yellow Fingers",
    ["No", "Yes"],
    help="Yellow fingers may indicate long-term smoking habits."
)

ANXIETY = st.selectbox(
    "Anxiety",
    ["No", "Yes"],
    help="Anxiety may be associated with lifestyle and health conditions."
)

PEER_PRESSURE = st.selectbox(
    "Peer Pressure",
    ["No", "Yes"],
    help="Peer pressure can influence smoking and alcohol consumption."
)

CHRONIC_DISEASE = st.selectbox(
    "Chronic Disease",
    ["No", "Yes"],
    help="Existing chronic diseases can increase health risks."
)

FATIGUE = st.selectbox(
    "Fatigue",
    ["No", "Yes"],
    help="Persistent fatigue can be an early symptom of serious illness."
)

ALLERGY = st.selectbox(
    "Allergy",
    ["No", "Yes"],
    help="Allergies may affect respiratory health."
)

WHEEZING = st.selectbox(
    "Wheezing",
    ["No", "Yes"],
    help="Wheezing indicates breathing difficulty."
)

ALCOHOL_CONSUMING = st.selectbox(
    "Alcohol Consuming",
    ["No", "Yes"],
    help="Alcohol consumption combined with smoking increases cancer risk."
)

COUGHING = st.selectbox(
    "Coughing",
    ["No", "Yes"],
    help="Persistent coughing is a common symptom in lung-related conditions."
)

SHORTNESS_OF_BREATH = st.selectbox(
    "Shortness of Breath",
    ["No", "Yes"],
    help="Breathing difficulty is a key respiratory warning sign."
)

SWALLOWING_DIFFICULTY = st.selectbox(
    "Swallowing Difficulty",
    ["No", "Yes"],
    help="Difficulty swallowing may indicate internal health issues."
)

CHEST_PAIN = st.selectbox(
    "Chest Pain",
    ["No", "Yes"],
    help="Chest pain is a serious symptom that should not be ignored."
)



if st.button("🔍 Predict Lung Cancer"):

    data = pd.DataFrame(0, index=[0], columns=model_columns)

    data["AGE"] = AGE
    data["GENDER_M"] = 1 if GENDER == "Male" else 0
    data["SMOKING"] = 1 if SMOKING == "Yes" else 0
    data["YELLOW_FINGERS"] = 1 if YELLOW_FINGERS == "Yes" else 0
    data["ANXIETY"] = 1 if ANXIETY == "Yes" else 0
    data["PEER_PRESSURE"] = 1 if PEER_PRESSURE == "Yes" else 0
    data["CHRONIC_DISEASE"] = 1 if CHRONIC_DISEASE == "Yes" else 0
    data["FATIGUE"] = 1 if FATIGUE == "Yes" else 0
    data["ALLERGY"] = 1 if ALLERGY == "Yes" else 0
    data["WHEEZING"] = 1 if WHEEZING == "Yes" else 0
    data["ALCOHOL_CONSUMING"] = 1 if ALCOHOL_CONSUMING == "Yes" else 0
    data["COUGHING"] = 1 if COUGHING == "Yes" else 0
    data["SHORTNESS_OF_BREATH"] = 1 if SHORTNESS_OF_BREATH == "Yes" else 0
    data["SWALLOWING_DIFFICULTY"] = 1 if SWALLOWING_DIFFICULTY == "Yes" else 0
    data["CHEST_PAIN"] = 1 if CHEST_PAIN == "Yes" else 0

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.warning(" Lung Cancer Detected")
    else:
        st.success("No Lung Cancer Detected")
