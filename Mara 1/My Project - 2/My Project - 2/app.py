# app.py
import streamlit as st
import numpy as np
import joblib
import re

# -------------------------------
# Load Artifacts
# -------------------------------
age_scaler = joblib.load("age_scaler.pkl")
tfidf_symptoms = joblib.load("tfidf_symptoms.pkl")
svd_symptoms = joblib.load("svd_symptoms.pkl")
tfidf_medhist = joblib.load("tfidf_medhist.pkl")
svd_medhist = joblib.load("svd_medhist.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_meta = joblib.load("feature_meta.pkl")
fast_model = joblib.load("fast_inference_model.pkl")

# -------------------------------
# Helper Functions
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def preprocess_input(age, symptoms, medical_history):
    # Scale Age
    age_norm = age_scaler.transform(np.array(age).reshape(-1, 1))

    # Symptoms -> TF-IDF + SVD
    symptoms_clean = [clean_text(symptoms)]
    symptoms_tfidf = tfidf_symptoms.transform(symptoms_clean)
    symptoms_svd = svd_symptoms.transform(symptoms_tfidf)

    # Medical History -> TF-IDF + SVD
    medhist_clean = [clean_text(medical_history)]
    medhist_tfidf = tfidf_medhist.transform(medhist_clean)
    medhist_svd = svd_medhist.transform(medhist_tfidf)

    # Combine
    X_features = np.hstack([age_norm, symptoms_svd, medhist_svd])
    return X_features

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🩺 Patient Diagnosis Prediction", layout="centered")

st.title("🩺 Patient Diagnosis Prediction App")
st.write("Provide patient details below to predict the **possible diagnosis**.")

# Input fields
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
symptoms = st.text_area("Enter Symptoms", placeholder="e.g., fever, cough, headache")
medical_history = st.text_area("Enter Medical History", placeholder="e.g., diabetes, hypertension")

# Predict button
if st.button("🔍 Predict Diagnosis"):
    if symptoms.strip() == "" or medical_history.strip() == "":
        st.warning("⚠️ Please enter both symptoms and medical history.")
    else:
        # Preprocess
        X_input = preprocess_input(age, symptoms, medical_history)

        # Predict
        y_pred = fast_model.predict(X_input)[0]
        y_prob = fast_model.predict_proba(X_input)[0]

        # Decode prediction
        predicted_diagnosis = label_encoder.inverse_transform([y_pred])[0]

        # Show result
        st.success(f"✅ Predicted Diagnosis: **{predicted_diagnosis}**")

        # Show probability chart
        top_n = 5
        sorted_idx = np.argsort(y_prob)[::-1][:top_n]
        top_labels = label_encoder.inverse_transform(sorted_idx)
        top_probs = y_prob[sorted_idx]

        st.subheader("🔎 Top Possible Diagnoses")
        for label, prob in zip(top_labels, top_probs):
            st.write(f"- {label}: {prob:.2%}")
