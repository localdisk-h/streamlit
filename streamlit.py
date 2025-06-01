import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_model.pkl")

st.title("Prediksi Diabetes - Pima Dataset")

# Input Features
preg = st.number_input("Jumlah Kehamilan", min_value=0)
glu = st.number_input("Glukosa", min_value=0)
bp = st.number_input("Tekanan Darah", min_value=0)
skin = st.number_input("Tebal Kulit", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Usia", min_value=1)

if st.button("Prediksi"):
    input_data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    hasil = "POSITIF Diabetes" if prediction == 1 else "NEGATIF Diabetes"
    st.success(f"Hasil Prediksi: {hasil}")
