import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("fruit_classifier.h5")
class_names = ['Apple', 'Banana', 'Orange']  # Sesuaikan dengan label asli

st.title("Klasifikasi Buah dengan CNN")

uploaded = st.file_uploader("Upload gambar buah...", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded).resize((100, 100))
    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

    img_array = np.expand_dims(np.array(image)/255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediksi: {predicted_class}")

