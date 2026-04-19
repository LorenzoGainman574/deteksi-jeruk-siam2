import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Jeruk Siam", layout="centered")
st.title("🍊 Deteksi Kematangan Jeruk Real-time")
st.write("Gunakan kamera di bawah untuk mengambil foto jeruk")

# 2. Inisialisasi Model TFLite
@st.cache_resource
def load_tflite_model():
    # Pastikan file final_model.tflite ada di root (luar folder)
    MODEL_PATH = "final_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Input Kamera (Fitur Real-time Streamlit)
img_file = st.camera_input("Ambil Foto Jeruk")

if img_file is not None:
    # Konversi file foto ke format yang bisa diproses
    image = Image.open(img_file)
    frame = np.array(image)
    
    # Preprocessing (Sesuai dengan input model skripsi kamu)
    # Resize ke 224x224
    resized = cv2.resize(frame, (224, 224))
    # Normalisasi 0-1
    normalized = resized.astype(np.float32) / 255.0
    # Tambah dimensi batch
    input_data = np.expand_dims(normalized, axis=0)

    # --- Prediksi TFLite ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction_score = predictions[0][0]

    # --- Tampilkan Hasil ---
    st.subheader("Hasil Analisis:")
    if prediction_score > 0.5:
        st.success(f"**MANIS (Kuning)**")
        st.write(f"Tingkat Keyakinan: {prediction_score*100:.2f}%")
    else:
        st.error(f"**ASAM (Hijau)**")
        st.write(f"Tingkat Keyakinan: {(1-prediction_score)*100:.2f}%")
