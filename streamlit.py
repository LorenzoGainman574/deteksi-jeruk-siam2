import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Kematangan Jeruk", layout="centered")
st.title("🍊 Deteksi Kematangan Jeruk (TFLite)")
st.write("Ambil foto jeruk agar sistem dapat mendeteksi kualitasnya")

# 2. Inisialisasi Model TFLite
@st.cache_resource
def load_tflite_model():
    # Pastikan file final_model.tflite ada di direktori utama GitHub
    MODEL_PATH = "final_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Konfigurasi Teknis
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# 3. Input Kamera
img_file = st.camera_input("Arahkan jeruk ke tengah kamera")

if img_file is not None:
    # --- Load Gambar ---
    image = Image.open(img_file)
    frame = np.array(image)
    # Konversi ke BGR untuk diproses OpenCV
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w, _ = display_frame.shape

    # --- Tentukan Koordinat Kotak Tengah (ROI) ---
    box_size = 250 
    x1, y1 = (w // 2) - (box_size // 2), (h // 2) - (box_size // 2)
    x2, y2 = (w // 2) + (box_size // 2), (h // 2) + (box_size // 2)

    # Potong area ROI untuk diprediksi
    roi = display_frame[y1:y2, x1:x2]
    
    # --- Preprocessing ROI ---
    resized_roi = cv2.resize(roi, IMG_SIZE)
    normalized_roi = resized_roi.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized_roi, axis=0)

    # --- Prediksi TFLite ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction_score = predictions[0][0]

    # --- Logika Label & Warna Kotak ---
    if prediction_score > THRESHOLD:
        label = f"Manis (Kuning): {prediction_score*100:.1f}%"
        rect_color = (0, 255, 255) # Kuning (BGR)
    else:
        label = f"Asam (Hijau): {(1-prediction_score)*100:.1f}%"
        rect_color = (0, 255, 0)   # Hijau (BGR)

    # --- Gambar Elemen Visual di Hasil Foto ---
    # Gambar kotak di tengah
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), rect_color, 8)
    
    # Tampilkan teks label di atas kotak
    cv2.putText(display_frame, label, (x1, y1 - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, rect_color, 3, cv2.LINE_AA)

    # Konversi balik ke RGB untuk ditampilkan di Streamlit
    result_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    # --- Tampilkan Hasil Akhir ---
    st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
    
    if prediction_score > THRESHOLD:
        st.success(f"Hasil Prediksi: {label}")
    else:
        st.error(f"Hasil Prediksi: {label}")
