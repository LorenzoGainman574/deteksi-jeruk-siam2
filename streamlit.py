import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Deteksi Kematangan Jeruk", layout="centered")
st.title("🍊 Deteksi Kematangan Jeruk Real-time")
st.text("Arahkan jeruk ke dalam kotak di tengah kamera")

# 2. Inisialisasi Model
@st.cache_resource
def load_my_model():
    # Menggunakan cache agar model tidak dimuat ulang setiap kali script berjalan
    MODEL_PATH = "model/final_model.h5"
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

# Konfigurasi Teknis
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# 3. Kontrol Sidebar
run = st.checkbox('Nyalakan Kamera', value=True)
FRAME_WINDOW = st.image([]) # Placeholder untuk frame video

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Gagal mengakses kamera.")
        break

    # Salinan untuk pengolahan & tampilan
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = display_frame.shape

    # --- Tentukan Koordinat Kotak Tengah ---
    box_size = 250 # Ukuran kotak (pixel)
    x1, y1 = (w // 2) - (box_size // 2), (h // 2) - (box_size // 2)
    x2, y2 = (w // 2) + (box_size // 2), (h // 2) + (box_size // 2)

    # Potong area di dalam kotak untuk diproses model (ROI - Region of Interest)
    roi = display_frame[y1:y2, x1:x2]
    
    # --- Preprocessing ROI ---
    resized_roi = cv2.resize(roi, IMG_SIZE)
    normalized_roi = resized_roi.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized_roi, axis=0)

    # --- Prediksi ---
    predictions = model.predict(input_data, verbose=0)
    prediction_score = predictions[0][0]

    # --- Logika Label & Warna Kotak ---
    if prediction_score > THRESHOLD:
        label = f"Manis (Kuning): {prediction_score*100:.1f}%"
        rect_color = (255, 255, 0) # Kuning (RGB di Streamlit/PIL)
    else:
        label = f"Asam (Hijau): {(1-prediction_score)*100:.1f}%"
        rect_color = (0, 255, 0)   # Hijau (RGB)

    # --- Gambar Elemen Visual ---
    # Gambar kotak di tengah
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), rect_color, 4)
    
    # Tampilkan teks label di atas kotak
    cv2.putText(display_frame, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2, cv2.LINE_AA)

    # Tampilkan ke Streamlit
    FRAME_WINDOW.image(display_frame)

else:
    cap.release()
    st.write("Kamera dimatikan.")