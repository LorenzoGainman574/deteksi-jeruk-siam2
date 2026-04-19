import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Jeruk Real-time", layout="centered")
st.title("🍊 Deteksi Kematangan Jeruk Real-time")
st.text("Arahkan jeruk tepat ke dalam kotak di layar.")

# 2. Inisialisasi Model TFLite
@st.cache_resource
def load_tflite_model():
    # Pastikan 'final_model.tflite' ada di folder utama
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

# 3. Kontrol Kamera Streamlit
run = st.checkbox('Mulai Kamera', value=True)
FRAME_WINDOW = st.image([]) # Tempat video akan ditayangkan

if run:
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses kamera.")
            break

        # Convert BGR (OpenCV) ke RGB (Streamlit)
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = display_frame.shape

        # --- Buat Kotak Patokan (ROI) di Tengah ---
        box_size = 250
        x1 = (w // 2) - (box_size // 2)
        y1 = (h // 2) - (box_size // 2)
        x2 = (w // 2) + (box_size // 2)
        y2 = (h // 2) + (box_size // 2)

        # Potong area spesifik di dalam kotak untuk diprediksi
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

        # --- Logika Warna & Teks Label ---
        if prediction_score > THRESHOLD:
            label = f"Manis: {prediction_score*100:.1f}%"
            color = (255, 255, 0) # Kuning (RGB)
        else:
            label = f"Asam: {(1-prediction_score)*100:.1f}%"
            color = (0, 255, 0)   # Hijau (RGB)

        # --- Gambar Kotak & Teks di Layar ---
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(display_frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Tayangkan frame yang sudah diedit ke Streamlit
        FRAME_WINDOW.image(display_frame)
    
    cap.release()
else:
    st.write("Kamera dimatikan.")
