import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Jeruk Real-time", layout="centered")
st.title("🍊 Deteksi Kematangan Jeruk Real-time")

# 2. Inisialisasi Model TFLite
@st.cache_resource
def load_tflite_model():
    MODEL_PATH = "final_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class OrangeClassifier(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # --- Preprocessing ---
        h, w, _ = img.shape
        box_size = 250
        x1, y1 = (w // 2) - (box_size // 2), (h // 2) - (box_size // 2)
        x2, y2 = (w // 2) + (box_size // 2), (h // 2) + (box_size // 2)
        
        roi = img[y1:y2, x1:x2]
        resized = cv2.resize(roi, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # --- Prediksi TFLite ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        score = predictions[0][0]

        # --- Visualisasi ---
        if score > 0.5:
            label = f"Manis: {score*100:.1f}%"
            color = (0, 255, 255) # Kuning
        else:
            label = f"Asam: {(1-score)*100:.1f}%"
            color = (0, 255, 0)   # Hijau

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return img

# 3. Jalankan Kamera WebRTC
webrtc_streamer(key="orange-deteksi", video_transformer_factory=OrangeClassifier)
