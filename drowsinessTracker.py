import cv2
import mediapipe as mp
import numpy as np
import time

# Import sesuai struktur sukses tadi
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- KONFIGURASI ---
# Gunakan r'...' agar path dibaca dengan benar oleh Windows
model_path = r'c:\Users\evang\Documents\Cawu 5 Research Metheodology\Project RM\face_landmarker.task'

def calculate_ear(landmarks):
    # Titik landmark mata (Indeks MediaPipe)
    def dist(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    # Mata Kiri
    v1_l = dist(landmarks[160], landmarks[144])
    v2_l = dist(landmarks[158], landmarks[153])
    h_l = dist(landmarks[33], landmarks[133])
    ear_left = (v1_l + v2_l) / (2.0 * h_l)
    
    return ear_left

# Inisialisasi Detektor
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    running_mode=vision.RunningMode.VIDEO)

cap = cv2.VideoCapture(0)

# Gunakan 'with' agar detector tertutup otomatis jika error
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    print("Sistem Aktif. Tekan 'ESC' untuk keluar.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal mengambil gambar dari kamera.")
            break

        # Balik frame agar seperti cermin (opsional tapi lebih nyaman)
        frame = cv2.flip(frame, 1)
        
        # Konversi ke MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Menggunakan milidetik yang presisi untuk sinkronisasi
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Deteksi wajah
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                ear_value = calculate_ear(landmarks)
                
                # Visualisasi
                color = (0, 255, 0) if ear_value > 0.22 else (0, 0, 255)
                status = "MATA TERBUKA" if ear_value > 0.22 else "BERKEDIP/TUTUP"
                
                cv2.putText(frame, f"EAR: {ear_value:.2f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, status, (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Tampilkan Jendela
        cv2.imshow('Drowsiness Research Prototype', frame)

        # Keluar jika tekan tombol ESC (27)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()