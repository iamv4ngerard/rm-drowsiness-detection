import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- KONFIGURASI ---
model_path = r'D:\PPTI\Cawu 5\Cawu 5 Gwe\RM\rm-drowsiness-detection\face_landmarker.task'

# --- PARAMETER RISET (Bisa lo ubah-ubah buat eksperimen di paper) ---
CALIBRATION_TIME = 5.0  # Waktu kalibrasi dalam detik
ALPHA = 0.75            # Persentase toleransi (75% dari bukaan mata normal)

def calculate_ear(landmarks):
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

# --- VARIABEL KALIBRASI ---
is_calibrating = True
calibration_ears = []
ear_baseline = 0.0
dynamic_threshold = 0.0
start_time = time.time()

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    print("Sistem Aktif. Memulai proses kalibrasi...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal mengambil gambar dari kamera.")
            break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                ear_value = calculate_ear(landmarks)
                
                # ==========================================
                # FASE 1: KALIBRASI (Adaptive Thresholding)
                # ==========================================
                if is_calibrating:
                    elapsed_time = time.time() - start_time
                    calibration_ears.append(ear_value)
                    
                    # UI Panduan Kalibrasi
                    cv2.putText(frame, "FASE KALIBRASI: BUKA MATA NORMAL", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Sisa Waktu: {CALIBRATION_TIME - elapsed_time:.1f} detik", (30, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Jika waktu kalibrasi habis
                    if elapsed_time >= CALIBRATION_TIME:
                        if len(calibration_ears) > 0:
                            ear_baseline = np.mean(calibration_ears) # Cari rata-rata
                            dynamic_threshold = ear_baseline * ALPHA # Hitung batas ngantuk
                            is_calibrating = False # Pindah ke fase utama
                            print(f"Kalibrasi Selesai! Baseline: {ear_baseline:.3f}, Threshold: {dynamic_threshold:.3f}")
                        else:
                            # Reset kalau wajah nggak kedeteksi selama 5 detik
                            start_time = time.time()
                
                # ==========================================
                # FASE 2: DETEKSI UTAMA
                # ==========================================
                else:
                    color = (0, 255, 0) if ear_value > dynamic_threshold else (0, 0, 255)
                    status = "MATA TERBUKA" if ear_value > dynamic_threshold else "BERKEDIP/TUTUP"
                    
                    # UI Deteksi
                    cv2.putText(frame, f"EAR: {ear_value:.3f} | Threshold: {dynamic_threshold:.3f}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, status, (30, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Drowsiness Research Prototype', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()