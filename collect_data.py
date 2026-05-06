import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- KONFIGURASI PATH ---
model_path = r'c:\Users\evang\Documents\Cawu 5 Research Metheodology\Project RM\face_landmarker.task'

# --- PARAMETER KALIBRASI ---
CALIBRATION_TIME = 10     
is_waiting_to_start = True 
is_calibrating = False    
calibration_data = []     
calibration_start_time = 0
personal_avg_ear = 0.25   
current_label = 0         

# --- BUFFER PERCLOS ---
PERCLOS_WINDOW = 300      
perclos_buffer = []       
EAR_THRESHOLD_TEMP = 0.20 # Akan diupdate setelah kalibrasi

# --- STATISTIK LOGGING ---
data_terkumpul = 0

# --- FUNGSI KALKULASI ---
def dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (getattr(p1, 'z', 0) - getattr(p2, 'z', 0))**2)

def check_head_yaw(landmarks):
    return abs(getattr(landmarks[33], 'z', 0) - getattr(landmarks[263], 'z', 0)) > 0.04

def get_head_yaw_diff(landmarks):
    return abs(getattr(landmarks[33], 'z', 0) - getattr(landmarks[263], 'z', 0))

def calculate_ear(landmarks):
    v1_l, v2_l = dist(landmarks[160], landmarks[144]), dist(landmarks[158], landmarks[153])
    h_l = dist(landmarks[33], landmarks[133])
    ear_left = (v1_l + v2_l) / (2.0 * h_l) if h_l > 0 else 0
    
    v1_r, v2_r = dist(landmarks[385], landmarks[380]), dist(landmarks[387], landmarks[373])
    h_r = dist(landmarks[362], landmarks[263])
    ear_right = (v1_r + v2_r) / (2.0 * h_r) if h_r > 0 else 0
    
    if check_head_yaw(landmarks):
        return max(ear_left, ear_right)
    return (ear_left + ear_right) / 2.0

def calculate_mar(landmarks):
    v = dist(landmarks[13], landmarks[14])
    h = dist(landmarks[78], landmarks[308])
    return v / h if h > 0 else 0

def get_head_pitch_ratio(landmarks):
    dahi_y, hidung_y, dagu_y = landmarks[10].y, landmarks[1].y, landmarks[152].y
    jarak_atas = hidung_y - dahi_y
    jarak_bawah = dagu_y - hidung_y
    return jarak_atas / jarak_bawah if jarak_bawah > 0 else 0

# --- SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    running_mode=vision.RunningMode.VIDEO)

cap = cv2.VideoCapture(0)

# --- SETUP DATA LOGGER ---
subjek_nama = input("Masukkan Nama/Kode Subjek (misal: Budi_Gelap): ")
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"dataset_{subjek_nama}_{session_id}.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Waktu', 'EAR_Aktual', 'Norm_EAR', 'MAR_Aktual', 'PERCLOS', 'Head_Pitch_Ratio', 'Head_Yaw_Diff', 'Intensitas_Cahaya', 'Adaptive_CLAHE_Aktif', 'Label'])
print(f"Data Logger Aktif! Menyimpan ke {csv_filename}")

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_frame)
        is_low_light = avg_brightness < 80 
        
        if is_low_light:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            frame_processed = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            env_status = f"Gelap ({avg_brightness:.0f}) | CLAHE: ON"
        else:
            frame_processed = frame
            env_status = f"Terang ({avg_brightness:.0f}) | CLAHE: OFF"

        frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                ear_value = calculate_ear(landmarks)
                
                if is_waiting_to_start:
                    cv2.putText(frame, "--- MODE PENGUMPULAN DATA ---", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, "Tekan 'S' untuk Kalibrasi Normal", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif is_calibrating:
                    elapsed = time.time() - calibration_start_time
                    rem = max(0, int(CALIBRATION_TIME - elapsed))
                    calibration_data.append(ear_value)
                    
                    cv2.putText(frame, f"KALIBRASI... Tahan wajah {rem} dtk", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if elapsed >= CALIBRATION_TIME:
                        is_calibrating = False
                        if calibration_data:
                            calibration_data.sort(reverse=True)
                            personal_avg_ear = np.mean(calibration_data[:len(calibration_data)//2])
                            EAR_THRESHOLD_TEMP = personal_avg_ear * 0.75
                            print(f"Kalibrasi OK! Rata-rata EAR: {personal_avg_ear:.3f}")

                else:
                    # Menghitung Fitur
                    mar_value = calculate_mar(landmarks)
                    head_pitch = get_head_pitch_ratio(landmarks)
                    head_yaw = get_head_yaw_diff(landmarks)
                    norm_ear = ear_value / personal_avg_ear if personal_avg_ear > 0 else 1.0
                    
                    # Update PERCLOS
                    perclos_buffer.append(1 if ear_value < EAR_THRESHOLD_TEMP else 0)
                    if len(perclos_buffer) > PERCLOS_WINDOW: perclos_buffer.pop(0)
                    current_perclos = sum(perclos_buffer) / len(perclos_buffer) if len(perclos_buffer) > 0 else 0
                    
                    # Simpan ke CSV
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        writer.writerow([current_time, round(ear_value,4), round(norm_ear,4), round(mar_value,4), 
                                         round(current_perclos,4), round(head_pitch,4), round(head_yaw,4), 
                                         round(avg_brightness,2), 1 if is_low_light else 0, current_label])
                    data_terkumpul += 1
                    
                    # TAMPILAN DASHBOARD PEREKAMAN
                    if current_label == 1:
                        bg_color, text_str = (0, 0, 255), "SEDANG MEREKAM: NGANTUK (1)"
                    else:
                        bg_color, text_str = (0, 200, 0), "SEDANG MEREKAM: SADAR (0)"
                        
                    cv2.rectangle(frame, (0, 0), (640, 40), bg_color, -1)
                    cv2.putText(frame, text_str, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    cv2.putText(frame, f"Total Data Tersimpan: {data_terkumpul} baris", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    cv2.putText(frame, f"Tekan '1' (Ngantuk) | '0' (Sadar) | 'ESC' (Keluar)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.putText(frame, env_status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow('Drowsiness Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key in [ord('s'), ord('S')] and is_waiting_to_start:
            is_waiting_to_start = False
            is_calibrating = True
            calibration_start_time = time.time()
        elif key == ord('0'): current_label = 0
        elif key == ord('1'): current_label = 1

cap.release()
cv2.destroyAllWindows()
