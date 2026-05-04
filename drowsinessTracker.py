import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime

# Import sesuai struktur sukses tadi
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- KONFIGURASI ---
# Gunakan r'...' agar path dibaca dengan benar oleh Windows
model_path = r'c:\Users\evang\Documents\Cawu 5 Research Metheodology\Project RM\face_landmarker.task'

# Parameter Deteksi Ngantuk
EAR_THRESHOLD = 0.22      # Ambang ini akan dioverride oleh sistem kalibrasi
DROWSY_FRAMES = 15        # Jumlah frame mata berada di bawah threshold untuk memicu alarm ngantuk
frame_counter = 0         # Penghitung frame mata tertutup berjalan

# --- PARAMETER MULTI-MODAL (BARU) ---
# 1. MAR (Mouth Aspect Ratio) untuk Menguap
MAR_THRESHOLD = 0.60      # Ambang batas mulut terbuka (nguap)
YAWN_FRAMES = 10          # Berapa frame mulut terbuka untuk dihitung 1 kali menguap
yawn_counter = 0

# 2. PERCLOS (Percentage of Eye Closure)
PERCLOS_WINDOW = 300      # History 300 frame terakhir (sekitar 10-15 detik)
perclos_buffer = []       # Menyimpan status mata tiap frame
PERCLOS_THRESHOLD = 0.20  # Jika lebih dari 20% total waktu mata tertutup -> Bahaya

# 3. Head Position (Kepala Menunduk/Nodding)
HEAD_DROP_FRAMES = 10
head_drop_counter = 0

# 4. Inattention / Menoleh ke Samping (Yaw)
YAW_FRAMES = 15
yaw_counter = 0

# --- PARAMETER KALIBRASI PERSONAL ---
CALIBRATION_TIME = 10     # Durasi kalibrasi di awal (10 detik)
is_waiting_to_start = True # Flag untuk menunggu user menekan START
is_calibrating = False    # Akan menjadi True setelah tombol 'S' ditekan
calibration_data = []     # Menyimpan nilai histori EAR selama kalibrasi
calibration_start_time = 0

# --- FUNGSI KALKULASI LANDMARK ---
def dist(p1, p2):
    # Perbaikan kalkulasi Pythagoras 3D (X, Y, Z)
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (getattr(p1, 'z', 0) - getattr(p2, 'z', 0))**2)

def check_head_yaw(landmarks):
    # Bandingkan kedalaman (Z-axis) dari ujung mata kiri dan kanan
    z_diff = getattr(landmarks[33], 'z', 0) - getattr(landmarks[263], 'z', 0)
    return abs(z_diff) > 0.04

def calculate_ear(landmarks):
    # Mata Kiri: Horizontal(33, 133), Vertikal rata-rata(160-144, 158-153)
    v1_l = dist(landmarks[160], landmarks[144])
    v2_l = dist(landmarks[158], landmarks[153])
    h_l = dist(landmarks[33], landmarks[133])
    ear_left = (v1_l + v2_l) / (2.0 * h_l) if h_l > 0 else 0
    
    # Mata Kanan: Horizontal(362, 263), Vertikal rata-rata(385-380, 387-373)
    v1_r = dist(landmarks[385], landmarks[380])
    v2_r = dist(landmarks[387], landmarks[373])
    h_r = dist(landmarks[362], landmarks[263])
    ear_right = (v1_r + v2_r) / (2.0 * h_r) if h_r > 0 else 0
    
    # Jika menoleh (Yaw), ambil EAR dari mata yang masih paling terbuka jelas
    if check_head_yaw(landmarks):
        return max(ear_left, ear_right)
    return (ear_left + ear_right) / 2.0

def calculate_mar(landmarks):
    # Inner lips: Top(13), Bottom(14), Left(78), Right(308)
    v = dist(landmarks[13], landmarks[14])
    h = dist(landmarks[78], landmarks[308])
    return v / h if h > 0 else 0

def check_head_drop(landmarks):
    # Proyeksi 2D sederhana wajah (Pitch)
    # Dahi: 10, Hidung: 1, Dagu: 152
    dahi_y = landmarks[10].y
    hidung_y = landmarks[1].y
    dagu_y = landmarks[152].y
    
    jarak_atas = hidung_y - dahi_y
    jarak_bawah = dagu_y - hidung_y
    
    # Saat menunduk, dagu bergerak makin dekat ke pusat wajah di kamera
    rasio = jarak_atas / jarak_bawah if jarak_bawah > 0 else 0
    return rasio > 1.8 # Nilai threshold jika wajah sangat tunduk

# Inisialisasi Detektor
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    running_mode=vision.RunningMode.VIDEO)

cap = cv2.VideoCapture(0)

# --- SETUP DATA LOGGER (CSV) UNTUK PEMBUKTIAN RISET ---
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"riset_mata_cahaya_{session_id}.csv"

# Membuat file CSV dan menulis Header (Kolom)
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Waktu', 
        'EAR_Aktual', 
        'Threshold_Personal', 
        'Intensitas_Cahaya', 
        'Adaptive_CLAHE_Aktif', 
        'Kondisi_Mata', 
        'Status_Sistem'
    ])
print(f"Data Logger Aktif: Menyimpan rekaman ke {csv_filename}")

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
        
        # --- 1. ADAPTIVE LOW-LIGHT DETECTION ---
        # Hitung rata-rata kecerahan (brightness) frame menggunakan Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_frame)
        
        # Jika nilai rata-rata pixel di bawah 80 (dari skala 0-255), kita anggap gelap
        is_low_light = avg_brightness < 80 
        
        if is_low_light:
            # --- PREPROCESSING UNTUK PENCAHAYAAN KURANG (LOW LIGHT) ---
            # Menggunakan CLAHE
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            lab_clahe = cv2.merge((cl, a, b))
            frame_processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            env_status = f"Gelap ({avg_brightness:.1f}) - CLAHE ON"
        else:
            # Jika ruangan terang, bypass CLAHE untuk menghemat CPU/Meningkatkan FPS
            frame_processed = frame
            env_status = f"Terang ({avg_brightness:.1f}) - Normal"

        # MediaPipe membutuhkan format RGB (OpenCV menggunakan BGR)
        frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
        
        # Konversi ke MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Menggunakan milidetik yang presisi untuk sinkronisasi
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Deteksi wajah
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                ear_value = calculate_ear(landmarks)
                
                if is_waiting_to_start:
                    # --- FASE PERSIAPAN ---
                    # Menunggu user menekan tombol 'S' untuk mulai
                    cv2.putText(frame, "SIAPKAN POSISI WAJAH & BUKA MATA NORMAL", (30, frame.shape[0]//2 - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, "TEKAN TOMBOL 'S' UNTUK MULAI KALIBRASI", (30, frame.shape[0]//2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Tampilkan live EAR sementara agar bisa bercermin
                    cv2.putText(frame, f"Live EAR Sementara: {ear_value:.2f}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                elif is_calibrating:
                    # --- FASE KALIBRASI AWAL (10 DETIK) ---
                    elapsed = time.time() - calibration_start_time
                    remaining = max(0, int(CALIBRATION_TIME - elapsed))
                    
                    calibration_data.append(ear_value)
                    
                    # Tampilan UI saat kalibrasi
                    cv2.putText(frame, "FASE KALIBRASI: BUKA MATA NORMAL KE KAMERA", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Sisa Waktu: {remaining} detik", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"EAR Saat Ini: {ear_value:.2f}", (30, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if elapsed >= CALIBRATION_TIME:
                        is_calibrating = False
                        if calibration_data:
                            # Mengambil 50% data EAR tertinggi (mengabaikan momen berkedip saat kalibrasi)
                            calibration_data.sort(reverse=True)
                            top_half = calibration_data[:len(calibration_data)//2]
                            personal_avg_ear = np.mean(top_half) if top_half else 0.25
                            
                            # Menetapkan threshold dinamis (misal: 75% dari bukaan mata normal orang tersebut)
                            EAR_THRESHOLD = personal_avg_ear * 0.75
                            print(f"Kalibrasi Selesai! EAR Rata-rata Pribadi: {personal_avg_ear:.3f} | Threshold Baru: {EAR_THRESHOLD:.3f}")

                else:
                    # --- LOGIKA DETEKSI NGANTUK (MULTI-MODAL) ---
                    mar_value = calculate_mar(landmarks)
                    is_head_dropped = check_head_drop(landmarks)
                    is_eye_closed = ear_value < EAR_THRESHOLD
                    
                    # 1. Update Engine PERCLOS
                    perclos_buffer.append(1 if is_eye_closed else 0)
                    if len(perclos_buffer) > PERCLOS_WINDOW:
                        perclos_buffer.pop(0)
                        
                    current_perclos = sum(perclos_buffer) / len(perclos_buffer) if len(perclos_buffer) > 0 else 0

                    # 2. Logic Accumulator Status
                    alarm_triggered = False
                    status_list = []

                    # Cek Mata (EAR)
                    if is_eye_closed:
                        frame_counter += 1
                        if frame_counter >= DROWSY_FRAMES:
                            alarm_triggered = True
                            status_list.append("MATA TERPEJAM")
                    else:
                        frame_counter = 0
                        
                    # Cek Mulut / Menguap (MAR)
                    if mar_value > MAR_THRESHOLD:
                        yawn_counter += 1
                        if yawn_counter >= YAWN_FRAMES:
                            status_list.append("MENGUAP")
                    else:
                        yawn_counter = 0

                    # Cek Menunduk (Head Drop)
                    if is_head_dropped:
                        head_drop_counter += 1
                        if head_drop_counter >= HEAD_DROP_FRAMES:
                            alarm_triggered = True
                            status_list.append("MENGANTUK (MENUNDUK)")
                    else:
                        head_drop_counter = 0
                        
                    # Cek Inattention / Distraksi Menoleh (Yaw)
                    if check_head_yaw(landmarks):
                        yaw_counter += 1
                        if yaw_counter >= YAW_FRAMES:
                            status_list.append("DISTRAKSI (MENOLEH)")
                    else:
                        yaw_counter = 0

                    # Cek Evaluasi Jangka Panjang (PERCLOS)
                    if current_perclos > PERCLOS_THRESHOLD:
                        alarm_triggered = True
                        status_list.append("PERCLOS BURUK")

                    # --- VISUALISASI LAYAR ---
                    if alarm_triggered:
                        color = (0, 0, 255) # Merah (Kritis)
                        cv2.putText(frame, "!!! BAHAYA NGANTUK !!!", (50, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        status_bawah = " | ".join(status_list)
                    elif len(status_list) > 0:
                        color = (0, 165, 255) # Orange (Indikasi Awal)
                        status_bawah = "INDIKASI: " + " | ".join(status_list)
                    elif is_eye_closed:
                        color = (0, 255, 255) # Kuning
                        status_bawah = "Mata Sedang Berkedip..."
                    else:
                        color = (0, 255, 0) # Hijau
                        status_bawah = "NORMAL BERSIAGA"

                    # Print Info Parameter Utama
                    cv2.putText(frame, f"Status: {status_bawah}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"EAR: {ear_value:.2f} (Batas: {EAR_THRESHOLD:.2f})", (30, 85), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"MAR: {mar_value:.2f} | PERCLOS: {current_perclos*100:.1f}%", (30, 115), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # --- LOGGING DATA KE CSV SECARA REAL-TIME ---
                    # Data dicatat HANYA SETELAH kalibrasi selesai (agar adil)
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        writer.writerow([
                            current_time,
                            round(ear_value, 4),           # Data Mentah (EAR)
                            round(EAR_THRESHOLD, 4),       # Batas Personal (Nilai Unik per Individu)
                            round(avg_brightness, 2),      # Kondisi Cahaya Lingkungan (Indoor)
                            1 if is_low_light else 0,      # Status ML Light-Invariant (1=On, 0=Off)
                            "Tutup" if is_eye_closed else "Buka",
                            status_bawah                   # Status (Ngantuk/Normal)
                        ])

        # Tambahkan visualisasi status kecerahan lingkungan
        cv2.putText(frame, env_status, (30, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Tampilkan Jendela
        cv2.imshow('Drowsiness Research Prototype', frame)

        # Input Tombol Keyboard
        key = cv2.waitKey(1) & 0xFF
        
        # Keluar jika tekan tombol ESC (27)
        if key == 27:
            break
        # Mulai kalibrasi jika tekan tombol S
        elif key == ord('s') or key == ord('S'):
            if is_waiting_to_start:
                is_waiting_to_start = False
                is_calibrating = True
                calibration_start_time = time.time()
                print("Memulai Kalibrasi Baseline...")

cap.release()
cv2.destroyAllWindows()