import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time

# --- KONFIGURASI RISET ---
CSV_FILE = "evan - cerah.csv"
SEQUENCE_LENGTH = 30  
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5  # Early stopping patience

# --- 1. LOAD & PREPROCESS DATA ---
print("Memuat dataset dari:", CSV_FILE)
df = pd.read_csv(CSV_FILE)
df = df.dropna()

features = ['Norm_EAR', 'MAR_Aktual', 'PERCLOS', 'Head_Pitch_Ratio', 'Head_Yaw_Diff']
X = df[features].values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(data, labels, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:(i + seq_len)])
        ys.append(labels[i + seq_len]) # Label pada frame terakhir window
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)

# Pakai stratify agar rasio ngantuk & sadar seimbang
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# --- 2. PENINGKATAN ARSITEKTUR: GRU (Gated Recurrent Unit) TIME-SERIES ---
class DrowsinessGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(DrowsinessGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU Layer: Sangat cerdas & efisien dalam mengenali pola temporal/waktu
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layer / Classification Head
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # PENTING AGAR GPU BEKERJA KERAS: Inisialisasi hidden state menggunakan device input 
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate GRU
        # x shape: (batch_size, seq_len, features)
        out, _ = self.gru(x, h0)
        
        # GRU telah merangkum konteks/memori masa lalu. Ambil output frame paling akhir.
        out = out[:, -1, :] 
        
        # Klasifikasi ke Neuron
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

# Deteksi GPU & Percepatan Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[!] Hardware Compute yang digunakan: {str(device).upper()}")
if 'cuda' in str(device):
    print(f"[!] Nama GPU: {torch.cuda.get_device_name(0)}")

model = DrowsinessGRU(input_size=len(features)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# [PENAMBAHAN NOVELTY: Learning Rate Scheduler & Early Stopping]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# --- 3. TRAINING LOOP ---
print("\n[ Memulai Advanced Training ]")
start_time = time.time()

train_losses, test_losses = [], []
train_accs, test_accs = [], []

best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = (correct / total) * 100
    
    # Fase Validasi
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            v_loss = criterion(outputs, batch_y)
            val_loss += v_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = (val_correct / val_total) * 100
    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    test_accs.append(avg_val_acc)
    
    scheduler.step(avg_val_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}%")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_gru_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n[!] Early Stopping dipicu pada epoch {epoch+1}. Mencegah Overfitting.")
            break

print(f"\nTraining selesai dalam {time.time() - start_time:.2f} detik")

# --- 4. VISUALISASI ---
model.load_state_dict(torch.load("best_gru_model.pth", weights_only=True))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(batch_y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n--- HASIL EVALUASI MODEL TERBAIK ---")
print(f"Akurasi: {accuracy_score(y_true, y_pred) * 100:.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Validation (Test) Loss')
plt.title('Kurva Loss (Overfitting Check)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Validation Accuracy')
plt.title('Kurva Akurasi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig("grafik_riset_drowsiness.png", dpi=300)
print("\n[V] Grafik hasil training disimpan sebagai 'grafik_riset_drowsiness.png'")