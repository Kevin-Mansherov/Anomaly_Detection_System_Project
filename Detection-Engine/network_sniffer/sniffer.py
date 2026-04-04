import os
import sys
import glob
import torch
import joblib
import pandas as pd 
from scapy.all import sniff, IP, TCP


#Path Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")

sys.path.append(PROJECT_ROOT)
from models.model import RBM

#Hyperparameters Configuration
PACKET_VISIBLE_UNITS = 12
PACKET_HIDDEN_UNITS = 256
PACKET_THRESHOLD = 3.369822
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

packet_model = None
packet_scaler = None

#Functions
def get_latest_model_path(model_prefix):
    pattern = os.path.join(ARTIFACTS_DIR, f"{model_prefix}*.pth")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"[-] ERROR: No model found with prefix '{model_prefix}'")
    
    files.sort() 
    return files[-1]

def load_rbm_model(model_path, visible_units, hidden_units):
    filename = os.path.basename(model_path)
    print(f"[INFO] Instantiating RBM and loading weights from: {filename}")
    
    model = RBM(n_visible=visible_units, n_hidden=hidden_units).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() 
    
    return model


def extract_features(packet):
    if IP in packet and TCP in packet:
        packet_len = len(packet)
        tcp_flags = packet[TCP].flags

        # חילוץ דגלי ה-TCP למספרים בינאריים
        fin = 1 if 'F' in tcp_flags else 0
        syn = 1 if 'S' in tcp_flags else 0
        rst = 1 if 'R' in tcp_flags else 0
        psh = 1 if 'P' in tcp_flags else 0
        ack = 1 if 'A' in tcp_flags else 0
        urg = 1 if 'U' in tcp_flags else 0

        features = {
            'Fwd Header Length': packet[IP].ihl * 4,
            'Bwd Header Length': 0, 
            'Min Packet Length': packet_len,
            'Max Packet Length': packet_len,
            'Packet Length Mean': packet_len,
            'FIN Flag Count': fin,
            'SYN Flag Count': syn,
            'RST Flag Count': rst,
            'PSH Flag Count': psh,
            'ACK Flag Count': ack,
            'URG Flag Count': urg,
            'Average Packet Size': packet_len
        }

        features_df = pd.DataFrame([features])
        scaled_data = packet_scaler.transform(features_df)

        input_tensor = torch.FloatTensor(scaled_data).to(DEVICE)

        with torch.no_grad():
            _, h_sample = packet_model.sample_h(input_tensor)
            v_recon, _ = packet_model.sample_v(h_sample)

            mse = torch.mean((input_tensor - v_recon) ** 2).item()
        
        is_anomaly = mse> PACKET_THRESHOLD
        status = "ANOMALY" if is_anomaly else "NORMAL"

        print(f"[{status}] MSE: {mse:.4f} | Threshold: {PACKET_THRESHOLD} | Packet Len: {packet_len}")


if __name__ == "__main__":
    print("[INFO] Initializing Detection Engine...")

    try:
        packet_model_path = get_latest_model_path("packet_model_master_")
        packet_model = load_rbm_model(packet_model_path, PACKET_VISIBLE_UNITS, PACKET_HIDDEN_UNITS)
        print("[+] Packet RBM model loaded successfully to memory.")

        packet_scaler_path = os.path.join(ARTIFACTS_DIR, "packet_scalar.pkl")
        packet_scaler = joblib.load(packet_scaler_path)
        print(f"[+] Scaler loaded successfully from: {os.path.basename(packet_scaler_path)}")

    except Exception as e:
        print(f"[-] ERROR during initialization: {e}")
        exit(1)

    print("[INFO] Starting real-time sniffer. Capturing TCP packets...")
    sniff(prn=extract_features, store=0)
    print("[INFO] Scaling test completed.")
