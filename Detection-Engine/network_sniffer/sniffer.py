import os
import sys
import glob
import torch
import joblib
import time
import requests
import pandas as pd 
import numpy as np
from scapy.all import sniff, IP, TCP


#Path Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")

sys.path.append(PROJECT_ROOT)
from models.model import RBM

#Packet Hyperparameters Configuration
PACKET_VISIBLE_UNITS = 12
PACKET_HIDDEN_UNITS = 256

#Flow Hyperparameters Configuration
FLOW_VISIBLE_UNITS = 10
FLOW_HIDDEN_UNITS = 256

SIEM_ALERT_URL = "http://localhost:8080/api/logs/alert"
SETTINGS_URL = "http://localhost:8080/api/settings"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Global Variables
packet_model = None
packet_scaler = None
flow_model = None
flow_scaler = None

active_flows = {}

# כאן אפשר לשים את הערכים ההתחלתיים החדשים כברירת מחדל (למקרה שהשרת לא זמין בשניות הראשונות)
DYNAMIC_PACKET_THRESHOLD = 3398.30  
DYNAMIC_FLOW_THRESHOLD = 30715.94
LAST_THRESHOLD_FETCH = 0.0 

def get_dynamic_thresholds():
    global DYNAMIC_PACKET_THRESHOLD, DYNAMIC_FLOW_THRESHOLD, LAST_THRESHOLD_FETCH
    current_time = time.time()
    
    if current_time - LAST_THRESHOLD_FETCH > 10.0:
        try:
            response = requests.get(SETTINGS_URL, timeout=1.5)
            if response.status_code == 200:
                data = response.json()
                
                # מושכים את שניהם בנפרד (עם fallbacks)
                new_packet_thresh = float(data.get('packetThreshold', DYNAMIC_PACKET_THRESHOLD))
                new_flow_thresh = float(data.get('flowThreshold', DYNAMIC_FLOW_THRESHOLD))
                
                # מדפיסים רק אם אחד מהם השתנה
                if new_packet_thresh != DYNAMIC_PACKET_THRESHOLD or new_flow_thresh != DYNAMIC_FLOW_THRESHOLD:
                    print(f"\n[SYSTEM] Thresholds updated from server!")
                    print(f"         Packet: {DYNAMIC_PACKET_THRESHOLD} -> {new_packet_thresh}")
                    print(f"         Flow: {DYNAMIC_FLOW_THRESHOLD} -> {new_flow_thresh}\n")
                    
                DYNAMIC_PACKET_THRESHOLD = new_packet_thresh
                DYNAMIC_FLOW_THRESHOLD = new_flow_thresh
                
            LAST_THRESHOLD_FETCH = current_time
        except Exception as e:
            LAST_THRESHOLD_FETCH = current_time

    return DYNAMIC_PACKET_THRESHOLD, DYNAMIC_FLOW_THRESHOLD


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


def calculate_flow_features(flow_data):
    duration_sec = flow_data['last_time'] - flow_data['start_time']
    duration_micro = duration_sec * 1_000_000

    total_fwd_len = sum(flow_data['fwd_lens'])
    total_bwd_len = sum(flow_data['bwd_lens'])

    bytes_per_sec = (total_fwd_len + total_bwd_len) / duration_sec if duration_sec > 0 else 0
    pkts_per_sec = (flow_data['fwd_pkts'] + flow_data['bwd_pkts']) / duration_sec if duration_sec > 0 else 0

    iats = [] #Inter-Arrival Times
    times = flow_data['packet_times']
    for i in range(1,len(times)):
        iats.append((times[i] - times[i-1]) * 1_000_000) #Convert to microseconds
    
    iat_mean = np.mean(iats) if iats else 0
    iat_max = np.max(iats) if iats else 0
    iat_min = np.min(iats) if iats else 0

    return {
        'Flow Duration': duration_micro,
        'Total Fwd Packets': flow_data['fwd_pkts'],
        'Total Backward Packets': flow_data['bwd_pkts'],
        'Total Length of Fwd Packets': total_fwd_len,
        'Total Length of Bwd Packets': total_bwd_len,
        'Flow Bytes/s': bytes_per_sec,
        'Flow Packets/s': pkts_per_sec,
        'Flow IAT Mean': iat_mean,
        'Flow IAT Max': iat_max,
        'Flow IAT Min': iat_min
    }

PACKET_FEATURES_ORDER = [
    'Fwd Header Length', 'Bwd Header Length', 'Min Packet Length', 
    'Max Packet Length', 'Packet Length Mean', 'FIN Flag Count', 
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 
    'ACK Flag Count', 'URG Flag Count', 'Average Packet Size'
]

def extract_features(packet):
    global active_flows, packet_scaler, packet_model, flow_scaler, flow_model

    curr_pkt_thresh, curr_flow_thresh = get_dynamic_thresholds()

    # בדיקת דיבאג - לוודא שהתקיפה נקלטת
    if IP in packet and packet[IP].dst == "8.8.8.8":
         print(f">>> [DEBUG] Captured attack packet! Flags: {packet[TCP].flags} | Size: {len(packet)}")

    if IP in packet and TCP in packet:
        packet_len = len(packet)
        tcp_flags = packet[TCP].flags
        current_time = float(packet.time)
        
        # 1. בניית הפיצ'רים לפאקט בודד
        raw_packet_features = {
            'Fwd Header Length': packet[IP].ihl * 4,
            'Bwd Header Length': 0, 
            'Min Packet Length': packet_len,
            'Max Packet Length': packet_len,
            'Packet Length Mean': packet_len,
            'FIN Flag Count': 1 if 'F' in tcp_flags else 0,
            'SYN Flag Count': 1 if 'S' in tcp_flags else 0,
            'RST Flag Count': 1 if 'R' in tcp_flags else 0,
            'PSH Flag Count': 1 if 'P' in tcp_flags else 0,
            'ACK Flag Count': 1 if 'A' in tcp_flags else 0,
            'URG Flag Count': 1 if 'U' in tcp_flags else 0,
            'Average Packet Size': packet_len
        }

        # אכיפת סדר עמודות - קריטי להתאמה ל-Scaler
        pkt_df = pd.DataFrame([raw_packet_features])[PACKET_FEATURES_ORDER]
        
        # Scaling + Clipping (שים לב לטווח המתוקן)
        pkt_scaled = packet_scaler.transform(pkt_df)
        pkt_scaled = np.clip(pkt_scaled, -10, 10)
        
        pkt_tensor = torch.FloatTensor(pkt_scaled).to(DEVICE)
        
        with torch.no_grad():
            # --- התיקון הקריטי: הורדנו את ה-1- ---
            # אנרגיה גבוהה = אנומליה. פשוט וברור.
            energy_packet = packet_model.free_energy(pkt_tensor).item()
            
        is_packet_anomaly = energy_packet > curr_pkt_thresh

        # 2. ניהול זרימות (Flow Management)
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        
        flow_key = tuple(sorted([src_ip, dst_ip]) + sorted([src_port, dst_port]))
        
        if flow_key not in active_flows:
            active_flows[flow_key] = {
                'fwd_ip': src_ip, 
                'start_time': current_time,
                'last_time': current_time,
                'fwd_pkts': 0, 'bwd_pkts': 0,
                'fwd_lens': [], 'bwd_lens': [],
                'packet_times': []
            }
            
        flow = active_flows[flow_key]
        flow['last_time'] = current_time
        flow['packet_times'].append(current_time)
        
        if src_ip == flow['fwd_ip']:
            flow['fwd_pkts'] += 1
            flow['fwd_lens'].append(packet_len)
        else:
            flow['bwd_pkts'] += 1
            flow['bwd_lens'].append(packet_len)
            
        total_packets_in_flow = flow['fwd_pkts'] + flow['bwd_pkts']
        
        is_flow_anomaly = False
        energy_flow = 0.0
        
        # 3. בדיקת אנומליה בזרימה (Flow Model)
        if total_packets_in_flow >= 3:
            flow_features = calculate_flow_features(flow)
            flow_df = pd.DataFrame([flow_features]) # וודא שגם כאן יש FLOW_FEATURES_ORDER
            
            flow_scaled = flow_scaler.transform(flow_df)
            flow_scaled = np.clip(flow_scaled, -10, 10)
            flow_tensor = torch.FloatTensor(flow_scaled).to(DEVICE)
            
            with torch.no_grad():
                # --- גם כאן: בלי מינוס 1 ---
                energy_flow = flow_model.free_energy(flow_tensor).item()
                
            is_flow_anomaly = energy_flow > curr_flow_thresh

        # 4. קביעת סטטוס משולב
        if is_packet_anomaly and is_flow_anomaly:
            status = "ANOMALY: BOTH"
        elif is_packet_anomaly:
            status = "ANOMALY: PACKET"
        elif is_flow_anomaly:
            status = "ANOMALY: FLOW"
        else:
            status = "NORMAL"
            
        # הדפסה לטרמינל
        print(f"[{status}] Pkt: {energy_packet:.2f} | Flow: {energy_flow:.2f} | Size: {packet_len} | FlowPkts: {total_packets_in_flow}")

        # 5. שליחת התראה ל-SIEM (Spring Boot)
        if status != "NORMAL":
            max_score = max(energy_packet, energy_flow)
            alert_payload = {
                "sourceIp": str(src_ip),
                "destinationIp": str(dst_ip),
                "detectedBy": f"NIDS Engine ({status})",
                "anomalyScore": float(max_score),
                "description": f"Anomaly detected. Pkt Score: {energy_packet:.2f}, Flow Score: {energy_flow:.2f}"
            }

            try:
                response = requests.post(SIEM_ALERT_URL, json=alert_payload, timeout=1)

                if response.status_code == 200:
                    print("    -> [SIEM] Alert successfully delivered to Central Server!")
                else:
                    print(f"    -> [WARNING] SIEM received the alert but returned status: {response.status_code}")
                        
            except requests.exceptions.RequestException as e:
                print(f"    -> [ERROR] Failed to send alert to SIEM! Is Spring Boot running? Error: {e}")


# Execution
if __name__ == "__main__":
    print("[INFO] Initializing Dual-Engine Detection System (Packet + Flow)...")

    from scapy.all import get_if_list, conf
    print("\n--- Network Interfaces Available ---")
    print(get_if_list())
    # אם אתה יודע את שם הכרטיס (למשל "Wi-Fi"), תוכל להגדיר אותו ב-conf.iface
    print(f"--- Currently Listening on: {conf.iface} ---\n")

    try:
        # Load Packet Model & Scaler
        packet_model_path = get_latest_model_path("packet_model_master_")
        packet_model = load_rbm_model(packet_model_path, PACKET_VISIBLE_UNITS, PACKET_HIDDEN_UNITS)
        packet_scaler_path = os.path.join(ARTIFACTS_DIR, "packet_scalar.pkl")
        packet_scaler = joblib.load(packet_scaler_path)
        print("[+] Packet RBM & Scaler loaded successfully.")

        # Load Flow Model & Scaler
        flow_model_path = get_latest_model_path("flow_model_master_")
        flow_model = load_rbm_model(flow_model_path, FLOW_VISIBLE_UNITS, FLOW_HIDDEN_UNITS)
        flow_scaler_path = os.path.join(ARTIFACTS_DIR, "flow_scalar.pkl")
        flow_scaler = joblib.load(flow_scaler_path)
        print("[+] Flow RBM & Scaler loaded successfully.\n")

    except Exception as e:
        print(f"[-] ERROR during initialization: {e}")
        exit(1)

    print("[INFO] Engine Online. Waiting for traffic... (Press Ctrl+C to stop)")
    sniff(prn=extract_features, store=0)