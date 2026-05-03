import os
import sys
import glob
import torch
import joblib
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
PACKET_THRESHOLD = 3.369822

#Flow Hyperparameters Configuration
FLOW_VISIBLE_UNITS = 10
FLOW_HIDDEN_UNITS = 256
FLOW_THRESHOLD = 14.386914

SIEM_ALERT_URL = "http://localhost:8080/api/logs/alert"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Global Variables
packet_model = None
packet_scaler = None
flow_model = None
flow_scaler = None

active_flows = {}

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



def extract_features(packet):
    global active_flows, packet_scaler, packet_model, flow_scaler, flow_model

    if IP in packet and TCP in packet:
        packet_len = len(packet)
        tcp_flags = packet[TCP].flags
        current_time = float(packet.time)
        
        packet_features = {
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

        # Inference for Packet Model
        pkt_df = pd.DataFrame([packet_features])
        pkt_scaled = packet_scaler.transform(pkt_df)
        pkt_tensor = torch.FloatTensor(pkt_scaled).to(DEVICE)
        with torch.no_grad():
            _, h_sample = packet_model.sample_h(pkt_tensor)
            v_recon, _ = packet_model.sample_v(h_sample)
            mse_packet = torch.mean((pkt_tensor - v_recon) ** 2).item()
            
        is_packet_anomaly = mse_packet > PACKET_THRESHOLD

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
        mse_flow = 0.0
        
        if total_packets_in_flow >= 3:
            flow_features = calculate_flow_features(flow)
            flow_df = pd.DataFrame([flow_features])
            
            flow_scaled = flow_scaler.transform(flow_df)
            flow_tensor = torch.FloatTensor(flow_scaled).to(DEVICE)
            
            with torch.no_grad():
                _, h_sample_f = flow_model.sample_h(flow_tensor)
                v_recon_f, _ = flow_model.sample_v(h_sample_f)
                mse_flow = torch.mean((flow_tensor - v_recon_f) ** 2).item()
                
            is_flow_anomaly = mse_flow > FLOW_THRESHOLD

        if is_packet_anomaly or is_flow_anomaly:
            status = "ANOMALY: BOTH"
        elif is_packet_anomaly:
            status = "ANOMALY: PACKET"
        elif is_flow_anomaly:
            status = "ANOMALY: FLOW"
        else:
            status = "NORMAL"
            
        if total_packets_in_flow >= 3:
            print(f"[{status}] Packet MSE: {mse_packet:.4f} | Flow MSE: {mse_flow:.4f} | PktLen: {packet_len} | FlowPkts: {total_packets_in_flow}")

        if status != "NORMAL":
            max_mse = max(mse_packet, mse_flow)

            alert_payload = {
                "sourceIp": str(src_ip),
                "destinationIp": str(dst_ip),
                "detectedBy": f"NIDS Engine ({status})",
                "anomalyScore": float(max_mse),
                "description": f"Network anomaly detected. Packet MSE: {mse_packet:.4f}, Flow MSE: {mse_flow:.4f}"
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


# import os
# import sys
# import glob
# import torch
# import joblib 
# import pandas as pd 
# import numpy as np
# from datetime import datetime
# from scapy.all import sniff, IP, TCP

# # ==========================================
# # 1. Path Configuration
# # ==========================================
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")

# sys.path.append(PROJECT_ROOT)
# from models.model import RBM

# # ==========================================
# # 2. Hyperparameters Configuration
# # ==========================================
# # Packet Model Settings
# PACKET_VISIBLE_UNITS = 12
# PACKET_HIDDEN_UNITS = 256
# PACKET_THRESHOLD = 3.369822

# # Flow Model Settings
# FLOW_VISIBLE_UNITS = 10
# FLOW_HIDDEN_UNITS = 256
# FLOW_THRESHOLD = 14.386914

# # Policy Model Settings (Model 3)
# POLICY_VISIBLE_UNITS = 24
# POLICY_HIDDEN_UNITS = 256
# POLICY_THRESHOLD = 0.500000  # <<< ⚠️ שים פה את ה-Threshold האמיתי שלך מהקולאב ⚠️

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Global Variables
# packet_model = None
# packet_scaler = None
# flow_model = None
# flow_scaler = None
# policy_model = None
# policy_scaler = None

# # מילון ניהול זרימות
# active_flows = {}

# # יצירת רשימת 24 הפיצ'רים של מודל ה-Policy בסדר לקסיקוגרפי מוחלט (בדיוק כמו באימון)
# POLICY_HOURS_COLS = sorted([f'hour_{h}' for h in range(5, 24)])
# POLICY_ACTION_COLS = sorted(['act_name_Connect', 'act_name_Disconnect', 'act_name_Logoff', 'act_name_Logon', 'act_name_http'])
# POLICY_FEATURES = POLICY_HOURS_COLS + POLICY_ACTION_COLS

# # ==========================================
# # 3. Helper Functions
# # ==========================================
# def get_latest_model_path(model_prefix):
#     pattern = os.path.join(ARTIFACTS_DIR, f"{model_prefix}*.pth")
#     files = glob.glob(pattern)
#     if not files:
#         raise FileNotFoundError(f"[-] ERROR: No model found with prefix '{model_prefix}'")
#     files.sort() 
#     return files[-1]

# def load_rbm_model(model_path, visible_units, hidden_units):
#     filename = os.path.basename(model_path)
#     print(f"[INFO] Instantiating RBM and loading weights from: {filename}")
#     model = RBM(n_visible=visible_units, n_hidden=hidden_units).to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval() 
#     return model

# def calculate_flow_features(flow_data):
#     """מחשב את 10 הפיצ'רים של Flow בהתבסס על המנות שנאגרו"""
#     duration_sec = flow_data['last_time'] - flow_data['start_time']
#     duration_micro = duration_sec * 1_000_000
    
#     total_fwd_len = sum(flow_data['fwd_lens'])
#     total_bwd_len = sum(flow_data['bwd_lens'])
    
#     bytes_per_sec = (total_fwd_len + total_bwd_len) / duration_sec if duration_sec > 0 else 0
#     pkts_per_sec = (flow_data['fwd_pkts'] + flow_data['bwd_pkts']) / duration_sec if duration_sec > 0 else 0
    
#     iats = []
#     times = flow_data['packet_times']
#     for i in range(1, len(times)):
#         iats.append((times[i] - times[i-1]) * 1_000_000)
        
#     iat_mean = np.mean(iats) if iats else 0
#     iat_max = np.max(iats) if iats else 0
#     iat_min = np.min(iats) if iats else 0
    
#     return {
#         'Flow Duration': duration_micro,
#         'Total Fwd Packets': flow_data['fwd_pkts'],
#         'Total Backward Packets': flow_data['bwd_pkts'],
#         'Total Length of Fwd Packets': total_fwd_len,
#         'Total Length of Bwd Packets': total_bwd_len,
#         'Flow Bytes/s': bytes_per_sec,
#         'Flow Packets/s': pkts_per_sec,
#         'Flow IAT Mean': iat_mean,
#         'Flow IAT Max': iat_max,
#         'Flow IAT Min': iat_min
#     }

# # ==========================================
# # 4. Main Sniffer Logic
# # ==========================================
# def extract_features(packet):
#     global active_flows, packet_scaler, packet_model, flow_scaler, flow_model, policy_scaler, policy_model

#     if IP in packet and TCP in packet:
#         packet_len = len(packet)
#         tcp_flags = packet[TCP].flags
#         current_time = float(packet.time)
        
#         # --- מנוע 1: Packet Model ---
#         packet_features = {
#             'Fwd Header Length': packet[IP].ihl * 4,
#             'Bwd Header Length': 0, 
#             'Min Packet Length': packet_len,
#             'Max Packet Length': packet_len,
#             'Packet Length Mean': packet_len,
#             'FIN Flag Count': 1 if 'F' in tcp_flags else 0,
#             'SYN Flag Count': 1 if 'S' in tcp_flags else 0,
#             'RST Flag Count': 1 if 'R' in tcp_flags else 0,
#             'PSH Flag Count': 1 if 'P' in tcp_flags else 0,
#             'ACK Flag Count': 1 if 'A' in tcp_flags else 0,
#             'URG Flag Count': 1 if 'U' in tcp_flags else 0,
#             'Average Packet Size': packet_len
#         }

#         pkt_df = pd.DataFrame([packet_features])
#         pkt_scaled = packet_scaler.transform(pkt_df)
#         pkt_tensor = torch.FloatTensor(pkt_scaled).to(DEVICE)
#         with torch.no_grad():
#             _, h_sample = packet_model.sample_h(pkt_tensor)
#             v_recon, _ = packet_model.sample_v(h_sample)
#             mse_packet = torch.mean((pkt_tensor - v_recon) ** 2).item()
            
#         is_packet_anomaly = mse_packet > PACKET_THRESHOLD

#         # --- מנוע 3: Policy Model (לכידת תעבורה יוצאת) ---
#         is_policy_anomaly = False
#         mse_policy = 0.0
#         dst_port = packet[TCP].dport
        
#         # אנחנו מתשאלים את ה-Policy רק אם זו גלישה החוצה (HTTP/HTTPS)
#         if dst_port in [80, 443]:
#             # חילוץ השעה שבה הפאקט נשלח
#             pkt_hour = datetime.fromtimestamp(current_time).hour
            
#             # אתחול וקטור 24 הפיצ'רים ב-0
#             policy_data = {col: 0.0 for col in POLICY_FEATURES}
            
#             # הדלקת (1.0) עמודת השעה הרלוונטית (אם השעה היא בין 5 ל-23)
#             hour_col = f'hour_{pkt_hour}'
#             if hour_col in policy_data:
#                 policy_data[hour_col] = 1.0
                
#             # הדלקת פעולת ה-HTTP
#             policy_data['act_name_http'] = 1.0
            
#             pol_df = pd.DataFrame([policy_data])[POLICY_FEATURES] # שמירה על סדר עמודות מדויק
#             pol_scaled = policy_scaler.transform(pol_df)
#             pol_tensor = torch.FloatTensor(pol_scaled).to(DEVICE)
            
#             with torch.no_grad():
#                 _, h_sample_p = policy_model.sample_h(pol_tensor)
#                 v_recon_p, _ = policy_model.sample_v(h_sample_p)
#                 mse_policy = torch.mean((pol_tensor - v_recon_p) ** 2).item()
                
#             is_policy_anomaly = mse_policy > POLICY_THRESHOLD

#         # --- מנוע 2: Flow Model ---
#         src_ip = packet[IP].src
#         dst_ip = packet[IP].dst
#         src_port = packet[TCP].sport
        
#         flow_key = tuple(sorted([src_ip, dst_ip]) + sorted([src_port, dst_port]))
        
#         if flow_key not in active_flows:
#             active_flows[flow_key] = {
#                 'fwd_ip': src_ip, 
#                 'start_time': current_time,
#                 'last_time': current_time,
#                 'fwd_pkts': 0, 'bwd_pkts': 0,
#                 'fwd_lens': [], 'bwd_lens': [],
#                 'packet_times': []
#             }
            
#         flow = active_flows[flow_key]
#         flow['last_time'] = current_time
#         flow['packet_times'].append(current_time)
        
#         if src_ip == flow['fwd_ip']:
#             flow['fwd_pkts'] += 1
#             flow['fwd_lens'].append(packet_len)
#         else:
#             flow['bwd_pkts'] += 1
#             flow['bwd_lens'].append(packet_len)
            
#         total_packets_in_flow = flow['fwd_pkts'] + flow['bwd_pkts']
        
#         is_flow_anomaly = False
#         mse_flow = 0.0
        
#         if total_packets_in_flow >= 3:
#             flow_features = calculate_flow_features(flow)
#             flow_df = pd.DataFrame([flow_features])
            
#             flow_scaled = flow_scaler.transform(flow_df)
#             flow_tensor = torch.FloatTensor(flow_scaled).to(DEVICE)
            
#             with torch.no_grad():
#                 _, h_sample_f = flow_model.sample_h(flow_tensor)
#                 v_recon_f, _ = flow_model.sample_v(h_sample_f)
#                 mse_flow = torch.mean((flow_tensor - v_recon_f) ** 2).item()
                
#             is_flow_anomaly = mse_flow > FLOW_THRESHOLD

#         # --- מנגנון ה-Ensemble הסופי (הצבעת 3 מנועים) ---
#         if is_packet_anomaly or is_flow_anomaly or is_policy_anomaly:
#             anomalies = []
#             if is_packet_anomaly: anomalies.append("PACKT")
#             if is_flow_anomaly: anomalies.append("FLOW")
#             if is_policy_anomaly: anomalies.append("POLICY")
            
#             status = f"🚨 ANOMALY: {'+'.join(anomalies)}"
#         else:
#             status = "✅ NORMAL"
            
#         # הדפסה רק כשהשיחה פעילה מעל 3 מנות כדי למנוע הצפה של המסך
#         if total_packets_in_flow >= 3:
#             # אם זו לא תעבורת אינטרנט, הפוליסי לא מצביע (נציג N/A)
#             pol_print = f"{mse_policy:.4f}" if dst_port in [80, 443] else "N/A   "
#             print(f"[{status.ljust(25)}] Pkt MSE: {mse_packet:.4f} | Flow MSE: {mse_flow:.4f} | Pol MSE: {pol_print} | PktLen: {packet_len}")


# # ==========================================
# # 5. Initialization and Execution
# # ==========================================
# if __name__ == "__main__":
#     print("[INFO] Initializing TRIPLE-ENGINE Detection System (Packet + Flow + Policy)...")

#     try:
#         # Load Packet Model & Scaler
#         packet_model_path = get_latest_model_path("packet_model_master_")
#         packet_model = load_rbm_model(packet_model_path, PACKET_VISIBLE_UNITS, PACKET_HIDDEN_UNITS)
#         packet_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "packet_scalar.pkl"))
#         print("[+] Packet Engine Loaded.")

#         # Load Flow Model & Scaler
#         flow_model_path = get_latest_model_path("flow_model_master_")
#         flow_model = load_rbm_model(flow_model_path, FLOW_VISIBLE_UNITS, FLOW_HIDDEN_UNITS)
#         flow_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "flow_scalar.pkl"))
#         print("[+] Flow Engine Loaded.")

#         # Load Policy Model & Scaler
#         policy_model_path = get_latest_model_path("policy_model_master_")
#         policy_model = load_rbm_model(policy_model_path, POLICY_VISIBLE_UNITS, POLICY_HIDDEN_UNITS)
#         policy_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "policy_scaler.pkl")) # שים לב לשם הקובץ שיש לך!
#         print("[+] Policy Engine Loaded.\n")

#     except Exception as e:
#         print(f"[-] ERROR during initialization: {e}")
#         exit(1)

#     print("[INFO] Triple-Engine Online. Waiting for traffic... (Press Ctrl+C to stop)")
#     sniff(prn=extract_features, store=0)