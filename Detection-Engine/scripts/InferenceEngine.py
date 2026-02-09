import torch
import numpy as np
import joblib
import requests
import time 
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import RBM
from scapy.all  import sniff

JAVA_SERVER_URL = "http://localhost:8080/api/alerts"
ARTIFACTS_DIR = os.path.join("..", "models", "artifacts")

class AnomalyDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.registry = {
            'flow': {
                'model_file': 'flow_model_20260206-182929.pth',
                'scaler_file': 'flow_scalar.pkl',
                'threshold': 0.0503,
                'dims': (10, 128)
            },
            'packet': {
                'model_file': 'best_packet_model.pth',
                'scaler_file': 'packet_scalar.pkl',
                'threshold': 0.0400,
                'dims': (12, 256)
            },
            'policy': {
              'model_file': 'policy_model_20260206-180007.pth',
                'scaler_file': 'policy_scaler.pkl',
                'threshold': 0.0810,
                'dims': (7, 128)
            }
        }

        self.models = {}
        self.scalers = {}
        self._load_resources()

    def _load_resources(self):
        """Loading weights and scalers into memory."""
        for name, cfg in self.registry.items():
            # Loading the Scaler (Min-Max Scaling).
            self.scalers[name] = joblib.load(os.path.join(ARTIFACTS_DIR, cfg['scaler_file']))
            
            # RBM initialization and .pth file loading.
            vis, hid = cfg['dims']
            model = RBM(vis, hid).to(self.device)
            model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, cfg['model_file']), map_location=self.device))
            model.eval()
            self.models[name] = model
            print(f"[INFO] {name.upper()} model is ready for inference.")

    def process_and_analyze(self, model_name, raw_features):
        """Processing phase: Normalization and reconstruction error calculation [cite: 183, 311]"""
        # Normalizing the data.
        scaled = self.scalers[model_name].transform([raw_features])
        tensor_in = torch.FloatTensor(scaled).to(self.device)
        
        # Calculating Reconstruction Error (Mean Squared Error).
        with torch.no_grad():
            v_pos, v_neg = self.models[model_name](tensor_in)
            mse = torch.mean((v_pos - v_neg) ** 2).item() # Calculating the gap.
            
        return mse

    def send_alert(self, model_name, mse):
        """Sending JSON alert to the Java server."""
        threshold = self.registry[model_name]['threshold']
        
        # Decision Logic.
        if mse > threshold:
            payload = {
                "modelName": model_name,
                "severity": "CRITICAL" if mse > threshold * 2 else "WARNING", # Severity determination.
                "mseScore": round(mse, 6),
                "threshold": threshold,
                "description": f"Statistical anomaly detected by {model_name} model."
            }
            
            try:
                requests.post(JAVA_SERVER_URL, json=payload)
                print(f"[!] ALERT: {model_name} MSE={mse:.4f} > Threshold.")
            except Exception as e:
                print(f"[X] Communication Error: {e}")

# Network Sniffing Component (Scapy)
def start_sniffing(detector):
    print("--- Starting Scapy Listener (Promiscuous Mode) ---")
    
    def packet_callback(packet):
        # Feature extraction will occur here. 
        # For example: packet size, TCP flags, addresses.
        pass # In the next stage, time-window aggregation will be implemented

    # Listening to live traffic.
    # sniff(prn=packet_callback, store=0) 

if __name__ == "__main__":
    detector = AnomalyDetector()
    # start_sniffing(detector)