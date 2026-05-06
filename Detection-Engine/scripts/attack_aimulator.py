import time
from scapy.all import IP, TCP, send

TARGET_IP = "8.8.8.8" 

def trigger_combined_anomaly():
    print(f"[*] Launching Massive Flow & Packet Anomaly to {TARGET_IP}...")
    
    # 1. שימוש בפורט מקבוע (1337) כדי שהכל ייכנס לאותו Flow!
    # 2. הוספת 2000 בתים של "זבל" כדי לשגע את מודל הפאקטים
    # 3. דגלים לא חוקיים (SYN + FIN ביחד)
    
    malformed_burst = [
        IP(dst=TARGET_IP) / TCP(sport=1337, dport=80, flags="SF") / (b"X" * 2000) 
        for _ in range(100)
    ]
    
    # שולחים את הכל במכה אחת
    send(malformed_burst, verbose=False)
    print("[+] Burst sent! Watch the Sniffer console.")

if __name__ == "__main__":
    print("=== NIDS Final Attack Simulator ===")
    trigger_combined_anomaly()