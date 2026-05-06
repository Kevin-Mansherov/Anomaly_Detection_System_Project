import os 
import time
import json
import requests
import pandas as pd


#Configuration
SIEM_SERVER_URL = "http://localhost:8080/api/logs"
LOG_DIR_PATH = "../Datasets/Model_3/r1/r1/"
FILES_TO_STREAM = ['logon.csv', 'device.csv', 'http.csv']

DELAY_SECONDS = 0.5

#Logic 
def stream_logs():
    print("==================================================")
    print("[INFO] Starting UBA Log Agent (User Behavior Analytics)")
    print(f"[INFO] Target SIEM: {SIEM_SERVER_URL}")
    print("==================================================\n")

    all_events = []

    try:
        for filename in FILES_TO_STREAM:
            file_path = os.path.join(LOG_DIR_PATH, filename)
            if os.path.exists(file_path):
                print(f"[INFO] Reading {filename}...")
                df = pd.read_csv(file_path)

                if filename == 'http.csv':
                    df['activity'] = 'http'
                
                df['source_file'] = filename
                all_events.append(df)
            
            else:
                print(f"[WARNING] File not found: {file_path}")
        
        if not all_events:
            print("[-] FATAL ERROR: No log files found in the directory.")
            return

        print("\n[INFO] Combining files and sorting chronologically...")
        master_df = pd.concat(all_events, ignore_index=True)

        master_df['date'] = pd.to_datetime(master_df['date'])
        master_df.sort_values('date', inplace=True)

        print(f"[INFO] Successfully loaded and sorted {len(master_df)} events. Starting real time transmission...\n")

        for index, row in master_df.iterrows():
            timestamp_str = row['date'].strftime('%m/%d/%Y %H:%M:%S')

            log_event = {
                "event_id": row.get('id', f"evt_{index}"),
                "user": row.get('user', 'Unknown'),
                "timestamp": timestamp_str,
                "activity": row.get('activity', 'Unknown'),
                "pc": row.get('pc', 'Unknown'),
                "source": row.get('source_file', 'Unknown')
            }

            payload_str = json.dumps(log_event, indent=2)
            print(f"[*] Capturing New Event:\n{payload_str}")

            try:
                response = requests.post(SIEM_SERVER_URL, json=log_event, timeout=2)
                print(f"    -> [SUCCESS] Delivered to SIEM. HTTP Status: {response.status_code}")
                
                
            except requests.exceptions.RequestException as e:
                print(f"    -> [ERROR] Failed to connect to SIEM Server: {e}")
            
            print("-" * 50)
            
            time.sleep(DELAY_SECONDS)

    except Exception as e:
        print(f"[-] UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    try:
        stream_logs()
    except KeyboardInterrupt:
        print("\n[INFO] Log Agent stopped by user. Shutting down gracefully.")  