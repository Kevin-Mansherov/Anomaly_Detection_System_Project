import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

# נתיבים
DATASETS_DIR = '../Datasets/Model_1_and_2/MachineLearningCSV/MachineLearningCVE/'
OUTPUT_DIR = '../data/processed/network/'
ARTIFACTS_DIR = '../models/artifacts/'

# רשימת העמודות להסרה (כפי שהחלטנו)
DROP_COLS = [
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp'
]

def process_all_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scaler = None
    
    # שלב א': אימון ה-Scaler על יום שני (הנורמלי)
    monday_file = os.path.join(DATASETS_DIR, 'Monday-WorkingHours.pcap_ISCX.csv')
    print(f"Fitting scaler on: {monday_file}")
    df_monday = pd.read_csv(monday_file)
    df_monday.columns = df_monday.columns.str.strip()
    df_monday.drop(columns=[c for c in DROP_COLS + ['Label'] if c in df_monday.columns], inplace=True)
    df_monday.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_monday.fillna(0, inplace=True)
    
    scaler = MinMaxScaler()
    scaler.fit(df_monday)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'network_scaler.pkl'))
    
    # שלב ב': עיבוד כל הקבצים (כולל ימי ההתקפות)
    for filename in os.listdir(DATASETS_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATASETS_DIR, filename)
            print(f"Processing: {filename}...")
            
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            # שמירת התוויות (Labels) בצד כדי שנדע בבדיקות אם צדקנו
            labels = df['Label'].values 
            
            # ניקוי
            df.drop(columns=[c for c in DROP_COLS + ['Label'] if c in df.columns], inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            
            # נרמול עם ה-Scaler המקורי
            scaled_data = scaler.transform(df)
            
            # שמירה
            base_name = filename.replace('.csv', '')
            np.save(os.path.join(OUTPUT_DIR, f'{base_name}_data.npy'), scaled_data)
            np.save(os.path.join(OUTPUT_DIR, f'{base_name}_labels.npy'), labels)

    print("Done! All network datasets are ready.")

if __name__ == "__main__":
    process_all_files()