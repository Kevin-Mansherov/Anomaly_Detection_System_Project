import pandas as pd
import numpy as np
import os
import joblib
import glob
from sklearn.preprocessing import RobustScaler

current_dir = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '../..'))

# נתיבים (לפי המבנה שלך)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'Datasets/Model_1_and_2/MachineLearningCSV/MachineLearningCVE')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data/processed/network')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models/artifacts')

# המאפיינים של מודל הפאקט ששלחת
PACKET_FEATURES = [
    'Fwd Header Length', 'Bwd Header Length', 'Min Packet Length', 
    'Max Packet Length', 'Packet Length Mean', 'FIN Flag Count', 
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 
    'ACK Flag Count', 'URG Flag Count', 'Average Packet Size'
]

def build_packet_master_set():
    print("[INFO] Starting Master Training Set creation for Model 2 (Packet)...")
    all_csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    
    benign_list = []
    
    for file in all_csv_files:
        filename = os.path.basename(file)
        print(f"[INFO] Processing {filename}...")
        
        # טעינה וניקוי בסיסי
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        
        # סינון רק תעבורה תקינה (BENIGN) ולקיחת המאפיינים הרלוונטיים
        if 'Label' in df.columns:
            benign_df = df[df['Label'] == 'BENIGN'].copy()
            # מוודא שכל הפיצ'רים קיימים בקובץ
            existing_features = [c for c in PACKET_FEATURES if c in benign_df.columns]
            features_df = benign_df[existing_features].copy()
            
            # טיפול בערכים בעייתיים (NaN/Inf)
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            features_df.fillna(0, inplace=True)
            
            benign_list.append(features_df)
    
    # איחוד כל הימים
    print("\n[INFO] Concatenating all benign packet data...")
    master_df = pd.concat(benign_list, ignore_index=True)
    print(f"[INFO] Total packet samples: {len(master_df)}")
    
    # נרמול חזק (Robust Scaling)
    print("[INFO] Applying RobustScaler...")
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(master_df)
    
    # הגבלת ערכים (Clipping) למניעת NaN ב-GPU בזמן אימון ה-RBM
    print("[INFO] Clipping values to range [-10, 10] for stability...")
    scaled_data = np.clip(scaled_data, -10, 10)
    
    # שמירה
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, 'packet_train_master.npy'), scaled_data)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'packet_scalar.pkl'))
    
    print(f"\n[SUCCESS] Master Packet Train set created!")
    print(f"Shape: {scaled_data.shape}")
    print(f"Artifacts: packet_train_master.npy, packet_scalar.pkl")

if __name__ == "__main__":
    build_packet_master_set()