import pandas as pd
import numpy as np
import os
import joblib
import glob
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# הנתיבים שלך
DATASET_DIR = '../Datasets/Model_1_and_2/MachineLearningCSV/MachineLearningCVE'
OUTPUT_DIR = '../data/processed/network'
MODEL_DIR = '../models/artifacts'

FLOW_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min'
]

def build_master_train_set():
    print("[INFO] Searching for all CSV files in dataset directory...")
    # מוצא את כל קבצי ה-CSV בתיקייה
    all_csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    
    benign_dataframes = []
    
    for file in all_csv_files:
        filename = os.path.basename(file)
        print(f"[INFO] Processing {filename}...")
        
        # קריאת הקובץ
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip() # ניקוי רווחים משמות העמודות
        
        # סינון: לוקחים רק תעבורה נורמלית
        benign_df = df[df['Label'] == 'BENIGN'].copy()
        
        # חילוץ רק העמודות שמעניינות אותנו
        features_df = benign_df[[c for c in FLOW_FEATURES if c in benign_df.columns]].copy()
        
        # טיפול בערכים חסרים או בעייתיים
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(0, inplace=True)
        
        benign_dataframes.append(features_df)
    
    # איחוד כל הימים לדאטה-פריים אחד ענק
    print("\n[INFO] Combining all days into one massive dataset...")
    master_df = pd.concat(benign_dataframes, ignore_index=True)
    
    print(f"[INFO] Total normal traffic samples collected: {len(master_df)}")

    print("[INFO] Scaling the data with RobustScaler...")
    scalar = RobustScaler()
    scaled_train_data = scalar.fit_transform(master_df)

    scaled_train_data = np.clip(scaled_train_data, -10, 10)
    
    # שמירת הקבצים
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, 'flow_train_master.npy'), scaled_train_data)
    joblib.dump(scalar, os.path.join(MODEL_DIR, 'flow_scalar.pkl'))
    
    print(f"[SUCCESS] Master training set saved! Shape: {scaled_train_data.shape}")
    print(f"[SUCCESS] New Scaler saved to {MODEL_DIR}/flow_scalar.pkl")

if __name__ == "__main__":
    build_master_train_set()