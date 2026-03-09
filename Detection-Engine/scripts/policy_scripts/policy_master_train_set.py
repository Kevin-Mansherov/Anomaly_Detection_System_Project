import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

current_dir = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '../..'))

BASE_DIR = os.path.join(PROJECT_ROOT, 'Datasets/Model_3/r1/r1/') 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data/processed/policy')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'models/artifacts')
FILES_TO_MERGE = ['logon.csv', 'device.csv', 'http.csv']

def build_policy_master_set():
    all_events = []
    print("--- Starting Master Policy Preprocessing (Model 3) ---")

    for filename in FILES_TO_MERGE:
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path):
            print(f"[WARNING] File {filename} not found in {BASE_DIR}. Skipping.")
            continue

        print(f"[INFO] Reading {filename}...")
        df = pd.read_csv(file_path)

        if filename == 'http.csv':
            df['act_name'] = 'http'
        elif filename in ['logon.csv', 'device.csv']:
            df['act_name'] = df['activity']

        all_events.append(df[['date', 'act_name']])
    
    combined_df = pd.concat(all_events, ignore_index=True)

    print(f"[INFO] Encoding Cyclic Time Features...")
    combined_df['date'] = pd.to_datetime(combined_df['date'])

    print(f"[INFO] Filtering training data to include only normal hours (>= 5:00)...")
    combined_df = combined_df[combined_df['date'].dt.hour >= 5].reset_index(drop=True)

    combined_df['hour'] = combined_df['date'].dt.hour

    combined_df = pd.get_dummies(combined_df, columns=['act_name'])

    for h in range(5,24):
        col = f'hour_{h}'
        if col not in combined_df.columns:
            combined_df[col] = 0


    hours_cols = sorted([col for col in combined_df.columns if 'hour_' in col])
    action_cols = sorted([col for col in combined_df.columns if 'act_name_' in col])
    feature_cols = action_cols + hours_cols

    features = combined_df[feature_cols].astype(float)

    print("[INFO] Applying MinMaxScaler & Clipping...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    scaled_features = np.clip(scaled_features, 0, 1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, 'policy_train_master.npy'), scaled_features)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'policy_scaler.pkl'))
    
    print(f"\n[SUCCESS] Master Policy Set Created!")
    print(f"Matrix shape: {scaled_features.shape}")
    print(f"Features used: {feature_cols}")

if __name__ == "__main__":
    build_policy_master_set()
