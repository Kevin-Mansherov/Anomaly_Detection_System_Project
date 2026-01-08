"""
PURPOSE:
Prepares user activity data for Model 3 (Policy Enforcement).
This script merges logs and performs advanced feature engineering, including 
Cyclic Time Encoding (Sine/Cosine) for hours to help the RBM understand 
temporal relationships.
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Path definitions
BASE_DIR = '../Datasets/Model_3/r1/r1/' 
OUTPUT_DIR = '../data/processed/policy'
ARTIFACTS_DIR = '../models/artifacts'
FILES_TO_MERGE = ['logon.csv', 'device.csv', 'http.csv']

def preprocess_unified_cert():
    all_events = []
    print("--- Starting Unified CERT Preprocessing ---")

    for filename in FILES_TO_MERGE:
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        df['event_source'] = filename.split('.')[0]
        all_events.append(df[['date', 'user', 'pc', 'activity', 'event_source']])

    # 1. Combine events
    combined_df = pd.concat(all_events, ignore_index=True)
    
    # 2. Time Handling & Cyclic Encoding
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['hour'] = combined_df['date'].dt.hour
    
    # Create sine and cosine features for the 24-hour cycle
    combined_df['hour_sin'] = np.sin(2 * np.pi * combined_df['hour'] / 24)
    combined_df['hour_cos'] = np.cos(2 * np.pi * combined_df['hour'] / 24)
    
    combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
    
    # 3. Categorical Encoding
    le_user = LabelEncoder()
    combined_df['user_id'] = le_user.fit_transform(combined_df['user'])
    le_pc = LabelEncoder()
    combined_df['pc_id'] = le_pc.fit_transform(combined_df['pc'])
    le_act = LabelEncoder()
    combined_df['activity_id'] = le_act.fit_transform(combined_df['activity'].astype(str))
    le_src = LabelEncoder()
    combined_df['source_id'] = le_src.fit_transform(combined_df['event_source'])

    # 4. Final Feature Vector (Including Cyclic Time)
    features = combined_df[['hour_sin', 'hour_cos', 'day_of_week', 'user_id', 'pc_id', 'activity_id', 'source_id']]
    
    # 5. Normalization
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 6. Saving
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'policy_train.npy'), scaled_features)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'policy_scaler.pkl'))
    
    print("\n--- Preprocessing Success with Cyclic Time ---")
    print(f"Matrix shape: {scaled_features.shape}")

if __name__ == "__main__":
    preprocess_unified_cert()