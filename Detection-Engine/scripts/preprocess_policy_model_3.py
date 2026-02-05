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

        if filename == 'http.csv':
            df['activity_type'] = 1 # HTTP Activity
        elif filename == 'logon.csv':
            df['activity_type'] = df['activity'].map({'Logon': 2, 'Logoff': 3}).fillna(0)
        elif filename == 'device.csv':
            df['activity_type'] = df['activity'].map({'Connect': 4, 'Disconnect': 5}).fillna(0)

        all_events.append(df[['date', 'activity_type']])

    # 1. Combine events
    combined_df = pd.concat(all_events, ignore_index=True)
    
    # 2. Time Handling & Cyclic Encoding
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['hour'] = combined_df['date'].dt.hour
    combined_df['hour_sin'] = np.sin(2*np.pi*combined_df['hour']/24)
    combined_df['hour_cos'] = np.cos(2*np.pi*combined_df['hour']/24)
    combined_df['is_weekend'] = (combined_df['date'].dt.dayofweek >= 5).astype(int)
    

    
    features = combined_df[['hour_sin', 'hour_cos', 'is_weekend', 'activity_type']]
    
    features = features.dropna()
    

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    variances = np.var(scaled_features, axis=0)
    print(f"Feature variances after scaling: {variances}")
    
    # Saving
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'policy_train.npy'), scaled_features)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'policy_scaler.pkl'))
    
    print("\n--- Preprocessing Success with Cyclic Time ---")
    print(f"Matrix shape: {scaled_features.shape}")

if __name__ == "__main__":
    preprocess_unified_cert()