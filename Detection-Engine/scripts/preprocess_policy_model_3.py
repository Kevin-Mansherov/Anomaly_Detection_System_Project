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
            df['act_name'] = 'http'
        elif filename == 'logon.csv': 
            df['act_name'] = df['activity']
        elif filename == 'device.csv': 
            df['act_name'] = df['activity']

        all_events.append(df[['date', 'act_name']])

    # 1. Combine events
    combined_df = pd.concat(all_events, ignore_index=True)
    
    # 2. Time Handling & Cyclic Encoding
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['hour'] = combined_df['date'].dt.hour
    combined_df['hour_sin'] = np.sin(2*np.pi*combined_df['hour']/24)
    combined_df['hour_cos'] = np.cos(2*np.pi*combined_df['hour']/24)

    combined_df = pd.get_dummies(combined_df, columns=['act_name'], )    

    
    feature_cols = ['hour_sin', 'hour_cos'] + [col for col in combined_df.columns if 'act_name_' in col]
    features = combined_df[feature_cols].astype(float)


    if len(features) > 500000:
        features = features.sample(n=500000, random_state=42)


    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Saving
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'policy_train.npy'), scaled_features)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'policy_scaler.pkl'))
    
    print(f"--- Success! Matrix shape: {scaled_features.shape} ---")
    print(f"Features: {feature_cols}")

if __name__ == "__main__":
    preprocess_unified_cert()