import pandas as pd
import numpy as np
import os
import joblib

DATASET_DIR = '../../Datasets/Model_1_and_2/MachineLearningCSV/MachineLearningCVE'
OUTPUT_DIR = '../../data/processed/network'
MODEL_DIR = '../../models/artifacts'

# טעינת הסקיילר של ה-Flow
scalar = joblib.load(os.path.join(MODEL_DIR, 'flow_scalar.pkl'))

FLOW_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min'
]

def create_flow_test_set():
    test_file = os.path.join(DATASET_DIR, 'Wednesday-workingHours.pcap_ISCX.csv')
    df = pd.read_csv(test_file)
    df.columns = df.columns.str.strip()

    labels = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1).values
    features_df = df[[c for c in FLOW_FEATURES if c in df.columns]].copy()
    
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    # נרמול וחיתוך (Clipping)
    scaled_data = scalar.transform(features_df)
    scaled_data = np.clip(scaled_data, -10, 10) 

    np.save(os.path.join(OUTPUT_DIR, 'flow_test.npy'), scaled_data)
    np.save(os.path.join(OUTPUT_DIR, 'flow_labels_test.npy'), labels)
    print(f"Flow Test Set Created! Shape: {scaled_data.shape}")

if __name__ == "__main__":
    create_flow_test_set()