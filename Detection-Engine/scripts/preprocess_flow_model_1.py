import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

DATASETS_DIR = '../Datasets/Model_1_and_2/MachineLearningCSV/MachineLearningCVE'
OUTPUT_DIR = '../data/processed/network'
ARTIFACTS_DIR = '../models/artifacts'

FLOW_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min'
]

def proccess_flow_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    monday_file = os.path.join(DATASETS_DIR, 'Monday-WorkingHours.pcap_ISCX.csv')

    print(f"--- Training Flow Scalar (Model 1) on: {monday_file} ---")
    df = pd.read_csv(monday_file)
    df.columns = df.columns.str.strip()

    df = df[[c for c in FLOW_FEATURES if c in df.columns]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0,inplace=True)

    scalar = MinMaxScaler()
    scaled_data = scalar.fit_transform(df)

    np.save(os.path.join(OUTPUT_DIR,'flow_train.npy'), scaled_data)
    joblib.dump(scalar, os.path.join(ARTIFACTS_DIR, 'flow_scalar.pkl'))
    print(f"Model 1 Preprocessing Complete. Shape: {scaled_data.shape}")

if __name__ == "__main__":
    proccess_flow_data()

    