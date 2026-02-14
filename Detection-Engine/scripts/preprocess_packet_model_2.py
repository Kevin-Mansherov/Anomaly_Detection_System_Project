import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

DATASETS_DIR = '../Datasets/Model_1_and_2/MachineLearningCSV/MachineLearningCVE'
OUTPUT_DIR = '../data/processed/network/'
ARTIFACTS_DIR = '../models/artifacts'

PACKET_FEATURES = [
    'Fwd Header Length', 'Bwd Header Length', 'Min Packet Length', 
    'Max Packet Length', 'Packet Length Mean', 'FIN Flag Count', 
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 
    'ACK Flag Count', 'URG Flag Count', 'Average Packet Size'
]

def proccess_packet_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    monday_file = os.path.join(DATASETS_DIR, 'Monday-WorkingHours.pcap_ISCX.csv')

    print(f"--- Training Packet Scalar (Model 2) on: {monday_file} ---")
    df = pd.read_csv(monday_file)
    df.columns = df.columns.str.strip()

    df = df[[c for c in PACKET_FEATURES if c in df.columns]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0,inplace=True)

    scalar = MinMaxScaler()
    scaled_data = scalar.fit_transform(df)

    np.save(os.path.join(OUTPUT_DIR,'packet_train.npy'), scaled_data)
    joblib.dump(scalar, os.path.join(ARTIFACTS_DIR, 'packet_scalar.pkl'))
    print(f"Model 2 Preprocessing Complete. Shape: {scaled_data.shape}")

if __name__ == "__main__":
    proccess_packet_data()
