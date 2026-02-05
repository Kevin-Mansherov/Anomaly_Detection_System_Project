"""
PURPOSE:
Implements the training pipeline for the Policy Enforcement RBM.
This script:
1. Loads preprocessed user behavior vectors.
2. Trains the RBM using Energy-Based objective (Contrastive Divergence).
3. Employs early stopping based on validation energy gap to prevent overfitting, and saves the best model.
"""

from models.model import RBM
from utils.early_stopping import EarlyStopping

import torch 
import torch.nn as nn
from torch.utils.data  import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import os

def train_rbm():
    # Hyperparameters settings
    BATCH_SIZE = 512
    # HIDDEN_UNITS = 64 
    HIDDEN_UNITS = 256
    # EPOCHS = 50
    EPOCHS = 200
    # LEARNING_RATE = 0.01
    LEARNING_RATE = 0.0002
    # CD_K = 1 # Contrastive Divergence steps
    CD_K = 10
    VAL_SPLIT = 0.2 # 20% for validation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Model 3 (Policy) on {device} ---")

    data_path = 'Datasets/processed/policy/policy_train.npy'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run preprocessing first.")
        return
    
    # Load data
    data = np.load(data_path)
    data_tensor = torch.from_numpy(data.astype(np.float32)).to(device)

    # Validation Split
    train_size = int((1-VAL_SPLIT) * len(data_tensor))
    
    val_size = len(data_tensor) - train_size
    train_subset, val_subset = random_split(data_tensor, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Model and Optimizer Initialization
    visible_units = data.shape[1]
    rbm = RBM(visible_units, HIDDEN_UNITS, k=CD_K).to(device)
    optimizer = torch.optim.SGD(rbm.parameters(), lr=LEARNING_RATE)

    # Initialize Early Stopping
    os.makedirs('models/artifacts', exist_ok=True)
    # stopper = EarlyStopping(patience=5, path='models/artifacts/best_policy_rbm.pth')
    stopper = EarlyStopping(patience=15, path='models/artifacts/best_policy_rbm.pth')


    for epoch in range(EPOCHS):
        rbm.train()
        train_mse = 0
        for batch in train_loader:
            v_pos, v_neg = rbm(batch)
            
            loss = rbm.free_energy(v_pos) - rbm.free_energy(v_neg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_mse += F.mse_loss(v_pos, v_neg).item()

        rbm.eval()
        val_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                v_pos, v_neg = rbm(batch)
                val_mse += F.mse_loss(v_pos, v_neg).item()

        avg_train_mse = train_mse / len(train_loader)
        avg_val_mse = val_mse / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")

        stopper(avg_val_mse, rbm)
        if stopper.early_stop:
            print(f"--> Stopping at epoch {epoch+1}. Model 3 is optimized.")
            break

if __name__ == "__main__":
    train_rbm()