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
import numpy as np
import os

def train_rbm():
    # Hyperparameters settings
    BATCH_SIZE = 512
    VISIBLE_UNITS = 7
    # HIDDEN_UNITS = 64 
    HIDDEN_UNITS = 128
    # EPOCHS = 50
    EPOCHS = 100
    # LEARNING_RATE = 0.01
    LEARNING_RATE = 0.005
    # CD_K = 1 # Contrastive Divergence steps
    CD_K = 3 
    VAL_SPLIT = 0.2 # 20% for validation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = np.load('Datasets/processed/policy/policy_train.npy')
    data_tensor = torch.from_numpy(data.astype(np.float32))

    actual_visible_units = data_tensor.shape[1]
    print(f"DEBUG: Detected {actual_visible_units} features in dataset.")
    print(f"Using device: {device}")

    # Validation Split
    train_size = int((1-VAL_SPLIT) * len(data_tensor))
    val_size = len(data_tensor) - train_size
    train_subset, val_subset = random_split(data_tensor, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Model and Optimizer Initialization
    rbm = RBM(actual_visible_units, HIDDEN_UNITS, k=CD_K).to(device)
    optimizer = torch.optim.SGD(rbm.parameters(), lr=LEARNING_RATE)

    # Initialize Early Stopping
    os.makedirs('models/artifacts', exist_ok=True)
    # stopper = EarlyStopping(patience=5, path='models/artifacts/best_policy_rbm.pth')
    stopper = EarlyStopping(patience=10, path='models/artifacts/best_policy_rbm.pth')

    print(f"--- Starting Energy-Based Training on {device} ---")

    for epoch in range(EPOCHS):
        rbm.train()
        train_energy = 0
        for batch in train_loader:
            batch = batch.to(device)
            v_pos, v_neg = rbm(batch)

            loss = rbm.free_energy(v_pos) - rbm.free_energy(v_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_energy += loss.item()

        rbm.eval()
        val_energy = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                v_pos, v_neg = rbm(batch)
                val_energy += (rbm.free_energy(v_pos) - rbm.free_energy(v_neg)).item()

        avg_train = train_energy / len(train_loader)
        avg_val = val_energy / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Energy: {avg_train:.6f} | Val Energy: {avg_val:.6f}")

        # Check early stopping 
        stopper(avg_val, rbm)
        if stopper.early_stop:
            print("--> Early stopping triggered. Ending training.")
            break

if __name__ == "__main__":
    train_rbm()