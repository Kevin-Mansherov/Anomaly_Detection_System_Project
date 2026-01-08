"""
PURPOSE:
Implements the training pipeline for the Policy Enforcement RBM[cite: 160].
This script:
1. Loads preprocessed user behavior vectors[cite: 301].
2. Trains the RBM using the Adam optimizer and Mean Squared Error (MSE) loss[cite: 304, 312].
3. Includes a Validation Split to monitor performance on unseen data and prevent overfitting[cite: 298].
4. Saves the best model version based on the lowest reconstruction error found during training[cite: 313, 317].
"""

from models.model import RBM

import torch 
import torch.nn as nn
import torch.utils.data
import numpy as np
import time

def train():
    # Hyperparameters settings
    BATCH_SIZE = 1024
    VISIBLE_UNITS = 6
    HIDDEN_UNITS = 24 # Slightly increased to capture more complex user patterns
    EPOCHS = 20
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2 # Reserve 20% of data for validation (unseen data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split the processed data
    data_np = np.load('data/processed/policy/policy_train.npy')
    data_tensor = torch.from_numpy(data_np.astype(np.float32))
    
    train_size = int((1 - VAL_SPLIT) * len(data_tensor))
    train_data, val_data = torch.utils.data.random_split(data_tensor, [train_size, len(data_tensor) - train_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS).to(device)
    optimizer = torch.optim.Adam(rbm.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Reconstruction error measure as per proposal [cite: 312]

    best_val_loss = float('inf')
    
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        rbm.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            # Contrastive Divergence approximation via forward pass
            v_reconstructed, _ = rbm(batch)
            loss = criterion(v_reconstructed, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase: monitor performance on unseen data
        rbm.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                v_rec, _ = rbm(batch)
                val_loss += criterion(v_rec, batch).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        # Early Stopping: save only the best performing model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(rbm.state_dict(), 'models/artifacts/best_policy_model.pth')
            print("--> Best model saved.")

if __name__ == "__main__":
    train()