from models.model import RBM
from utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
import os
import torch.nn.functional as F


def train_flow_rbm():
    BATCH_SIZE = 512
    HIDDEN_UNITS = 256
    LEARNING_RATE = 0.0002
    EPOCHS = 200
    CD_K = 10
    VAL_SPLIT = 0.1


    # BATCH_SIZE = 256
    # HIDDEN_UNITS = 128
    # EPOCHS = 100
    # LEARNING_RATE = 0.0001
    # CD_K = 10
    # VAL_SPLIT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Model 1 (Flow) on {device} ---")

    data_path = 'data/processed/network/flow_train.npy'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run preprocessing first.")

    data = np.load(data_path)
    data_tensor = torch.from_numpy(data.astype(np.float32)).to(device)

    train_size = int((1-VAL_SPLIT) * len(data_tensor))
    train_subset, val_subset = random_split(data_tensor, [train_size, len(data_tensor) - train_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    visible_units = data.shape[1]
    rbm = RBM(visible_units, HIDDEN_UNITS, k=CD_K).to(device)
    optimizer = torch.optim.Adam(rbm.parameters(), lr=LEARNING_RATE)

    os.makedirs('models/artifacts', exist_ok=True)
    stopper = EarlyStopping(patience=15, path='models/artifacts/best_flow_model.pth')

    for epoch in range(EPOCHS):
        rbm.train()
        train_mse = 0
        for batch in train_loader:
            # batch = batch.to(device)
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

        avg_train = train_mse / len(train_loader)
        avg_val = val_mse / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        stopper(avg_val, rbm)
        if stopper.early_stop:
            print("--> Early stopping triggered. Ending training.")
            break

if __name__ == "__main__":
    train_flow_rbm()