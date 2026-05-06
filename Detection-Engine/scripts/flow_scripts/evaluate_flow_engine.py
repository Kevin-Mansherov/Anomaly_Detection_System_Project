import sys
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(PROJECT_ROOT)

from models.model import RBM 

# --- Path Configuration ---
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'models/artifacts')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/network/flow_test.npy') 
TEST_LABELS_PATH = os.path.join(PROJECT_ROOT, 'data/processed/network/flow_labels_test.npy')

# Flow Model Parameters
INPUT_DIM = 10
HIDDEN_DIM = 256

def get_latest_model_path():
    pattern = os.path.join(ARTIFACTS_DIR, "flow_model_master_*.pth")
    files = glob.glob(pattern)
    if not files: 
        raise FileNotFoundError("[-] No flow model found in artifacts directory.")
    files.sort()
    return files[-1]

def load_trained_model(device):
    model_path = get_latest_model_path()
    print(f"[INFO] Loading latest Flow Model: {os.path.basename(model_path)}")
    model = RBM(INPUT_DIM, HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def find_optimal_threshold(y_true, energy_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, energy_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[INFO] Loading Flow test data...")
    test_data = np.load(TEST_DATA_PATH)
    y_true = np.load(TEST_LABELS_PATH)
    
    model = load_trained_model(device)
    
    test_tensor = torch.from_numpy(test_data.astype(np.float32)).to(device)
    energy_scores = []
    
    print("[INFO] Calculating Free Energy for each sample...")
    with torch.no_grad():
        for i in range(0, len(test_tensor), 512):
            batch = test_tensor[i:i+512]
            energy = model.free_energy(batch)
            energy_scores.extend(energy.cpu().numpy())
    
    energy_scores = np.array(energy_scores)

    anomaly_scores = -1 * energy_scores
    
    print("[INFO] Searching for optimal Free Energy threshold...")
    opt_threshold, max_f1 = find_optimal_threshold(y_true, anomaly_scores)
    
    y_pred = (anomaly_scores > opt_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n" + "*"*15)
    print("   FLOW MODEL: OPTIMIZED RESULTS (ENERGY)")
    print("*"*15)
    print(f"Best Energy Threshold: {opt_threshold:.6f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1-Score:              {max_f1:.4f}")
    print("*"*15)

    plot_results(y_true, anomaly_scores, opt_threshold, tp, tn, fp, fn, "Flow")

def plot_results(y_true, energy_scores, threshold, tp, tn, fp, fn, model_name):
    plt.figure(figsize=(10, 6))
    plt.hist(energy_scores[y_true==0], bins=100, alpha=0.5, label='Normal', color='green')
    plt.hist(energy_scores[y_true==1], bins=100, alpha=0.5, label='Attack', color='red')
    plt.axvline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.title(f'RBM {model_name} Performance - Free Energy')
    plt.xlabel('Free Energy')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'optimized_distribution_{model_name.lower()}.png')
    plt.show()

    cm_data = [[int(tn), int(fp)], [int(fn), int(tp)]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Pred: Normal', 'Pred: Attack'],
                yticklabels=['Actual: Normal', 'Actual: Attack'])
    plt.title(f'{model_name} Energy-Based Confusion Matrix')
    plt.savefig(f'optimized_confusion_matrix_{model_name.lower()}.png')
    plt.show()

if __name__ == "__main__":
    run_evaluation()