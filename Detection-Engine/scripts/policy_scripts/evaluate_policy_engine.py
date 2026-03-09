import sys
import os
import glob
import torch
import numpy as np
import joblib # תיקון: ייבוא חסר
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
# עולים 3 רמות אם הקובץ בתוך scripts/policy/
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(PROJECT_ROOT)

from models.model import RBM 

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'models/artifacts')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/policy/policy_test.npy')
TEST_LABELS_PATH = os.path.join(PROJECT_ROOT, 'data/processed/policy/policy_labels_test.npy')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'policy_scaler.pkl')

def get_latest_model_path():
    pattern = os.path.join(ARTIFACTS_DIR, "policy_model_master_*.pth")
    files = glob.glob(pattern)
    if not files: raise FileNotFoundError("No model found.")
    files.sort()
    return files[-1]

def run_evaluation():
    print("[INFO] Starting Evaluation Engine (Weighted One-Hot)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_data_np = np.load(TEST_DATA_PATH).astype(np.float32)
    y_true = np.load(TEST_LABELS_PATH)
    test_data = torch.from_numpy(test_data_np).to(device)

    model_path = get_latest_model_path()
    input_dim = test_data.shape[1]
    model = RBM(input_dim, 256).to(device) # וודא שזה 256 כמו באימון
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        v_pos, v_neg = model(test_data)
        diff_sq = (v_pos - v_neg) ** 2
        
        # חישוב כמות עמודות השעה מהסקיילר
        scaler = joblib.load(SCALER_PATH)
        hour_col_count = len([c for c in scaler.feature_names_in_ if 'hour_' in c])
        
        # מתן משקל 10 לעמודות השעה (הן הראשונות בזכות התיקון ב-Train)
        weights = torch.ones(input_dim).to(device)
        weights[0:hour_col_count] = 10.0 
        
        mse_scores = (diff_sq * weights).mean(dim=1).cpu().numpy()

    # חישוב מדדים
    precisions, recalls, thresholds = precision_recall_curve(y_true, mse_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    opt_threshold = thresholds[best_idx]

    y_pred = (mse_scores > opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n" + "⭐"*15)
    print(f"RESULTS:\nThreshold: {opt_threshold:.6f}\nPrecision: {precisions[best_idx]:.4f}\nRecall: {recalls[best_idx]:.4f}\nF1: {f1_scores[best_idx]:.4f}")
    print(f"Missed Attacks: {fn} (out of {tp+fn})")
    print("⭐"*15)
    
    # גרף
    plt.figure(figsize=(10,5))
    sns.histplot(mse_scores[y_true==0], color='green', label='Normal', kde=True, log_scale=True)
    sns.histplot(mse_scores[y_true==1], color='red', label='Attack', kde=True, log_scale=True)
    plt.axvline(opt_threshold, color='blue', linestyle='--', label='Threshold')
    plt.title('Weighted One-Hot Policy Performance')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_evaluation()