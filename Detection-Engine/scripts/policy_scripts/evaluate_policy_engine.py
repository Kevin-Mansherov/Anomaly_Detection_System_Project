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
    if not files: 
        raise FileNotFoundError("No model found.")
    files.sort()
    return files[-1]

def run_evaluation():
    print("[INFO] Starting Evaluation Engine (Weighted One-Hot)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"[ERROR] Test data not found at {TEST_DATA_PATH}")
        return
    
    test_data_np = np.load(TEST_DATA_PATH).astype(np.float32)
    test_data = torch.from_numpy(test_data_np).to(device)
    y_true = np.load(TEST_LABELS_PATH)

    model_path = get_latest_model_path()
    input_dim = test_data.shape[1]
    print(f"[INFO] Loading Model: {os.path.basename(model_path)} with input_dim/features={input_dim}")

    model = RBM(input_dim, 64).to(device) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        energies = model.free_energy(test_data).cpu().numpy()
        # v_pos, v_neg = model(test_data)
        # diff_sq = (v_pos - v_neg) ** 2
        
        # # חישוב כמות עמודות השעה מהסקיילר
        # scaler = joblib.load(SCALER_PATH)
        # hour_col_count = len([c for c in scaler.feature_names_in_ if 'hour_' in c])
        
        # # מתן משקל 10 לעמודות השעה (הן הראשונות בזכות התיקון ב-Train)
        # weights = torch.ones(input_dim).to(device)
        # weights[0:hour_col_count] = 10.0 
        
        # mse_scores = (diff_sq * weights).mean(dim=1).cpu().numpy()

    # חישוב מדדים
    precisions, recalls, thresholds = precision_recall_curve(y_true, energies)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)

    opt_threshold = thresholds[min(best_idx, len(thresholds)-1)]

    y_pred = (energies > opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n" + "⭐"*20)
    print("      POLICY EVALUATION RESULTS")
    print("⭐"*20)
    print(f"Optimal Threshold:  {opt_threshold:.2f}")
    print(f"Precision:          {precisions[best_idx]:.4f}")
    print(f"Recall (Detection):  {recalls[best_idx]:.4f}")
    print(f"F1-Score:           {f1_scores[best_idx]:.4f}")
    print(f"False Alarms (FP):  {fp} / {tn+fp}")
    print(f"Missed Attacks (FN): {fn} / {tp+fn}")
    print("⭐"*20)
    
    # 6. הצגת גרף התפלגות
    plot_results(energies, y_true, opt_threshold)

def plot_results(energies, y_true, threshold):
    plt.figure(figsize=(12, 6))
    
    # נורמלי בירוק, התקפה באדום
    sns.histplot(energies[y_true==0], color='green', label='Normal (Work Hours)', kde=True, alpha=0.5)
    sns.histplot(energies[y_true==1], color='red', label='Attack (Off Hours)', kde=True, alpha=0.5)
    
    plt.axvline(threshold, color='blue', linestyle='--', label=f'Optimal Threshold ({threshold:.2f})')
    
    plt.title('Anomaly Detection Performance - Free Energy Distribution')
    plt.xlabel('Free Energy Score (Higher = More Anomalous)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # שמירת הגרף
    output_png = os.path.join(current_dir, 'policy_energy_dist.png')
    plt.savefig(output_png)
    print(f"\n[INFO] Distribution plot saved to: {output_png}")
    plt.show()

#     print(f"\n" + "⭐"*20)
#     print(f"       POLICY MODEL RESULTS")
#     print("⭐"*20)
#     print(f"Optimal Threshold: {opt_threshold:.6f}")
#     print(f"Precision:         {precisions[best_idx]:.4f}")
#     print(f"Recall (Detection): {recalls[best_idx]:.4f}")
#     print(f"F1-Score:          {f1_scores[best_idx]:.4f}")
#     print(f"Missed Attacks:    {fn} (out of {tp+fn})")
#     print(f"False Alarms:      {fp} (out of {tn+fp})")
#     print("⭐"*20)
    
#     # ויזואליזציה
#     plot_evaluation(anomaly_scores, y_true, opt_threshold)
    
# def plot_evaluation(anomaly_scores, y_true, threshold):
#     plt.figure(figsize=(12, 6))
#     sns.histplot(anomaly_scores[y_true==0], color='green', label='Normal (Work Hours)', kde=True, log_scale=True, alpha=0.5)
#     sns.histplot(anomaly_scores[y_true==1], color='red', label='Attack (Off Hours)', kde=True, log_scale=True, alpha=0.5)
#     plt.axvline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
#     plt.title('Policy Model Anomaly Distribution (Weighted MSE)')
#     plt.xlabel('Reconstruction Error (Log Scale)')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # שמירת התוצאה
#     save_path = os.path.join(current_dir, 'evaluation_plot.png')
#     plt.savefig(save_path)
#     print(f"\n[INFO] Distribution plot saved to: {save_path}")
#     plt.show()

if __name__ == "__main__":
    run_evaluation()