import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score 
from models.model import RBM 

# --- הגדרות נתיבים ---
MODEL_PATH = '../models/artifacts/policy_model_master_20260306-174032.pth' 
TEST_DATA_PATH = '../data/processed/policy/policy_test.npy'
TEST_LABELS_PATH = '../data/processed/policy/policy_labels_test.npy'

def load_trained_model(input_dim, hidden_dim, device):
    model = RBM(input_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def find_optimal_threshold(y_true, mse_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, mse_scores)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    return best_threshold, f1_scores[best_idx]

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. טעינת נתונים
    print("[INFO] Loading test data...")
    test_data = np.load(TEST_DATA_PATH)
    y_true = np.load(TEST_LABELS_PATH)
    
    # 2. טעינת מודל (כאן תעדכן ל-256 אחרי האימון מחדש)
    input_dim = test_data.shape[1]
    hidden_dim = 256 # שנה ל-256 ברגע שתאמן מחדש
    model = load_trained_model(input_dim, hidden_dim, device)
    
    # 3. הרצה וחישוב MSE
    test_tensor = torch.from_numpy(test_data.astype(np.float32)).to(device)
    mse_scores = []
    
    print("[INFO] Calculating MSE for each sample...")
    with torch.no_grad():
        for i in range(0, len(test_tensor), 512):
            batch = test_tensor[i:i+512]
            v_pos, v_neg = model(batch)
            loss = torch.mean((v_pos - v_neg)**2, dim=1)
            mse_scores.extend(loss.cpu().numpy())
    
    mse_scores = np.array(mse_scores)
    
    # 4. מציאת הסף האופטימלי (שיפור הדיוק)
    print("[INFO] Searching for the optimal threshold...")
    opt_threshold, max_f1 = find_optimal_threshold(y_true, mse_scores)
    
    # 5. חישוב מדדים ידני לפי הסף האופטימלי
    y_pred = (mse_scores > opt_threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n" + "⭐"*15)
    print(f"   OPTIMIZED RESULTS")
    print("⭐"*15)
    print(f"Best Threshold:  {opt_threshold:.6f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-Score:        {max_f1:.4f}")
    print(f"Missed Attacks:  {fn} (out of {tp+fn})")
    print("⭐"*15)

    # 6. יצירת גרפים
    plot_results(y_true, mse_scores, opt_threshold, tp, tn, fp, fn)

def plot_results(y_true, mse_scores, threshold, tp, tn, fp, fn):
    # גרף התפלגות
    plt.figure(figsize=(10, 6))
    plt.hist(mse_scores[y_true==0], bins=100, alpha=0.5, label='Normal', color='green', log=True)
    plt.hist(mse_scores[y_true==1], bins=100, alpha=0.5, label='Attack', color='red', log=True)
    plt.axvline(threshold, color='blue', linestyle='--', label=f'Optimal Threshold ({threshold:.4f})')
    plt.title('RBM Performance - Optimized Threshold')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count (Log Scale)')
    plt.legend()
    plt.savefig('optimized_distribution.png')
    plt.show()

    # 

    # מטריצת בלבול
    cm_data = [[int(tn), int(fp)], [int(fn), int(tp)]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Pred: Normal', 'Pred: Attack'],
                yticklabels=['Actual: Normal', 'Actual: Attack'])
    plt.title('Optimized Confusion Matrix')
    plt.savefig('optimized_confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    run_evaluation()