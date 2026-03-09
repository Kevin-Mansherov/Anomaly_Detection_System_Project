import pandas as pd
import numpy as np
import os
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '../..'))

# הגדרות נתיבים
BASE_DIR = os.path.join(PROJECT_ROOT, 'Datasets/Model_3/r1/r1/') 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data/processed/policy')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'models/artifacts')
FILES_TO_MERGE = ['logon.csv', 'device.csv', 'http.csv']

def create_policy_test_set_with_labels():
    print("--- Creating Policy Test Set + Labels (Model 3) ---")
    
    # 1. טעינת הסקיילר
    scaler_path = os.path.join(ARTIFACTS_DIR, 'policy_scaler.pkl')
    if not os.path.exists(scaler_path):
        print("[ERROR] Policy Scaler not found!")
        return
    scaler = joblib.load(scaler_path)
    
    # שליפת שמות העמודות שהיו בזמן האימון (Fit)
    expected_features = scaler.feature_names_in_

    all_events = []
    for filename in FILES_TO_MERGE:
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path): 
            print(f"[WARNING] File {filename} not found in {BASE_DIR}. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # זיהוי סוג האירוע
        if filename == 'http.csv': 
            df['act_name'] = 'http'
        elif filename in ['logon.csv', 'device.csv']: 
            df['act_name'] = df['activity']
        
        # לוגיקת תוויות (Labels)
        df['date'] = pd.to_datetime(df['date'])
        df['is_anomaly'] = (df['date'].dt.hour < 5).astype(int)

        all_events.append(df[['date', 'act_name', 'is_anomaly']])

    combined_df = pd.concat(all_events, ignore_index=True)
    # 2. קידוד זמן מעגלי
    combined_df['hour'] = combined_df['date'].dt.hour
    # 3. שמירת תוויות
    labels = combined_df['is_anomaly'].values

    # 4. One-Hot Encoding
    combined_df = pd.get_dummies(combined_df, columns=['act_name'])
    
    # --- התיקון הקריטי כאן: יישור עמודות ---
    # אנחנו מכריחים את ה-DataFrame להכיל רק את העמודות שהסקיילר מכיר
    # עמודות חסרות ימולאו ב-0, ועמודות חדשות יימחקו
    features = combined_df.reindex(columns=expected_features, fill_value=0)
    # ---------------------------------------

    # 5. נרמול וקיצוץ (עכשיו זה יעבוד!)
    scaled_features = scaler.transform(features)


    scaled_features = np.clip(scaled_features, 0, 1)

    # 6. שמירה
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'policy_test.npy'), scaled_features)
    np.save(os.path.join(OUTPUT_DIR, 'policy_labels_test.npy'), labels)
    
    print(f"[SUCCESS] Policy Test Set & Labels saved!")
    print(f"Features Aligned: {len(expected_features)} columns maintained.")

if __name__ == "__main__":
    create_policy_test_set_with_labels()