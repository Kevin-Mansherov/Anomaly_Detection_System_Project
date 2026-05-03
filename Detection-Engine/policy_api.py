import os
import torch
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ייבוא המודל מתוך תיקיית models
from models.model import RBM

# הגדרת השרת
app = FastAPI(title="UBA Policy API")

# נתיבים קשיחים לפי המבנה שלך
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, "models", "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "policy_model_master_20260311-143625.pth") # <-- שים לב: שנה לשם קובץ ה-pth האמיתי שלך
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "policy_scaler.pkl")

policy_model = None
policy_scaler = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- הגדרות ---
# לפי סקריפט האימון שלך, יש לנו 19 עמודות של שעות (5 עד 23) ועוד עמודות של פעילויות
# לרוב מדובר ב-5 פעילויות (Logon, Logoff, http, Connect, Disconnect). סך הכל 24 עמודות.
POLICY_VISIBLE_UNITS = 24 
POLICY_HIDDEN_UNITS = 64
POLICY_THRESHOLD = -115.38

# רשימת העמודות המדויקת כפי שסודרה בסקריפט האימון
EXPECTED_HOURS = sorted([f'hour_{h}' for h in range(5, 24)])
EXPECTED_ACTS = sorted(['act_name_Connect', 'act_name_Disconnect', 'act_name_Logoff', 'act_name_Logon', 'act_name_http'])
FEATURE_COLS = EXPECTED_HOURS + EXPECTED_ACTS

# הגדרת מבנה הבקשה מ-Spring Boot
class LogEvent(BaseModel):
    event_id: str
    user: str
    timestamp: str
    activity: str
    pc: str
    source: str

@app.on_event("startup")
def load_assets():
    global policy_model, policy_scaler
    try:
        print("[INFO] Loading Policy RBM & Scaler...")
        policy_scaler = joblib.load(SCALER_PATH)
        
        policy_model = RBM(n_visible=POLICY_VISIBLE_UNITS, n_hidden=POLICY_HIDDEN_UNITS).to(DEVICE)
        policy_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        policy_model.eval()
        
        print("[+] Assets loaded successfully.")
    except Exception as e:
        print(f"[-] ERROR loading assets: {e}")

def preprocess_log(event: LogEvent):
    """
    משחזר את תהליך יצירת הפיצ'רים מקובץ האימון
    """
    # יצירת מילון עם אפסים לכל העמודות האפשריות
    features = {col: 0.0 for col in FEATURE_COLS}
    
    # 1. חילוץ שעה
    try:
        dt = pd.to_datetime(event.timestamp)
        hour = dt.hour
    except:
        hour = 0
        
    # הדלקת עמודת השעה (רק אם היא בין 5 ל-23)
    hour_col = f'hour_{hour}'
    if hour_col in features:
        features[hour_col] = 1.0

    # 2. חילוץ סוג פעילות
    # בסקריפט האימון אמרת שאם זה קובץ http, הפעילות היא 'http'
    act_name = 'http' if 'http' in event.source.lower() else event.activity
    
    act_col = f'act_name_{act_name}'
    if act_col in features:
        features[act_col] = 1.0
        
    # 3. החזרת DataFrame עם העמודות בסדר המדויק
    df = pd.DataFrame([features])[FEATURE_COLS]
    return df

@app.post("/api/analyze_log")
def analyze_log(event: LogEvent):
    global policy_model, policy_scaler
    
    try:
        # Preprocessing
        df_features = preprocess_log(event)
        
        # Scaling (חובה, כי ככה המודל אומן)
        scaled_features = policy_scaler.transform(df_features)
        scaled_features = np.clip(scaled_features, 0, 1) # כמו שעשית באימון
        tensor_features = torch.FloatTensor(scaled_features).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            energy = policy_model.free_energy(tensor_features).item()
            
        # Decision
        is_anomaly = energy > POLICY_THRESHOLD
        
        return {
            "status": "success",
            "event_id": event.event_id,
            "is_anomaly": is_anomaly,
            "energy_score": energy 
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # השרת ירוץ על פורט 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)