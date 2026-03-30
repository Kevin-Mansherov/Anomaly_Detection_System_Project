import numpy as np
from InferenceEngine import AnomalyDetector

def run_smoke_test():
    print("--- Starting Interface Engine Smoke Test ---")

    try:
        detector = AnomalyDetector()
    except Exception as e:
        print(f"[FAIL] Initialization Error: {e}")
        return
    
    normal_flow = np.zeros(10)
    mse_normal = detector.process_and_analyze('flow', normal_flow)
    print(f"[TEST] Flow normal data MSE: {mse_normal:.6f}")

    anomaly_flow = np.ones(10) * 10000.0
    mse_anomaly = detector.process_and_analyze('flow', anomaly_flow)    
    print(f"[TEST] Flow anomaly data MSE: {mse_anomaly:.6f}")

    print(f"[TEST] Threshold for Flow: {detector.registry['flow']['threshold']}")
    if mse_anomaly > detector.registry['flow']['threshold']:
        print("[SUCCESS] Anomaly correctly identified.")
    else:
        print("[WARNING] Anomaly NOT identified. Check thresholds.")

if __name__ == "__main__":
    run_smoke_test()