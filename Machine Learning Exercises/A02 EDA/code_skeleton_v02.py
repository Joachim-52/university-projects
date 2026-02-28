import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, PowerTransformer
from xgboost import XGBClassifier

# --- GLOBALE MODELLE ---
scaler = None
outlier_detector = None
final_clf = None

def train_my_logic():
    global scaler, outlier_detector, final_clf
    
    print("--- Start Training (Upgrade: XGBoost + Scaling) ---")
    
    # 1. Daten laden
    try:
        df_train = pd.read_csv("D.csv")
    except FileNotFoundError:
        print("FEHLER: D.csv fehlt!")
        return

    X_train_raw = df_train.drop(columns=['id', 'label'], errors='ignore')
    y_train = df_train['label']

    # --- UPGRADE 1: SCALING & TRANSFORMATION ---
    scaler = PowerTransformer(method='yeo-johnson') 
    X_train = scaler.fit_transform(X_train_raw)
    
    print("Daten skaliert (PowerTransformer).")

    outlier_detector = IsolationForest(
        n_estimators=500, 
        contamination=0.04,
        random_state=42,
        n_jobs=-1
    )
    outlier_detector.fit(X_train)
 
    is_inlier = outlier_detector.predict(X_train) == 1
    
    X_clean = X_train[is_inlier]
    y_clean = y_train[is_inlier]
    
    print(f"Outlier entfernt. Training Size: {len(X_train)} -> {len(X_clean)}")

    # --- UPGRADE 3: XGBOOST CLASSIFIER ---

    final_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        n_jobs=-1
    )
    final_clf.fit(X_clean, y_clean)
    print("XGBoost Training abgeschlossen.\n")


def predict(X_test):
    global scaler, outlier_detector, final_clf
    
    if scaler is None:
        raise ValueError("Modelle nicht trainiert!")

    # 1. Features vorbereiten & SKALIEREN
    features_raw = X_test.drop(columns=["id"], errors="ignore")
    features_scaled = scaler.transform(features_raw)
    
    # 2. Outlier Detection
    iso_pred = outlier_detector.predict(features_scaled)
    outliers = np.where(iso_pred == -1, 1, 0)
    
    # 3. Classification (XGBoost)
    labels = final_clf.predict(features_scaled)
    
    return labels, outliers


def generate_submission(test_data):
    label_predictions, outlier_predictions = predict(test_data)
    submission_df = pd.DataFrame({ 
        "id": test_data["id"],
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df

def main():
    train_my_logic()
    
    try:
        df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
        submission_df = generate_submission(df_leaderboard)
        submission_df.to_csv("submission_leaderboard_Group17.csv", index=False)
        print("Leaderboard File erstellt.")
    except: pass

    try:
        df_final = pd.read_csv("D_test_final.csv")
        submission_df = generate_submission(df_final)
        submission_df.to_csv("submission_final_Group17.csv", index=False)
        print("Final File erstellt.")
    except: pass

if __name__ == "__main__":
    main()