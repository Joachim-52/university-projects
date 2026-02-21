import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- GLOBALE MODELLE ---
scaler = None
outlier_clf = None
label_clf = None

def train_my_logic():
    global scaler, outlier_clf, label_clf
    
    print("--- Start Training: XGBoost ---")
    
    # 1. Daten laden
    try:
        df_train = pd.read_csv("D.csv")
        df_out_ref = pd.read_csv("D_out.csv")
    except FileNotFoundError:
        print("FEHLER: Dateien fehlen!")
        return

    X_in = df_train.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    y_in_labels = df_train['label']
    X_out = df_out_ref.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    
    # 2. Scaling
    scaler = StandardScaler()
    X_in_scaled = scaler.fit_transform(X_in)
    X_out_scaled = scaler.transform(X_out)
    
    # 3. Outlier Training
    X_binary = np.vstack([X_in_scaled, X_out_scaled])
    y_binary = np.hstack([np.zeros(len(X_in)), np.ones(len(X_out))])
    
    # Starke Gewichtung beibehalten
    ratio = len(X_in) / len(X_out)
    
    outlier_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        n_jobs=-1
    )
    outlier_clf.fit(X_binary, y_binary)
    print("  -> Outlier-Modell trainiert.")
    
    # 4. Label Training
    label_clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        n_jobs=-1
    )
    label_clf.fit(X_in_scaled, y_in_labels)
    print("  -> Label-Modell trainiert.")


def predict(X_test):
    global scaler, outlier_clf, label_clf
    
    if scaler is None:
        raise ValueError("Modell nicht trainiert!")

    features_raw = X_test.drop(columns=["id"], errors="ignore")
    features_scaled = scaler.transform(features_raw)
    
    probs = outlier_clf.predict_proba(features_scaled)[:, 1]
 
    THRESHOLD = 0.20 
    outlier_preds = (probs > THRESHOLD).astype(int)
    
    count_ones = np.sum(outlier_preds)
    print(f"  -> PREDICT CHECK: Habe {count_ones} Outliers gefunden (bei Threshold {THRESHOLD})")

    label_preds = label_clf.predict(features_scaled).astype(int)
    
    return label_preds, outlier_preds


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
        print("\nErstelle Leaderboard Submission...")
        sub = generate_submission(df_leaderboard)
        sub.to_csv("submission_leaderboard_Group17.csv", index=False)
        print("Fertig.")
    except: pass

    try:
        df_final = pd.read_csv("D_test_final.csv")
        print("\nErstelle Final Submission...")
        sub = generate_submission(df_final)
        sub.to_csv("submission_final_Group17.csv", index=False)
        print("Fertig.")
    except: pass

if __name__ == "__main__":
    main()