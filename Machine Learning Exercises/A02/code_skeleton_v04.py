import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- GLOBALE MODELLE ---
scaler = None
outlier_clf = None
final_clf = None

def train_my_logic():
    global scaler, outlier_clf, final_clf
    
    print("--- Start Training---")
    
    # 1. Daten laden
    try:
        df_train = pd.read_csv("D.csv")
        df_out = pd.read_csv("D_out.csv")
    except FileNotFoundError:
        print("FEHLER: Dateien fehlen!")
        return

    # 2. Features vorbereiten
    X_good = df_train.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    y_good_labels = df_train['label']
    
    X_bad = df_out.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    
    # 3. Scaling
    scaler = StandardScaler()
    X_good_scaled = scaler.fit_transform(X_good)
    X_bad_scaled = scaler.transform(X_bad)
    
    # --- PHASE 1: OUTLIER DETECTOR TRAINIEREN ---
    X_combined = np.vstack([X_good_scaled, X_bad_scaled])
    # Label 0 für Inlier (D.csv), Label 1 für Outlier (D_out.csv)
    y_combined = np.hstack([np.zeros(len(X_good_scaled)), np.ones(len(X_bad_scaled))])
    
    print(f"Trainiere Outlier-Detector auf {len(X_combined)} Zeilen...")
    
    outlier_clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    outlier_clf.fit(X_combined, y_combined)
    
    # --- PHASE 2: CLASS DETECTOR TRAINIEREN ---
    print(f"Trainiere Classifier auf {len(X_good_scaled)} sauberen Zeilen...")
    
    final_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        n_jobs=-1
    )
    final_clf.fit(X_good_scaled, y_good_labels)
    
    print("Training abgeschlossen. Bereit für den Sieg.")


def predict(X_test):
    global scaler, outlier_clf, final_clf
    
    if scaler is None:
        raise ValueError("Modell nicht trainiert!")

    # 1. Features vorbereiten & Skalieren
    features_raw = X_test.drop(columns=["id"], errors="ignore")
    features_scaled = scaler.transform(features_raw)
    
    # 2. Outlier vorhersagen
    # 1 = Outlier, 0 = Inlier
    outlier_preds = outlier_clf.predict(features_scaled).astype(int)
    
    # 3. Label vorhersagen
    label_preds = final_clf.predict(features_scaled).astype(int)
    
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
    
    # Leaderboard
    try:
        df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
        sub = generate_submission(df_leaderboard)
        sub.to_csv("submission_leaderboard_Group17.csv", index=False)
        print("Datei erstellt: submission_leaderboard_Group17.csv")
    except Exception as e:
        print(f"Leaderboard Fehler: {e}")

    # Final
    try:
        df_final = pd.read_csv("D_test_final.csv")
        sub = generate_submission(df_final)
        sub.to_csv("submission_final_Group17.csv", index=False)
        print("Datei erstellt: submission_final_Group17.csv")
    except Exception as e:
        print(f"Final Fehler: {e}")

if __name__ == "__main__":
    main()