import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- GLOBALE MODELLE ---
scaler = None
gmm = None
final_clf = None
tau = None

def add_features(df):
    """
    Feature Engineering
    """
    X = df.copy()
    
    cols = [c for c in X.columns if c not in ['id', 'label', 'outlier']]
    
    # Neue Features berechnen
    X['mean'] = X[cols].mean(axis=1)
    X['std'] = X[cols].std(axis=1)
    X['min'] = X[cols].min(axis=1)
    X['max'] = X[cols].max(axis=1)
    
    return X

def train_my_logic():
    global scaler, gmm, final_clf, tau
    
    print("--- Start Training (Hybrid: XGBoost + GMM + Feature Engineering) ---")
    
    # 1. Daten laden
    try:
        df_train = pd.read_csv("D.csv")
        df_out_ref = pd.read_csv("D_out.csv")
    except FileNotFoundError:
        print("FEHLER: D.csv oder D_out.csv fehlt!")
        return

    # 2. Daten vorbereiten (X und y trennen)
    y_train = df_train['label']
    
    # Feature Engineering anwenden (auf beiden Sets)
    X_train_raw = add_features(df_train.drop(columns=['id', 'label'], errors='ignore'))
    X_out_raw = add_features(df_out_ref.drop(columns=['id', 'label'], errors='ignore'))

    # 3. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_out_ref = scaler.transform(X_out_raw)
    
    print("Feature Engineering & Scaling abgeschlossen.")

    # --- PART A: OUTLIER DETECTION (GMM mit Referenz-Tuning) ---
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm.fit(X_train)
    
    # Scores berechnen (Log-Likelihood)
    scores_train = gmm.score_samples(X_train)
    scores_out = gmm.score_samples(X_out_ref)
    
    # Threshold Tau bestimmen
    tau = np.percentile(scores_out, 99) 
    print(f"GMM trainiert. Threshold Tau gesetzt auf: {tau:.4f}")
    
    # --- PART B: CLASSIFICATION (XGBoost) ---
    mask_clean = scores_train >= tau
    X_train_clean = X_train[mask_clean]
    y_train_clean = y_train[mask_clean]
    
    print(f"Training XGBoost auf {len(X_train_clean)} Samples (Cleaned)...")
    
    final_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        n_jobs=-1
    )
    final_clf.fit(X_train_clean, y_train_clean)
    print("XGBoost Training abgeschlossen.\n")


def predict(X_test):
    global scaler, gmm, final_clf, tau
    
    if scaler is None:
        raise ValueError("Modelle nicht trainiert!")

    # 1. Feature Engineering
    features_raw = X_test.drop(columns=["id"], errors="ignore")
    features_eng = add_features(features_raw)
    
    # 2. Skalieren
    features_scaled = scaler.transform(features_eng)
    
    # 3. Outlier Detection (GMM)
    scores = gmm.score_samples(features_scaled)
    # Wenn Score < Tau -> Outlier (1), sonst Normal (0)
    outliers = (scores < tau).astype(int)
    
    # 4. Classification (XGBoost)
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
    
    # Leaderboard
    try:
        df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
        submission_df = generate_submission(df_leaderboard)
        submission_df.to_csv("submission_leaderboard_Group17.csv", index=False)
        print("Leaderboard-Datei erstellt.")
    except Exception as e:
        print(f"Fehler bei Leaderboard: {e}")

    # Final
    try:
        df_final = pd.read_csv("D_test_final.csv")
        submission_df = generate_submission(df_final)
        submission_df.to_csv("submission_final_Group17.csv", index=False)
        print("Final-Datei erstellt.")
    except Exception as e:
        print(f"Fehler bei Final: {e}")

if __name__ == "__main__":
    main()