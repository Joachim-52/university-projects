import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
from sklearn.preprocessing import PowerTransformer

# --- GLOBALE MODELLE ---
gmm_model = None
xgb_model = None
scaler = None
tau = None

def train_my_logic():
    global gmm_model, xgb_model, scaler, tau
    
    print("--- Start Training ---")
    
    # 1. Daten laden
    try:
        df_train = pd.read_csv("D.csv")
        df_out_ref = pd.read_csv("D_out.csv")
    except FileNotFoundError:
        print("FEHLER: Dateien fehlen!")
        return

    # Features trennen
    X_train_raw = df_train.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    X_out_raw = df_out_ref.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    y_train = df_train['label']

    # TEIL 1: OUTLIER DETECTION
    print("1. Trainiere Outlier Detector (GMM Raw)...")
    gmm_model = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm_model.fit(X_train_raw)
    
    # Threshold Tuning: 99. Perzentil
    scores_out = gmm_model.score_samples(X_out_raw)
    tau = np.percentile(scores_out, 99)
    print(f"   -> Threshold Tau gesetzt auf: {tau:.4f}")

    # TEIL 2: KLASSIFIKATION
    print("2. Trainiere Classifier (XGBoost PowerTransformer)...")
    
    # PowerTransformer statt StandardScaler
    scaler = PowerTransformer(method='yeo-johnson')
    X_train_transformed = scaler.fit_transform(X_train_raw)
    
    print(f"   -> Trainiere XGBoost auf vollem Dataset ({len(X_train_transformed)} Samples)...")
    
    xgb_model = XGBClassifier(
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
    xgb_model.fit(X_train_transformed, y_train)
    print("Training abgeschlossen.")


def predict(X_test):
    global gmm_model, xgb_model, scaler, tau
    
    if gmm_model is None:
        raise ValueError("Modell nicht trainiert!")

    features_raw = X_test.drop(columns=["id"], errors="ignore")
    
    # 1. Outlier Prediction
    scores = gmm_model.score_samples(features_raw)
    outliers = (scores < tau).astype(int)
    
    # 2. Label Prediction
    features_transformed = scaler.transform(features_raw)
    labels = xgb_model.predict(features_transformed).astype(int)
    
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
        print("\nErstelle Leaderboard Submission...")
        sub = generate_submission(df_leaderboard)
        sub.to_csv("submission_leaderboard_Group17.csv", index=False)
        print(f"Check: {sub['outlier'].sum()} Outliers gefunden.")
    except: pass

    try:
        df_final = pd.read_csv("D_test_final.csv")
        print("\nErstelle Final Submission...")
        sub = generate_submission(df_final)
        sub.to_csv("submission_final_Group17.csv", index=False)
    except: pass

if __name__ == "__main__":
    main()