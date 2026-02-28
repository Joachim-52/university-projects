import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline

# --- GLOBALE MODELLE ---
gmm_model = None
clf_model = None
tau = None

def train_my_logic():
    global gmm_model, clf_model, tau
    
    print("--- Start Training (Optimiertes Stacking Modell) ---")
    
    # 1. Daten laden
    try:
        df_train = pd.read_csv("D.csv")
        df_out_ref = pd.read_csv("D_out.csv")
    except FileNotFoundError:
        print("FEHLER: Dateien (D.csv oder D_out.csv) fehlen!")
        return

    # Features trennen
    X_train_raw = df_train.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    X_out_raw = df_out_ref.drop(columns=['id', 'label', 'outlier'], errors='ignore')
    y_train = df_train['label']

    # --- TEIL 1: OUTLIER DETECTION ---
    print("1. Trainiere Outlier Detector (GMM Raw)...")
    gmm_model = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm_model.fit(X_train_raw)
    
    # Threshold Tuning: 99. Perzentil der bekannten Outlier
    scores_out = gmm_model.score_samples(X_out_raw)
    tau = np.percentile(scores_out, 99)
    print(f"   -> Threshold Tau gesetzt auf: {tau:.4f}")

    # --- TEIL 2: KLASSIFIKATION ---
    print("2. Trainiere Stacking Classifier (RF + XGB + KNN)...")

    # A. Random Forest
    pipe_rf = RandomForestClassifier(
        n_estimators=500, 
        min_samples_split=5, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    # B. XGBoost
    pipe_xgb = Pipeline([
        ('scaler', PowerTransformer(method='yeo-johnson')),
        ('clf', XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # C. KNN
    pipe_knn = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=9))
    ])
    
    estimators = [
        ('rf', pipe_rf),
        ('xgb', pipe_xgb),
        ('knn', pipe_knn)
    ]
    
    clf_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    
    print(f"   -> Starte Training auf {len(X_train_raw)} Samples...")
    clf_model.fit(X_train_raw, y_train)
    print("Training abgeschlossen.")


def predict(X_test):
    global gmm_model, clf_model, tau
    
    if gmm_model is None or clf_model is None:
        raise ValueError("Modell nicht trainiert!")

    features_raw = X_test.drop(columns=["id"], errors="ignore")
    
    # 1. Outlier Prediction
    scores = gmm_model.score_samples(features_raw)
    outliers = (scores < tau).astype(int)
    
    # 2. Label Prediction
    labels = clf_model.predict(features_raw).astype(int)
    
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
        sub.to_csv("submission_leaderboard_Group17Stacking.csv", index=False)
        print(f"Check: {sub['outlier'].sum()} Outliers gefunden.")
    except: 
        print("D_test_leaderboard.csv nicht gefunden.")

    try:
        df_final = pd.read_csv("D_test_final.csv")
        print("\nErstelle Final Submission...")
        sub = generate_submission(df_final)
        sub.to_csv("submission_final_Group17Stacking.csv", index=False)
        print("Final Submission erstellt.")
    except: 
        print("D_test_final.csv nicht gefunden.")

if __name__ == "__main__":
    main()