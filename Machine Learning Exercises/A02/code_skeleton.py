import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --- GLOBALE VARIABLEN ---
# Wir speichern das trainierte "Gehirn" hier, damit die predict-Funktion darauf zugreifen kann.
gmm = None
final_clf = None
tau = -np.inf # Startwert

def train_my_logic():
    """
    Hier steckt DEIN Code drin (Schritt 1 bis 3 aus deinem Notebook).
    Diese Funktion wird einmal ganz am Anfang aufgerufen.
    """
    global gmm, final_clf, tau
    
    print("--- Start Training (Group17 Logic) ---")
    
    # 1. Daten laden (Pfade müssen stimmen!)
    try:
        df_train = pd.read_csv("D.csv")
        df_out_ref = pd.read_csv("D_out.csv")
    except FileNotFoundError:
        print("FEHLER: D.csv oder D_out.csv fehlen! Skript bricht ab.")
        return

    # Features extrahieren
    X_train = df_train.drop(columns=['id', 'label'], errors='ignore')
    y_train = df_train['label']
    X_out_ref = df_out_ref.drop(columns=['id'], errors='ignore')

    print("--- Step 1: Training Outlier Detector (GMM) ---")
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm.fit(X_train)

    # Scores berechnen
    scores_train = gmm.score_samples(X_train)
    scores_out_ref = gmm.score_samples(X_out_ref)

    # Threshold Tau bestimmen (99. Perzentil der Outlier)
    tau = np.percentile(scores_out_ref, 99)
    print(f"Determined Threshold Tau: {tau:.4f}")

    # (Optional: Plotting Code könnte hier stehen, lassen wir weg für reine Berechnungsspeed)

    print("--- Step 2: Cleaning Training Data ---")
    mask_clean = scores_train >= tau
    X_train_clean = X_train[mask_clean]
    y_train_clean = y_train[mask_clean]
    
    n_removed = len(X_train) - len(X_train_clean)
    print(f"Removed {n_removed} noisy samples.")

    print("--- Step 3: Training Final Classifier (Random Forest) ---")
    final_clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    final_clf.fit(X_train_clean, y_train_clean)
    print("Training abgeschlossen.\n")


def predict(X_test):
    """
    Diese Funktion wird vom Skeleton für jeden Datensatz (Leaderboard & Final) aufgerufen.
    Sie ersetzt deine 'predict_pipeline'.
    """
    global gmm, final_clf, tau
    
    # Sicherstellen, dass trainiert wurde
    if gmm is None or final_clf is None:
        raise ValueError("Modelle nicht trainiert!")

    # Features vorbereiten (ID wegwerfen)
    features = X_test.drop(columns=["id"], errors="ignore")
    
    # 1. Outlier Detection (GMM Logik)
    test_scores = gmm.score_samples(features)
    # Dein Code: score < tau heißt Outlier (1), sonst Normal (0)
    outliers = (test_scores < tau).astype(int)
    
    # 2. Classification (Random Forest)
    labels = final_clf.predict(features)
    
    return labels, outliers


def generate_submission(test_data):
    # DIESE FUNKTION NICHT ÄNDERN (Vom Skeleton vorgegeben)
    label_predictions, outlier_predictions = predict(test_data)
    
    submission_df = pd.DataFrame({ 
        "id": test_data["id"],
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df


def main():
    # 1. ZUERST: Dein Training ausführen
    train_my_logic()
    
    # 2. Leaderboard Submission (Skeleton Logik)
    try:
        df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
        submission_df = generate_submission(df_leaderboard)
        # Group17 Name eingetragen
        submission_df.to_csv("submission_leaderboard_Group17.csv", index=False)
        print("Datei 'submission_leaderboard_Group17.csv' erstellt.")
    except FileNotFoundError:
        print("D_test_leaderboard.csv nicht gefunden, überspringe...")

    # 3. Final Submission (Skeleton Logik)
    try:
        df_final = pd.read_csv("D_test_final.csv")
        submission_df = generate_submission(df_final)
        # Group17 Name eingetragen
        submission_df.to_csv("submission_final_Group17.csv", index=False)
        print("Datei 'submission_final_Group17.csv' erstellt.")
    except FileNotFoundError:
        print("D_test_final.csv nicht gefunden, überspringe...")

if __name__ == "__main__":
    main()