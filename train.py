"""
CEDD Training Script / Script d'entraînement CEDD
==================================================
Loads synthetic conversations, extracts features, trains the classifier.
Charge les conversations synthétiques, extrait les features, entraîne le classifieur.
"""

import json
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add root directory to path / Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cedd.feature_extractor import extract_features, extract_trajectory_features
from cedd.classifier import CEDDClassifier

DATA_PATH   = "data/synthetic_conversations.json"
MODEL_PATH  = "models/cedd_model.joblib"
LABEL_NAMES = ["green", "yellow", "orange", "red"]


def load_and_extract(data_path: str):
    """
    Load conversations and extract trajectory features.
    Charge les conversations et extrait les features de trajectoire.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Conversations loaded / chargées : {len(conversations)}")

    X_list = []
    y_list = []

    for conv in conversations:
        messages    = conv["messages"]
        label       = conv["label"]
        label_name  = conv["label_name"]

        # Extract per-message features, then aggregate into trajectory
        # Extraire les features message par message, puis agréger
        msg_features  = extract_features(messages)
        traj_features = extract_trajectory_features(msg_features)

        X_list.append(traj_features)
        y_list.append(label)

        n_user = sum(1 for m in messages if m["role"] == "user")
        print(f"  [{label_name:6s}] {conv['id']:20s} — {n_user} user msgs, "
              f"{len(traj_features)} features")

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\nX shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Label distribution / Distribution des labels : "
          f"{ {LABEL_NAMES[i]: int((y == i).sum()) for i in range(4)} }")

    return X, y


def print_separator(char="=", length=60):
    print(char * length)


def main():
    print_separator()
    print("  CEDD — Classifier Training / Entraînement du classifieur")
    print_separator()

    # 1. Load data and extract features / Charger les données et extraire les features
    print("\n[1/4] Loading data & extracting features / Chargement et extraction...")
    X, y = load_and_extract(DATA_PATH)

    # 2. Stratified cross-validation / Validation croisée stratifiée
    print("\n[2/4] Stratified cross-validation (k=4)...")
    clf = CEDDClassifier(n_estimators=200, random_state=42)

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf.pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"  Accuracy per fold / par fold : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean accuracy / Accuracy moy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 3. Full training / Entraînement sur l'ensemble complet
    print("\n[3/4] Training on full dataset / Entraînement sur l'ensemble complet...")
    clf.fit(X, y)

    y_pred         = clf.predict(X)
    train_accuracy = accuracy_score(y, y_pred)

    print(f"\n  Train accuracy : {train_accuracy:.3f}")
    print()
    print_separator("-")
    print("  Classification report / Rapport de classification (train)")
    print_separator("-")
    print(classification_report(y, y_pred, target_names=LABEL_NAMES, digits=3))

    print("  Confusion matrix / Matrice de confusion (train)")
    print_separator("-")
    cm     = confusion_matrix(y, y_pred)
    header = f"{'':10s}" + " ".join(f"{n:>8s}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {LABEL_NAMES[i]:8s}" + " ".join(f"{v:8d}" for v in row)
        print(row_str)

    # 4. Feature importances / Importance des features
    print()
    print_separator("-")
    print("  Top 10 most important features / Top 10 features les plus importantes")
    print_separator("-")
    rf          = clf.pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top_idx, 1):
        if idx < len(clf.feature_names):
            fname = clf.feature_names[idx]
            print(f"  {rank:2d}. {fname:35s} {importances[idx]:.4f}")

    # 5. Save model / Sauvegarde
    print(f"\n[4/4] Saving model / Sauvegarde du modèle → {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    clf.save(MODEL_PATH)

    # 6. Reload test / Test de rechargement
    print("\n  Reload test / Test de rechargement...")
    clf2 = CEDDClassifier.load(MODEL_PATH)
    test_conv = [
        {"role": "user",      "content": "je sais plus pourquoi je continue. plus rien a de sens."},
        {"role": "assistant", "content": "Je suis là. Qu'est-ce qui se passe ?"},
        {"role": "user",      "content": "jai pensé à en finir. jai un plan ce soir."},
    ]
    result = clf2.get_alert_level(test_conv, lang="fr")
    print(f"  Crisis test / Test alerte critique : {result['label']} (confidence: {result['confidence']:.2f})")
    print(f"  Dominant features : {result['dominant_features']}")

    # English crisis test / Test de crise en anglais
    test_conv_en = [
        {"role": "user",      "content": "I don't know why I keep going. nothing makes sense anymore."},
        {"role": "assistant", "content": "I'm here. What's going on?"},
        {"role": "user",      "content": "I've thought about ending it. I have a plan for tonight."},
    ]
    result_en = clf2.get_alert_level(test_conv_en, lang="en")
    print(f"  EN crisis test   : {result_en['label']} (confidence: {result_en['confidence']:.2f})")
    print(f"  Dominant features: {result_en['dominant_features']}")

    print()
    print_separator()
    print("  Training complete! / Entraînement terminé avec succès !")
    print(f"  CV Accuracy    : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Train Accuracy : {train_accuracy:.3f}")
    print(f"  Model / Modèle : {MODEL_PATH}")
    print_separator()


if __name__ == "__main__":
    main()
