"""
DDEC Training Script
Charge les conversations synthétiques, extrait les features, entraîne le classifieur.
"""

import json
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ddec.feature_extractor import extract_features, extract_trajectory_features
from ddec.classifier import DDECClassifier

DATA_PATH = "data/synthetic_conversations.json"
MODEL_PATH = "models/ddec_model.joblib"
LABEL_NAMES = ["verte", "jaune", "orange", "rouge"]


def load_and_extract(data_path: str):
    """Charge les conversations et extrait les features de trajectoire."""
    with open(data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Conversations chargées : {len(conversations)}")

    X_list = []
    y_list = []

    for conv in conversations:
        messages = conv["messages"]
        label = conv["label"]
        label_name = conv["label_name"]

        # Extraire les features message par message, puis agréger
        msg_features = extract_features(messages)
        traj_features = extract_trajectory_features(msg_features)

        X_list.append(traj_features)
        y_list.append(label)

        n_user = sum(1 for m in messages if m["role"] == "user")
        print(f"  [{label_name:6s}] {conv['id']:20s} - {n_user} messages user, "
              f"{len(traj_features)} features")

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\nShape X : {X.shape}")
    print(f"Shape y : {y.shape}")
    print(f"Distribution des labels : { {LABEL_NAMES[i]: int((y==i).sum()) for i in range(4)} }")

    return X, y


def print_separator(char="=", length=60):
    print(char * length)


def main():
    print_separator()
    print("  DDEC - Entraînement du classifieur")
    print_separator()

    # 1. Charger et extraire les features
    print("\n[1/4] Chargement des données et extraction des features...")
    X, y = load_and_extract(DATA_PATH)

    # 2. Validation croisée (stratifiée, k=4 pour avoir au moins 1 ex. par fold)
    print("\n[2/4] Validation croisée stratifiée (k=4)...")
    clf = DDECClassifier(n_estimators=200, random_state=42)

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        clf.pipeline, X, y, cv=cv, scoring="accuracy"
    )
    print(f"  Accuracy par fold : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Accuracy moyenne  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 3. Entraînement sur toutes les données
    print("\n[3/4] Entraînement sur l'ensemble complet...")
    clf.fit(X, y)

    y_pred = clf.predict(X)
    train_accuracy = accuracy_score(y, y_pred)

    print(f"\n  Accuracy (train) : {train_accuracy:.3f}")
    print()
    print_separator("-")
    print("  Rapport de classification (train)")
    print_separator("-")
    print(classification_report(
        y, y_pred,
        target_names=LABEL_NAMES,
        digits=3,
    ))

    print("  Matrice de confusion (train)")
    print_separator("-")
    cm = confusion_matrix(y, y_pred)
    header = f"{'':10s}" + " ".join(f"{n:>8s}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {LABEL_NAMES[i]:8s}" + " ".join(f"{v:8d}" for v in row)
        print(row_str)

    # 4. Feature importances
    print()
    print_separator("-")
    print("  Top 10 features les plus importantes")
    print_separator("-")
    rf = clf.pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top_idx, 1):
        if idx < len(clf.feature_names):
            fname = clf.feature_names[idx]
            print(f"  {rank:2d}. {fname:35s} {importances[idx]:.4f}")

    # 5. Sauvegarde
    print(f"\n[4/4] Sauvegarde du modèle dans {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    clf.save(MODEL_PATH)

    # 6. Test de rechargement
    print("\n  Test de rechargement du modèle...")
    clf2 = DDECClassifier.load(MODEL_PATH)
    test_conv = [
        {"role": "user", "content": "je sais plus pourquoi je continue. plus rien a de sens."},
        {"role": "assistant", "content": "Je suis là. Qu'est-ce qui se passe ?"},
        {"role": "user", "content": "jai pensé à en finir. jai un plan ce soir."},
    ]
    result = clf2.get_alert_level(test_conv)
    print(f"  Test alerte niveau critique : {result['label']} (confiance: {result['confidence']:.2f})")
    print(f"  Features dominantes : {result['dominant_features']}")

    print()
    print_separator()
    print("  Entraînement terminé avec succès !")
    print(f"  CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Train Accuracy : {train_accuracy:.3f}")
    print(f"  Modèle : {MODEL_PATH}")
    print_separator()


if __name__ == "__main__":
    main()
