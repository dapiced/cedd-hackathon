"""
DDEC Classifier
Classifie le niveau de dérive émotionnelle d'une conversation.
Utilise un GradientBoostingClassifier sklearn.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .feature_extractor import (
    extract_features,
    extract_trajectory_features,
    TRAJECTORY_FEATURE_NAMES,
)

LEVEL_LABELS = {0: "verte", 1: "jaune", 2: "orange", 3: "rouge"}

# Noms des features dominantes pour l'interprétabilité
_FEATURE_DISPLAY_NAMES = {
    "score_finalite_last": "Mots de finalité récents",
    "score_finalite_slope": "Tendance hausse finalité",
    "score_finalite_mean": "Niveau moyen finalité",
    "score_negatif_last": "Sentiment négatif récent",
    "score_negatif_slope": "Tendance sentiment négatif",
    "score_negatif_mean": "Niveau moyen négatif",
    "score_espoir_slope": "Baisse espoir",
    "score_espoir_last": "Espoir résiduel (faible)",
    "longueur_mots_slope": "Raccourcissement messages",
    "longueur_mots_last": "Longueur dernier message",
    "ratio_ponctuation_last": "Ponctuation faible",
    "delta_longueur_slope": "Variation longueur",
}


class DDECClassifier:
    """
    Classifieur de dérive émotionnelle conversationnelle.
    Encapsule un RandomForest sklearn avec preprocessing des features.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                random_state=random_state,
            )),
        ])
        self.is_fitted = False
        self.feature_names = TRAJECTORY_FEATURE_NAMES

    def _messages_to_vector(self, messages: list) -> np.ndarray:
        """Convertit une liste de messages en vecteur de features de trajectoire."""
        msg_features = extract_features(messages)
        traj_features = extract_trajectory_features(msg_features)
        return traj_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entraîne le classifieur.

        Args:
            X: array (n_samples, n_features) - features de trajectoire
            y: array (n_samples,) - labels 0/1/2/3
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités de chaque classe.

        Args:
            X: array (n_samples, n_features) ou (n_features,)

        Returns:
            array (n_samples, 4) - probabilités pour classes 0,1,2,3
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné. Appelez fit() d'abord.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit la classe (0/1/2/3)."""
        if not self.is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict(X)

    def get_alert_level(self, conversation_messages: list) -> dict:
        """
        Analyse une conversation complète et retourne le niveau d'alerte.

        Args:
            conversation_messages: liste de dicts {"role": str, "content": str}

        Returns:
            dict avec:
              - level (int 0-3)
              - label (str "verte"/"jaune"/"orange"/"rouge")
              - confidence (float 0-1)
              - probabilities (dict niveau→proba)
              - dominant_features (list de str)
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")

        # ── Règle 1 : contexte insuffisant (< 3 messages) ────────────────────
        messages_user = [m for m in conversation_messages if m["role"] == "user"]

        if len(messages_user) < 3:
            # Pas assez de contexte pour le ML
            # Applique UNIQUEMENT les règles lexicales de sécurité
            texte = " ".join([m["content"] for m in messages_user]).lower()

            mots_critique = ["disparaitre", "partir", "fardeau",
                             "inutile", "plus envie", "a quoi ca sert",
                             "personne", "jamais mieux", "tout arreter",
                             "mourir", "suicide", "me tuer"]

            mots_detresse = ["pleure", "larmes", "coeur gros",
                             "vide", "souffre", "seul", "peine",
                             "triste", "me sens mal", "ça va mal", "peur", "angoisse"]

            score_critique = sum(1 for m in mots_critique if m in texte)
            score_detresse = sum(1 for m in mots_detresse if m in texte)

            if score_critique >= 1:
                return {"level": 2, "label": "orange",
                        "confidence": 0.70,
                        "dominant_features": ["mot critique détecté"]}
            elif score_detresse >= 2:
                return {"level": 1, "label": "jaune",
                        "confidence": 0.65,
                        "dominant_features": ["mots de détresse détectés"]}
            else:
                return {"level": 0, "label": "verte",
                        "confidence": 0.80,
                        "dominant_features": ["contexte insuffisant — mode sécuritaire"]}
        # ─────────────────────────────────────────────────────────────────────

        # ── Règles de sécurité prioritaires (évaluées avant le ML) ──────────
        user_messages = [m["content"] for m in conversation_messages if m["role"] == "user"]
        texte_complet = " ".join(user_messages).lower()

        mots_detresse = ["pleure", "larmes", "triste", "me sens mal", "ça va mal",
                         "souffre", "coeur gros", "vide", "peine",
                         "seul", "fatigue", "peur", "angoisse"]

        mots_critique = ["disparaitre", "partir", "fardeau",
                         "inutile", "plus envie", "a quoi ca sert",
                         "personne", "jamais mieux", "tout arreter"]

        score_detresse = sum(1 for m in mots_detresse if m in texte_complet)
        score_critique = sum(1 for m in mots_critique if m in texte_complet)

        if score_critique >= 1:
            niveau_minimum = 2  # minimum Orange si mot critique détecté
        elif score_detresse >= 2:
            niveau_minimum = 1  # minimum Jaune si 2+ mots de détresse
        else:
            niveau_minimum = 0
        # ────────────────────────────────────────────────────────────────────

        vector = self._messages_to_vector(conversation_messages)
        probas = self.predict_proba(vector)[0]
        niveau_ml = int(np.argmax(probas))
        confidence = float(probas[niveau_ml])

        # Règle de seuil : si le classifieur n'est pas assez confiant, retourner jaune
        if confidence < 0.45:
            niveau_ml = 1  # jaune par défaut
            confidence = float(probas[niveau_ml])

        # Le classifieur ML ne peut pas descendre sous le niveau minimum de sécurité
        predicted_class = max(niveau_ml, niveau_minimum)

        # Features dominantes : importance × valeur normalisée
        clf = self.pipeline.named_steps["clf"]
        scaler = self.pipeline.named_steps["scaler"]
        importances = clf.feature_importances_

        # Valeurs après scaling pour la conversation courante
        scaled = scaler.transform(vector.reshape(1, -1))[0]
        # Score composite : importance × |valeur scalée|
        composite = importances * np.abs(scaled)

        top_indices = np.argsort(composite)[::-1][:5]
        dominant_features = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                fname = self.feature_names[idx]
                display = _FEATURE_DISPLAY_NAMES.get(fname, fname)
                dominant_features.append(display)

        return {
            "level": predicted_class,
            "label": LEVEL_LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": {
                LEVEL_LABELS[i]: float(probas[i]) for i in range(4)
            },
            "dominant_features": dominant_features[:3],
        }

    def save(self, path: str):
        """Sauvegarde le modèle avec joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        joblib.dump({
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }, path)
        print(f"Modèle sauvegardé : {path}")

    @classmethod
    def load(cls, path: str) -> "DDECClassifier":
        """Charge un modèle sauvegardé."""
        data = joblib.load(path)
        instance = cls()
        instance.pipeline = data["pipeline"]
        instance.is_fitted = data["is_fitted"]
        instance.feature_names = data["feature_names"]
        return instance
