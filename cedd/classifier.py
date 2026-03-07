"""
CEDD Classifier
===============
Classifies the emotional drift level of a conversation.
Uses a GradientBoostingClassifier (sklearn).

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

# ML class labels (language-neutral internal keys)
# Libellés des classes (clés internes indépendantes de la langue)
LEVEL_LABELS = {0: "green", 1: "yellow", 2: "orange", 3: "red"}

# ── Human-readable feature display names ─────────────────────────────────────

# French / Français
_FEATURE_DISPLAY_NAMES_FR = {
    "score_finalite_last":       "Mots de finalité récents",
    "score_finalite_slope":      "Tendance hausse finalité",
    "score_finalite_mean":       "Niveau moyen finalité",
    "score_negatif_last":        "Sentiment négatif récent",
    "score_negatif_slope":       "Tendance sentiment négatif",
    "score_negatif_mean":        "Niveau moyen négatif",
    "score_espoir_slope":        "Baisse espoir",
    "score_espoir_last":         "Espoir résiduel (faible)",
    "longueur_mots_slope":       "Raccourcissement messages",
    "longueur_mots_last":        "Longueur dernier message",
    "ratio_ponctuation_last":    "Ponctuation faible",
    "delta_longueur_slope":      "Variation longueur",
}

# English / Anglais
_FEATURE_DISPLAY_NAMES_EN = {
    "score_finalite_last":       "Recent finality words",
    "score_finalite_slope":      "Rising finality trend",
    "score_finalite_mean":       "Average finality level",
    "score_negatif_last":        "Recent negative sentiment",
    "score_negatif_slope":       "Negative sentiment trend",
    "score_negatif_mean":        "Average negativity level",
    "score_espoir_slope":        "Declining hope",
    "score_espoir_last":         "Residual hope (low)",
    "longueur_mots_slope":       "Shortening messages",
    "longueur_mots_last":        "Last message length",
    "ratio_ponctuation_last":    "Low punctuation",
    "delta_longueur_slope":      "Length variation trend",
}

_FEATURE_DISPLAY_NAMES = {
    "fr": _FEATURE_DISPLAY_NAMES_FR,
    "en": _FEATURE_DISPLAY_NAMES_EN,
}


class CEDDClassifier:
    """
    Conversational emotional drift classifier.
    Wraps a GradientBoosting sklearn pipeline with feature preprocessing.

    Classifieur de dérive émotionnelle conversationnelle.
    Encapsule un GradientBoosting sklearn avec preprocessing des features.
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
        """
        Convert a message list into a trajectory feature vector.
        Convertit une liste de messages en vecteur de features de trajectoire.
        """
        msg_features = extract_features(messages)
        traj_features = extract_trajectory_features(msg_features)
        return traj_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier.
        Entraîne le classifieur.

        Args:
            X: array (n_samples, n_features) — trajectory features
            y: array (n_samples,) — labels 0/1/2/3
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.
        Retourne les probabilités de chaque classe.

        Args:
            X: array (n_samples, n_features) or (n_features,)

        Returns:
            array (n_samples, 4) — probabilities for classes 0, 1, 2, 3
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not trained yet. Call fit() first. / "
                "Le modèle n'est pas encore entraîné. Appelez fit() d'abord."
            )
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class (0/1/2/3).
        Prédit la classe (0/1/2/3).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not trained. / Le modèle n'est pas encore entraîné."
            )
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict(X)

    def get_alert_level(self, conversation_messages: list, lang: str = "fr") -> dict:
        """
        Analyse a full conversation and return the alert level.
        Analyse une conversation complète et retourne le niveau d'alerte.

        Args:
            conversation_messages: list of {"role": str, "content": str}
            lang: display language for feature names ("fr" or "en")

        Returns:
            dict with / dict avec :
              - level (int 0-3)
              - label (str "verte"/"jaune"/"orange"/"rouge")  — ML internal key
              - confidence (float 0-1)
              - probabilities (dict level_label → proba)
              - dominant_features (list of str, in the requested language)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not trained. / Le modèle n'est pas encore entraîné."
            )

        display_names = _FEATURE_DISPLAY_NAMES.get(lang, _FEATURE_DISPLAY_NAMES_FR)

        # ── Rule 1: insufficient context (< 3 user messages) ─────────────────
        # Règle 1 : contexte insuffisant (< 3 messages utilisateur)
        messages_user = [m for m in conversation_messages if m["role"] == "user"]

        if len(messages_user) < 3:
            # Not enough context for ML — apply safety lexical rules only
            # Pas assez de contexte pour le ML — règles lexicales de sécurité uniquement
            texte = " ".join([m["content"] for m in messages_user]).lower()

            mots_urgence = [
                # FR crisis words
                "mourir", "suicide", "me tuer", "veux mourir", "envie de mourir",
                "arme", "pistolet", "fusil", "couteau", "me suicider", "me pendre",
                # EN crisis words
                "die", "kill myself", "want to die", "end my life",
                "gun", "weapon", "shoot", "shooting", "bullet", "knife", "pistol",
                "rifle", "firearm", "hang myself", "overdose",
            ]
            mots_critique = [
                # FR critical words
                "disparaitre", "partir", "fardeau", "inutile", "plus envie",
                "a quoi ca sert", "personne", "jamais mieux", "tout arreter",
                # EN critical words
                "disappear", "burden", "useless", "don't want to", "what's the point",
                "no one", "never better", "stop everything",
            ]
            mots_detresse = [
                # FR distress words
                "pleure", "larmes", "coeur gros", "vide", "souffre", "seul",
                "peine", "triste", "me sens mal", "ça va mal", "peur", "angoisse",
                # EN distress words
                "crying", "tears", "heartbroken", "empty", "suffering", "alone",
                "hurting", "sad", "feel bad", "things are bad", "scared", "anxious",
            ]

            score_urgence = sum(1 for m in mots_urgence if m in texte)
            score_critique = sum(1 for m in mots_critique if m in texte)
            score_detresse = sum(1 for m in mots_detresse if m in texte)

            if score_urgence >= 1:
                label_feat = "mot de crise détecté" if lang == "fr" else "crisis word detected"
                return {
                    "level": 3, "label": "red",
                    "confidence": 0.90,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                }
            if score_critique >= 1:
                label_feat = "mot critique détecté" if lang == "fr" else "critical word detected"
                return {
                    "level": 2, "label": "orange",
                    "confidence": 0.70,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                }
            elif score_detresse >= 2:
                label_feat = "mots de détresse détectés" if lang == "fr" else "distress words detected"
                return {
                    "level": 1, "label": "yellow",
                    "confidence": 0.65,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                }
            else:
                label_feat = (
                    "contexte insuffisant — mode sécuritaire"
                    if lang == "fr"
                    else "insufficient context — safe mode"
                )
                return {
                    "level": 0, "label": "green",
                    "confidence": 0.80,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                }

        # ── Safety rules evaluated before ML ─────────────────────────────────
        # Règles de sécurité prioritaires (évaluées avant le ML)
        user_messages = [m["content"] for m in conversation_messages if m["role"] == "user"]
        texte_complet = " ".join(user_messages).lower()

        mots_urgence = [
            # FR crisis words
            "mourir", "suicide", "me tuer", "veux mourir", "envie de mourir",
            "arme", "pistolet", "fusil", "couteau", "me suicider", "me pendre",
            # EN crisis words
            "die", "kill myself", "want to die", "end my life",
            "gun", "weapon", "shoot", "shooting", "bullet", "knife", "pistol",
            "rifle", "firearm", "hang myself", "overdose",
        ]
        mots_detresse = [
            # FR
            "pleure", "larmes", "triste", "me sens mal", "ça va mal",
            "souffre", "coeur gros", "vide", "peine", "seul",
            "fatigue", "peur", "angoisse",
            # EN
            "crying", "tears", "sad", "feel bad", "things are bad",
            "suffering", "heartbroken", "empty", "hurting", "alone",
            "tired", "scared", "anxious",
        ]
        mots_critique = [
            # FR
            "disparaitre", "partir", "fardeau", "inutile", "plus envie",
            "a quoi ca sert", "personne", "jamais mieux", "tout arreter",
            # EN
            "disappear", "burden", "useless", "don't want to", "what's the point",
            "no one", "never better", "stop everything",
        ]

        score_urgence = sum(1 for m in mots_urgence if m in texte_complet)
        score_detresse = sum(1 for m in mots_detresse if m in texte_complet)
        score_critique = sum(1 for m in mots_critique if m in texte_complet)

        if score_urgence >= 1:
            niveau_minimum = 3  # minimum Rouge if crisis word detected
        elif score_critique >= 1:
            niveau_minimum = 2  # minimum Orange if critical word detected
        elif score_detresse >= 2:
            niveau_minimum = 1  # minimum Yellow if 2+ distress words
        else:
            niveau_minimum = 0

        # ── ML classification ─────────────────────────────────────────────────
        vector = self._messages_to_vector(conversation_messages)
        probas = self.predict_proba(vector)[0]
        niveau_ml = int(np.argmax(probas))
        confidence = float(probas[niveau_ml])

        # Confidence threshold: default to Yellow if too uncertain
        # Seuil de confiance : retourner Jaune si le classifieur n'est pas assez confiant
        if confidence < 0.45:
            niveau_ml = 1  # yellow / jaune by default
            confidence = float(probas[niveau_ml])

        # Short-conversation cap: ML trained on 12-msg conversations — with < 6 user
        # messages the trajectory features are too noisy to trust a RED prediction.
        # Cap it at Orange; RED via ML requires sufficient conversational context.
        # Also redistribute the capped probability so the display is consistent.
        if len(messages_user) < 6 and niveau_ml > 2:
            # Move the red probability mass into orange
            probas = probas.copy()
            probas[2] += probas[3]
            probas[3] = 0.0
            niveau_ml = 2
            confidence = float(probas[2])

        # ML level cannot go below the safety minimum
        # Le classifieur ML ne peut pas descendre sous le niveau minimum de sécurité
        predicted_class = max(niveau_ml, niveau_minimum)

        # If safety rules raised the level above what ML predicted, the ML
        # probabilities are misleading — return empty so the override badge shows.
        safety_override = predicted_class > niveau_ml

        # ── Dominant features: importance × scaled value ──────────────────────
        # Features dominantes : importance × valeur normalisée
        clf_step = self.pipeline.named_steps["clf"]
        scaler   = self.pipeline.named_steps["scaler"]
        importances = clf_step.feature_importances_

        scaled    = scaler.transform(vector.reshape(1, -1))[0]
        composite = importances * np.abs(scaled)

        top_indices = np.argsort(composite)[::-1][:5]
        dominant_features = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                fname = self.feature_names[idx]
                display = display_names.get(fname, fname)
                dominant_features.append(display)

        if safety_override:
            label_feat = "crisis word detected" if lang == "en" else "mot de crise détecté"
            return {
                "level": predicted_class,
                "label": LEVEL_LABELS[predicted_class],
                "confidence": 0.95,
                "probabilities": {},
                "dominant_features": [label_feat],
            }

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
        """
        Save the model with joblib.
        Sauvegarde le modèle avec joblib.
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pipeline":      self.pipeline,
            "is_fitted":     self.is_fitted,
            "feature_names": self.feature_names,
        }, path)
        print(f"Model saved / Modèle sauvegardé : {path}")

    @classmethod
    def load(cls, path: str) -> "CEDDClassifier":
        """
        Load a saved model.
        Charge un modèle sauvegardé.
        """
        data = joblib.load(path)
        instance = cls()
        instance.pipeline      = data["pipeline"]
        instance.is_fitted     = data["is_fitted"]
        instance.feature_names = data["feature_names"]
        return instance
