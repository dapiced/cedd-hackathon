"""
CEDD Classifier
===============
Classifies the emotional drift level of a conversation.
Uses a GradientBoostingClassifier (sklearn).

Classifie le niveau de dérive émotionnelle d'une conversation.
Utilise un GradientBoostingClassifier sklearn.
"""

import os
import re
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
    "finality_score_last":    "Mots de finalité récents",
    "finality_score_slope":   "Tendance hausse finalité",
    "finality_score_mean":    "Niveau moyen finalité",
    "negative_score_last":    "Sentiment négatif récent",
    "negative_score_slope":   "Tendance sentiment négatif",
    "negative_score_mean":    "Niveau moyen négatif",
    "hope_score_slope":       "Baisse espoir",
    "hope_score_last":        "Espoir résiduel (faible)",
    "word_count_slope":       "Raccourcissement messages",
    "word_count_last":        "Longueur dernier message",
    "punctuation_ratio_last": "Ponctuation faible",
    "length_delta_slope":     "Variation longueur",
    "negation_score_mean":    "Négation d'état positif",
    "negation_score_last":    "Négations récentes",
    "negation_score_slope":   "Tendance hausse négation",
    "embedding_drift":        "Dérive sémantique entre messages",
    "crisis_similarity":      "Similarité avec langage de crise",
    "embedding_slope":        "Tendance sémantique directionnelle",
    "embedding_variance":     "Incohérence conversationnelle",
    "identity_conflict_score_mean":  "Conflit identitaire moyen",
    "identity_conflict_score_last":  "Conflit identitaire récent",
    "identity_conflict_score_slope": "Tendance conflit identitaire",
    "somatization_score_mean":       "Somatisation moyenne",
    "somatization_score_last":       "Somatisation récente",
    "somatization_score_slope":      "Tendance somatisation",
    "short_response_ratio":          "Ratio réponses courtes",
    "min_topic_coherence":           "Cohérence thématique minimale",
    "question_response_ratio":       "Ratio réponse aux questions",
}

# English / Anglais
_FEATURE_DISPLAY_NAMES_EN = {
    "finality_score_last":    "Recent finality words",
    "finality_score_slope":   "Rising finality trend",
    "finality_score_mean":    "Average finality level",
    "negative_score_last":    "Recent negative sentiment",
    "negative_score_slope":   "Negative sentiment trend",
    "negative_score_mean":    "Average negativity level",
    "hope_score_slope":       "Declining hope",
    "hope_score_last":        "Residual hope (low)",
    "word_count_slope":       "Shortening messages",
    "word_count_last":        "Last message length",
    "punctuation_ratio_last": "Low punctuation",
    "length_delta_slope":     "Length variation trend",
    "negation_score_mean":    "Negated positive states",
    "negation_score_last":    "Recent negated positive states",
    "negation_score_slope":   "Rising negation trend",
    "embedding_drift":        "Semantic drift between messages",
    "crisis_similarity":      "Similarity to crisis language",
    "embedding_slope":        "Directional semantic trend",
    "embedding_variance":     "Conversational incoherence",
    "identity_conflict_score_mean":  "Average identity conflict",
    "identity_conflict_score_last":  "Recent identity conflict",
    "identity_conflict_score_slope": "Rising identity conflict trend",
    "somatization_score_mean":       "Average somatization",
    "somatization_score_last":       "Recent somatization",
    "somatization_score_slope":      "Somatization trend",
    "short_response_ratio":          "Short response ratio",
    "min_topic_coherence":           "Minimum topic coherence",
    "question_response_ratio":       "Question response ratio",
}

_FEATURE_DISPLAY_NAMES = {
    "fr": _FEATURE_DISPLAY_NAMES_FR,
    "en": _FEATURE_DISPLAY_NAMES_EN,
}


def _keyword_match(keyword, text):
    """Match keyword in text. Uses word boundaries for single words,
    substring matching for multi-word phrases.
    Special handling for 'personne' to avoid French article false positives
    ('une personne' = a person ≠ 'personne ne m'écoute' = nobody listens).

    Correspondance de mot-clé dans le texte. Utilise les frontières de mots
    pour les mots simples, correspondance de sous-chaîne pour les phrases.
    """
    if ' ' in keyword:
        return keyword in text
    if keyword == "personne":
        # Match "personne" as "nobody" — skip when preceded by French article
        # Correspondre "personne" comme "personne/nobody" — ignorer après article
        return bool(re.search(
            r'(?<!une )(?<!la )(?<!cette )(?<!chaque )(?<!toute )\bpersonne\b',
            text
        ))
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


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
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        traj_features = extract_trajectory_features(msg_features, user_texts=user_texts,
                                                     messages=messages)
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

    def get_alert_level(self, conversation_messages: list, lang: str = "fr",
                         response_delay_s: float = None) -> dict:
        """
        Analyse a full conversation and return the alert level.
        Analyse une conversation complète et retourne le niveau d'alerte.

        Args:
            conversation_messages: list of {"role": str, "content": str}
            lang: display language for feature names ("fr" or "en")
            response_delay_s: seconds since last assistant message (runtime only, not ML)

        Returns:
            dict with / dict avec :
              - level (int 0-3)
              - label (str "green"/"yellow"/"orange"/"red")  — ML internal key
              - confidence (float 0-1)
              - probabilities (dict level_label → proba)
              - dominant_features (list of str, in the requested language)
              - delay_bumped (bool) — True if Gate 7 response delay bump was applied
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not trained. / Le modèle n'est pas encore entraîné."
            )

        display_names = _FEATURE_DISPLAY_NAMES.get(lang, _FEATURE_DISPLAY_NAMES_FR)

        # ── Rule 1: insufficient context (< 3 user messages) ─────────────────
        # Règle 1 : context insuffisant (< 3 messages utilisateur)
        messages_user = [m for m in conversation_messages if m["role"] == "user"]

        if len(messages_user) < 3:
            # Not enough context for ML — apply safety lexical rules only
            # Pas assez de context pour le ML — règles lexicales de sécurité uniquement
            text = " ".join([m["content"] for m in messages_user]).lower()

            crisis_words = [
                # FR crisis words / Mots de crise français
                "mourir", "suicide", "me tuer", "veux mourir", "envie de mourir",
                "arme", "pistolet", "fusil", "couteau", "me suicider", "me pendre",
                "en finir", "en finir avec tout", "finir avec tout",
                "mettre fin à mes jours", "mettre fin a mes jours",
                "plus envie de vivre", "veux plus vivre",
                # EN crisis words / Mots de crise anglais
                "die", "kill myself", "want to die", "end my life",
                "gun", "weapon", "shoot", "shooting", "bullet", "knife", "pistol",
                "rifle", "firearm", "hang myself", "overdose",
                "end it all", "end it tonight", "end it all tonight",
                "not worth living", "take my life", "end my pain",
                "want it to end", "make it stop forever",
            ]
            critical_words = [
                # FR critical words
                "disparaitre", "partir", "fardeau", "inutile", "plus envie",
                "a quoi ca sert", "personne", "jamais mieux", "tout arreter",
                # EN critical words
                "disappear", "burden", "useless", "don't want to", "what's the point",
                "no one", "never better", "stop everything",
            ]
            distress_words = [
                # FR distress words
                "pleure", "larmes", "coeur gros", "vide", "souffre", "seul",
                "peine", "triste", "me sens mal", "ça va mal", "peur", "angoisse",
                # EN distress words
                "crying", "tears", "heartbroken", "empty", "suffering", "alone",
                "hurting", "sad", "feel bad", "things are bad", "scared", "anxious",
            ]

            crisis_score = sum(1 for m in crisis_words if _keyword_match(m, text))
            critical_score = sum(1 for m in critical_words if _keyword_match(m, text))
            distress_score = sum(1 for m in distress_words if _keyword_match(m, text))

            if crisis_score >= 1:
                label_feat = "mot de crise détecté" if lang == "fr" else "crisis word detected"
                return {
                    "level": 3, "label": "red",
                    "confidence": 0.90,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                    "delay_bumped": False,
                }
            if critical_score >= 1:
                label_feat = "mot critique détecté" if lang == "fr" else "critical word detected"
                return {
                    "level": 2, "label": "orange",
                    "confidence": 0.70,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                    "delay_bumped": False,
                }
            elif distress_score >= 2:
                label_feat = "mots de détresse détectés" if lang == "fr" else "distress words detected"
                return {
                    "level": 1, "label": "yellow",
                    "confidence": 0.65,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                    "delay_bumped": False,
                }
            else:
                label_feat = (
                    "context insuffisant — mode sécuritaire"
                    if lang == "fr"
                    else "insufficient context — safe mode"
                )
                return {
                    "level": 0, "label": "green",
                    "confidence": 0.80,
                    "dominant_features": [label_feat],
                    "probabilities": {},
                    "delay_bumped": False,
                }

        # ── Safety rules evaluated before ML ─────────────────────────────────
        # Règles de sécurité prioritaires (évaluées avant le ML)
        user_messages = [m["content"] for m in conversation_messages if m["role"] == "user"]
        full_text = " ".join(user_messages).lower()

        crisis_words = [
            # FR crisis words / Mots de crise français
            "mourir", "suicide", "me tuer", "veux mourir", "envie de mourir",
            "arme", "pistolet", "fusil", "couteau", "me suicider", "me pendre",
            "en finir", "en finir avec tout", "finir avec tout",
            "mettre fin à mes jours", "mettre fin a mes jours",
            "plus envie de vivre", "veux plus vivre",
            # EN crisis words / Mots de crise anglais
            "die", "kill myself", "want to die", "end my life",
            "gun", "weapon", "shoot", "shooting", "bullet", "knife", "pistol",
            "rifle", "firearm", "hang myself", "overdose",
            "end it all", "end it tonight", "end it all tonight",
            "not worth living", "take my life", "end my pain",
            "want it to end", "make it stop forever",
        ]
        distress_words = [
            # FR
            "pleure", "larmes", "triste", "me sens mal", "ça va mal",
            "souffre", "coeur gros", "vide", "peine", "seul",
            "fatigue", "peur", "angoisse",
            # EN
            "crying", "tears", "sad", "feel bad", "things are bad",
            "suffering", "heartbroken", "empty", "hurting", "alone",
            "tired", "scared", "anxious",
        ]
        critical_words = [
            # FR
            "disparaitre", "partir", "fardeau", "inutile", "plus envie",
            "a quoi ca sert", "personne", "jamais mieux", "tout arreter",
            # EN
            "disappear", "burden", "useless", "don't want to", "what's the point",
            "no one", "never better", "stop everything",
        ]

        crisis_score = sum(1 for m in crisis_words if _keyword_match(m, full_text))
        distress_score = sum(1 for m in distress_words if _keyword_match(m, full_text))
        critical_score = sum(1 for m in critical_words if _keyword_match(m, full_text))

        if crisis_score >= 1:
            minimum_level = 3  # minimum Rouge if crisis word detected
        elif critical_score >= 1:
            minimum_level = 2  # minimum Orange if critical word detected
        elif distress_score >= 2:
            minimum_level = 1  # minimum Yellow if 2+ distress words
        else:
            minimum_level = 0

        # ── ML classification ─────────────────────────────────────────────────
        vector = self._messages_to_vector(conversation_messages)
        probas = self.predict_proba(vector)[0]
        ml_level = int(np.argmax(probas))
        confidence = float(probas[ml_level])

        # Confidence threshold: default to Yellow if too uncertain
        # Seuil de confiance : retourner Jaune si le classifieur n'est pas assez confiant
        if confidence < 0.45:
            ml_level = 1  # yellow / jaune by default
            confidence = float(probas[ml_level])

        # Short-conversation cap: ML trained on 12-msg conversations — with < 6 user
        # messages the trajectory features are too noisy to trust a RED prediction.
        # Cap it at Orange; RED via ML requires sufficient conversational context.
        # Also redistribute the capped probability so the display is consistent.
        if len(messages_user) < 6 and ml_level > 2:
            # Move the red probability mass into orange
            probas = probas.copy()
            probas[2] += probas[3]
            probas[3] = 0.0
            ml_level = 2
            confidence = float(probas[2])

        # ML level cannot go below the safety minimum
        # Le classifieur ML ne peut pas descendre sous le niveau minimum de sécurité
        predicted_class = max(ml_level, minimum_level)

        # ── Gate 7: Response delay bump (runtime only, not ML) ─────────────
        # Porte 7 : hausse si délai de réponse long (runtime seulement, pas ML)
        # 300s+ AND Yellow+ → bump +1 (cap at Red)
        # 120s+ AND Orange+ → bump +1 (cap at Red)
        # Green never bumped (delay alone doesn't create alert)
        delay_bumped = False
        if response_delay_s is not None and predicted_class >= 1:
            if response_delay_s >= 300 and predicted_class >= 1:
                predicted_class = min(predicted_class + 1, 3)
                delay_bumped = True
            elif response_delay_s >= 120 and predicted_class >= 2:
                predicted_class = min(predicted_class + 1, 3)
                delay_bumped = True

        # If safety rules raised the level above what ML predicted, the ML
        # probabilities are misleading — return empty so the override badge shows.
        safety_override = predicted_class > ml_level

        # ── Dominant features: importance × scaled value ──────────────────────
        # Features dominantes : importance × valeur normalisée
        clf_step = self.pipeline.named_steps["clf"]
        scaler   = self.pipeline.named_steps["scaler"]
        importances = clf_step.feature_importances_

        scaled    = scaler.transform(vector.reshape(1, -1))[0]
        composite = importances * np.abs(scaled)

        top_indices = np.argsort(composite)[::-1][:5]
        dominant_features = []
        feature_scores = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                fname = self.feature_names[idx]
                display = display_names.get(fname, fname)
                dominant_features.append(display)
                feature_scores.append({
                    "name": display,
                    "raw_name": fname,
                    "score": float(composite[idx]),
                })

        if safety_override:
            label_feat = "crisis word detected" if lang == "en" else "mot de crise détecté"
            return {
                "level": predicted_class,
                "label": LEVEL_LABELS[predicted_class],
                "confidence": 0.95,
                "probabilities": {},
                "dominant_features": [label_feat],
                "feature_scores": feature_scores,
                "delay_bumped": delay_bumped,
            }

        return {
            "level": predicted_class,
            "label": LEVEL_LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": {
                LEVEL_LABELS[i]: float(probas[i]) for i in range(4)
            },
            "dominant_features": dominant_features[:3],
            "feature_scores": feature_scores,
            "delay_bumped": delay_bumped,
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
