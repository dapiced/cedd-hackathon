"""
CEDD Feature Extractor
======================
Extracts lexical and structural features from conversation messages.
Uses only numpy and re — no LLM required.

Extraction de features lexicales et structurelles des messages d'une conversation.
Utilise uniquement numpy et re, aucun LLM.
"""

import re
import numpy as np

# ── Finality / distress lexicon (FR + EN) ────────────────────────────────────
# Champ lexical de finalité / détresse (français + anglais)
FINALITY_WORDS = [
    # French / Français
    "jamais", "toujours", "plus rien", "personne", "inutile", "fardeau",
    "sans espoir", "impossible", "à quoi bon", "rien ne sert", "trop tard",
    "terminé", "fini", "dernière", "dernier", "dernier fois",
    "disparaître", "en finir", "mourir", "mort", "suicide", "tuer",
    "adieu", "au revoir pour toujours", "plus jamais",
    "à bout", "épuisé", "vide", "seul au monde", "abandonné",
    "partir", "à quoi ça sert", "plus envie",
    # English / Anglais
    "never", "nothing left", "no one", "useless", "burden",
    "hopeless", "what's the point", "nothing matters", "too late",
    "finished", "done", "last time", "disappear", "end it",
    "die", "death", "suicide", "kill myself",
    "goodbye forever", "never again", "at the end", "exhausted",
    "empty", "all alone", "abandoned", "leave", "what's the use",
    "don't want to anymore", "give up",
]

# ── Hope / resources lexicon (FR + EN) ───────────────────────────────────────
# Champ lexical d'espoir / ressources (français + anglais)
HOPE_WORDS = [
    # French / Français
    "demain", "essayer", "peut-être", "améliorer", "mieux", "espoir",
    "changer", "avenir", "bientôt", "progresser", "aider", "soutien",
    "ensemble", "solution", "possibilité", "essai", "recommencer",
    "guérir", "récupérer", "avancer", "continuer", "tenir",
    "famille", "ami", "rire",
    # English / Anglais
    "tomorrow", "try", "maybe", "better", "hope",
    "change", "future", "soon", "progress", "help", "support",
    "together", "solution", "possibility", "attempt", "start again",
    "heal", "recover", "move forward", "continue", "hold on",
    "family", "friend", "laugh",
]

# ── Negative sentiment words (FR + EN) ───────────────────────────────────────
# Mots négatifs courants (français + anglais)
NEGATIVE_WORDS = [
    # French / Français
    "me sens mal", "ça va mal", "tout va mal", "me sens pas bien", "pas bien du tout",
    "pas", "jamais", "rien", "personne", "nul", "mauvais",
    "terrible", "horrible", "triste", "seul", "perdu", "inutile",
    "fatigué", "épuisé", "peur", "anxieux", "inquiet", "déprimé",
    "sombre", "noir", "vide", "fardeau", "honte", "coupable",
    "échec", "raté", "déchet", "merdique", "impossible", "désespéré",
    "pleure", "larmes", "coeur gros", "souffre", "peine",
    # English / Anglais
    "feel bad", "things are bad", "everything's wrong", "not feeling well", "not well at all",
    "not", "never", "nothing", "worthless", "terrible", "horrible",
    "sad", "alone", "lost", "tired", "scared", "anxious", "worried", "depressed",
    "dark", "empty", "ashamed", "guilty",
    "failure", "failed", "awful", "impossible", "hopeless",
    "crying", "tears", "heartbroken", "suffering", "pain",
]

# ── Physical context words (FR + EN) — reduces false positives ───────────────
# Contexte physique (français + anglais) : réduit le score négatif de 50 %
PHYSICAL_CONTEXT_WORDS = [
    # French / Français
    "dos", "tête", "ventre", "genou", "bras", "jambe", "gorge", "estomac",
    "mal au", "j'ai mal",
    # English / Anglais
    "back", "head", "stomach", "knee", "arm", "leg", "throat", "belly",
    "pain in my", "i hurt", "sore",
]


# ── Low-level feature functions ───────────────────────────────────────────────

def _count_words(text: str) -> int:
    """Count the number of words. / Compte le nombre de mots."""
    return len(text.split())


def _punctuation_ratio(text: str) -> float:
    """
    Punctuation-to-character ratio.
    Ratio de ponctuation par rapport au nombre total de caractères.
    """
    if len(text) == 0:
        return 0.0
    punct = len(re.findall(r'[.,!?;:…\-—]', text))
    return punct / len(text)


def _has_question(text: str) -> float:
    """
    1.0 if the message contains a question mark, 0.0 otherwise.
    1.0 si le message contient une question, 0.0 sinon.
    """
    return 1.0 if '?' in text else 0.0


def _negative_score(text: str) -> float:
    """
    Negative-sentiment score: ratio of negative words to total words.
    Halved when a physical context is detected in the same message.

    Score de sentiment négatif : ratio de mots négatifs / total de mots.
    Réduit de 50 % si un contexte physique est détecté dans le même message.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0

    multi_word_neg = [neg for neg in NEGATIVE_WORDS if ' ' in neg]
    single_word_neg = [neg for neg in NEGATIVE_WORDS if ' ' not in neg]

    count = sum(1 for phrase in multi_word_neg if phrase in text_lower)
    count += sum(1 for w in words if any(neg in w for neg in single_word_neg))

    score = count / len(words)

    # Physical context → halve the score / Contexte physique → divise par 2
    if any(phys in text_lower for phys in PHYSICAL_CONTEXT_WORDS):
        score *= 0.5

    return score


def _finality_score(text: str) -> float:
    """
    Finality / distress lexical score.
    Score du champ lexical de finalité / détresse.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0
    count = 0
    for phrase in FINALITY_WORDS:
        if ' ' in phrase:
            if phrase in text_lower:
                count += 1
        else:
            count += sum(1 for w in words if phrase in w)
    return min(count / max(len(words), 1), 1.0)


def _hope_score(text: str) -> float:
    """
    Hope / resources lexical score.
    Score du champ lexical d'espoir / ressources.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0
    count = 0
    for phrase in HOPE_WORDS:
        if ' ' in phrase:
            if phrase in text_lower:
                count += 1
        else:
            count += sum(1 for w in words if phrase in w)
    return min(count / max(len(words), 1), 1.0)


# ── Main feature extraction ───────────────────────────────────────────────────

def extract_features(messages: list) -> np.ndarray:
    """
    Extract a feature vector for each USER message in a conversation.
    Extrait un vecteur de features pour chaque message USER d'une conversation.

    Args:
        messages: list of dicts {"role": str, "content": str}
                  liste de dicts {"role": str, "content": str}

    Returns:
        np.ndarray of shape (n_user_messages, n_features)
        Features per message / Features par message :
          [0] word count / longueur (nb mots)
          [1] punctuation ratio / ratio de ponctuation
          [2] question presence 0/1 / présence de question (0/1)
          [3] negative sentiment score / score sentiment négatif
          [4] finality lexical score / score champ lexical finalité
          [5] hope lexical score / score champ lexical espoir
          [6] length delta vs previous message / delta longueur vs message précédent
    """
    user_messages = [m for m in messages if m["role"] == "user"]

    if not user_messages:
        return np.zeros((1, 7))

    feature_list = []
    prev_length = None

    for msg in user_messages:
        text = msg["content"]
        n_words = _count_words(text)

        if prev_length is None:
            delta = 0.0
        else:
            # Relative length change / Variation relative de longueur
            delta = (n_words - prev_length) / max(prev_length, 1)
        prev_length = n_words

        feat = np.array([
            float(n_words),
            _punctuation_ratio(text),
            _has_question(text),
            _negative_score(text),
            _finality_score(text),
            _hope_score(text),
            delta,
        ])
        feature_list.append(feat)

    return np.array(feature_list)


def extract_trajectory_features(features_array: np.ndarray) -> np.ndarray:
    """
    Compute trajectory features from a (n_messages, n_features) matrix.
    Aggregates the temporal dynamics of each base feature.

    Calcule des features de trajectoire globale à partir d'une matrice
    (n_messages, n_features) décrivant la dynamique de la conversation.

    Returns:
        1D np.ndarray:  [mean, std, slope, last, max, min] × n_base_features
        shape: (n_features * 6,)
    """
    if features_array.ndim == 1:
        features_array = features_array.reshape(1, -1)

    n_msgs, n_feat = features_array.shape
    trajectory_feats = []

    x = np.arange(n_msgs, dtype=float)
    x_norm = x / max(x[-1], 1) if n_msgs > 1 else x

    for f in range(n_feat):
        col = features_array[:, f]
        mean_val = float(np.mean(col))
        std_val  = float(np.std(col))
        last_val = float(col[-1])
        max_val  = float(np.max(col))
        min_val  = float(np.min(col))

        # Linear trend slope / Pente de tendance (régression linéaire simple)
        if n_msgs > 1:
            slope = float(np.polyfit(x_norm, col, 1)[0])
        else:
            slope = 0.0

        trajectory_feats.extend([mean_val, std_val, slope, last_val, max_val, min_val])

    return np.array(trajectory_feats)


# ── Feature name lists ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "word_count",
    "punctuation_ratio",
    "question_presence",
    "negative_score",
    "finality_score",
    "hope_score",
    "length_delta",
]

TRAJECTORY_FEATURE_NAMES = [
    f"{feat}_{stat}"
    for feat in FEATURE_NAMES
    for stat in ["mean", "std", "slope", "last", "max", "min"]
]
