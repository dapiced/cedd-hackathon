"""
CEDD Feature Extractor
======================
Extracts lexical and structural features from conversation messages.
Uses only numpy and re — no LLM required.

Extraction de features lexicales et structurelles des messages d'une conversation.
Utilise uniquement numpy et re, aucun LLM.
"""

import re
import logging
import numpy as np
from sklearn.decomposition import PCA

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

# ── Physical context words (FR + EN) — used by somatization detection ────────
# Contexte physique (français + anglais) : utilisé par la détection de somatisation
PHYSICAL_CONTEXT_WORDS = [
    # French / Français
    "dos", "tête", "ventre", "genou", "bras", "jambe", "gorge", "estomac",
    "mal au", "j'ai mal",
    # English / Anglais
    "back", "head", "stomach", "knee", "arm", "leg", "throat", "belly",
    "pain in my", "i hurt", "sore",
]

# ── Identity-conflict lexicon (FR + EN) — 2SLGBTQ+ & cultural identity ───────
# Champ lexical de conflit identitaire (français + anglais)
IDENTITY_CONFLICT_WORDS = [
    # French / Français (multi-word phrases)
    "ma famille ne m'accepte pas", "je dois faire semblant",
    "je dois me cacher", "honte de ce que je suis",
    "pas comme les autres", "cacher qui je suis",
    "sortir du placard",
    # English / Anglais (multi-word phrases)
    "my family won't accept me", "i can't be who i am",
    "i have to pretend", "i have to hide",
    "ashamed of who i am", "hide who i am",
    # Single words (both languages) / Mots simples (deux langues)
    "rejeté", "rejected", "closeted", "coming out",
]

# ── Somatization emotional co-occurrence words ───────────────────────────────
# Mots émotionnels co-occurrents avec plaintes physiques (somatisation)
SOMATIZATION_EMOTIONAL_WORDS = [
    # French / Français
    "triste", "seul", "vide", "anxieux", "déprimé", "peur", "angoisse",
    "pleure", "désespéré", "souffre", "mort", "mourir", "fardeau",
    "abandonné", "inutile", "honte",
    # English / Anglais
    "sad", "alone", "empty", "anxious", "depressed", "scared", "crying",
    "hopeless", "suffering", "death", "die", "burden", "abandoned",
    "worthless", "ashamed",
]

# ── Negation patterns (FR + EN) ───────────────────────────────────────────────
# Patterns de négation : détectent les structures "ne...pas bien", "can't cope", etc.
NEGATION_PATTERNS_FR = [
    r"ne\s+\w+\s+pas\s+(bien|mieux|ok|correct)",    # ne ... pas bien/mieux
    r"ne\s+\w+\s+plus",                               # ne ... plus
    r"ne\s+\w+\s+jamais",                              # ne ... jamais
    r"plus\s+envie",                                   # plus envie
    r"pas\s+(bien|ok|correct|capable)",                # pas bien/ok/capable
    r"pas\s+la\s+force",                               # pas la force
    r"rien\s+ne\s+va",                                 # rien ne va
    r"plus\s+capable",                                 # plus capable
    r"ne\s+sers?\s+à\s+rien",                          # ne sers à rien
]

NEGATION_PATTERNS_EN = [
    r"don'?t\s+feel\s+(good|well|better|ok|fine)",     # don't feel good/well
    r"can'?t\s+(cope|go on|take it|do this|anymore)",  # can't cope/go on
    r"no\s+(hope|point|reason|future|way out)",        # no hope/point/reason
    r"never\s+(get better|be happy|feel ok|be ok)",    # never get better
    r"not\s+(ok|fine|alright|good|well)",               # not ok/fine
    r"nothing\s+(helps|works|matters)",                 # nothing helps/works
    r"won'?t\s+get\s+better",                           # won't get better
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
    Score de sentiment négatif : ratio de mots négatifs / total de mots.

    Note: Physical dampening removed — somatization_score handles
    the physical/emotional distinction at ML level.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0

    multi_word_neg = [neg for neg in NEGATIVE_WORDS if ' ' in neg]
    single_word_neg = [neg for neg in NEGATIVE_WORDS if ' ' not in neg]

    count = sum(1 for phrase in multi_word_neg if phrase in text_lower)
    count += sum(1 for w in words if any(neg in w for neg in single_word_neg))

    return count / len(words)


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


def _negation_score(text: str) -> float:
    """
    Negation-of-positive-state score: detects structures like
    "je ne me sens pas bien", "can't cope", "no hope".
    Ratio of negation pattern matches to word count, capped at 1.0.

    Score de négation d'état positif : détecte les structures comme
    "je ne me sens pas bien", "can't cope", "no hope".
    Ratio de correspondances de patterns de négation / nombre de mots, plafonné à 1.0.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0

    count = 0
    for pattern in NEGATION_PATTERNS_FR:
        count += len(re.findall(pattern, text_lower))
    for pattern in NEGATION_PATTERNS_EN:
        count += len(re.findall(pattern, text_lower))

    return min(count / max(len(words), 1), 1.0)


def _identity_conflict_score(text: str) -> float:
    """
    Identity-conflict lexical score: detects expressions of identity rejection,
    concealment, or family non-acceptance (2SLGBTQ+, cultural).

    Score de conflit identitaire : détecte les expressions de rejet identitaire,
    dissimulation ou non-acceptation familiale (2SLGBTQ+, culturel).
    """
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0
    count = 0
    for phrase in IDENTITY_CONFLICT_WORDS:
        if ' ' in phrase:
            if phrase in text_lower:
                count += 1
        else:
            count += sum(1 for w in words if phrase in w)
    return min(count / max(len(words), 1), 1.0)


def _somatization_score(text: str) -> float:
    """
    Somatization score: detects emotional distress co-occurring with physical
    complaints. Pure physical = 0.0, physical + emotional = positive.

    Score de somatisation : détecte la détresse émotionnelle co-occurrente
    avec des plaintes physiques. Physique pur = 0.0, physique + émotionnel = positif.
    """
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0

    # No physical context → no somatization signal
    if not any(phys in text_lower for phys in PHYSICAL_CONTEXT_WORDS):
        return 0.0

    # Physical context found — count emotional co-occurrences
    multi_word_emo = [w for w in SOMATIZATION_EMOTIONAL_WORDS if ' ' in w]
    single_word_emo = [w for w in SOMATIZATION_EMOTIONAL_WORDS if ' ' not in w]

    count = sum(1 for phrase in multi_word_emo if phrase in text_lower)
    count += sum(1 for w in words if any(emo in w for emo in single_word_emo))

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
          [7] negation score / score de négation (ne...pas bien, can't cope)
          [8] identity conflict score / score conflit identitaire (2SLGBTQ+)
          [9] somatization score / score somatisation (physique + émotionnel)
    """
    user_messages = [m for m in messages if m["role"] == "user"]

    if not user_messages:
        return np.zeros((1, 10))

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
            _negation_score(text),
            _identity_conflict_score(text),
            _somatization_score(text),
        ])
        feature_list.append(feat)

    return np.array(feature_list)


# ── Sentence embedding infrastructure ─────────────────────────────────────────
# Modèle d'embeddings chargé paresseusement (une seule fois)

_embedding_model = None
_crisis_centroid = None

_CRISIS_PHRASES = [
    # French / Français
    "je veux mourir", "je n'en peux plus", "je suis un fardeau",
    "plus rien n'a de sens", "je veux en finir",
    # English / Anglais
    "I want to die", "I can't take it anymore", "I am a burden",
    "nothing matters anymore", "I want to end it all",
]


def _get_embedding_model():
    """
    Lazy-load the multilingual sentence embedding model (once).
    Chargement paresseux du modèle d'embeddings multilingue (une seule fois).
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            logging.warning(f"Embedding model unavailable: {e}")
            return None
    return _embedding_model


def _get_crisis_centroid():
    """
    Compute and cache the crisis centroid vector (mean of crisis phrase embeddings).
    Calcule et met en cache le centroïde de crise (moyenne des embeddings de phrases de crise).
    """
    global _crisis_centroid
    if _crisis_centroid is None:
        model = _get_embedding_model()
        if model is None:
            return None
        embeddings = model.encode(_CRISIS_PHRASES, normalize_embeddings=True)
        _crisis_centroid = np.mean(embeddings, axis=0)
        _crisis_centroid /= np.linalg.norm(_crisis_centroid)
    return _crisis_centroid


def _encode_user_texts(user_texts: list):
    """
    Encode user texts once with the embedding model.
    Returns normalized embeddings or None if model unavailable.

    Encode les textes utilisateur une seule fois avec le modèle d'embeddings.
    Retourne les embeddings normalisés ou None si le modèle est indisponible.
    """
    model = _get_embedding_model()
    if model is None or len(user_texts) == 0:
        return None
    try:
        return model.encode(user_texts, normalize_embeddings=True)
    except Exception as e:
        logging.warning(f"Embedding encoding failed: {e}")
        return None


def _compute_embedding_trajectory_features(user_texts: list, embeddings=None) -> np.ndarray:
    """
    Compute 4 trajectory-level features from sentence embeddings:
      [0] embedding_drift — mean cosine distance between consecutive messages
      [1] crisis_similarity — cosine similarity of last message to crisis centroid
      [2] embedding_slope — PCA→1D slope over messages (directional semantic drift)
      [3] embedding_variance — mean pairwise cosine distance (coherence)

    Calcule 4 features de trajectoire à partir des embeddings de phrases :
      [0] embedding_drift — distance cosinus moyenne entre messages consécutifs
      [1] crisis_similarity — similarité cosinus du dernier message avec le centroïde de crise
      [2] embedding_slope — pente PCA→1D (dérive sémantique directionnelle)
      [3] embedding_variance — distance cosinus moyenne par paires (cohérence)

    Args:
        user_texts: list of user message strings
        embeddings: pre-computed embeddings (optional, avoids re-encoding)

    Returns np.ndarray(4,) — zeros if embedding model is unavailable.
    """
    if embeddings is None:
        embeddings = _encode_user_texts(user_texts)
    if embeddings is None:
        return np.zeros(4)

    try:
        n = len(embeddings)

        # 1. Embedding drift: mean cosine distance between consecutive messages
        if n > 1:
            dists = [1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
                     for i in range(n - 1)]
            embedding_drift = float(np.mean(dists))
        else:
            embedding_drift = 0.0

        # 2. Crisis similarity: cosine similarity of last message to crisis centroid
        centroid = _get_crisis_centroid()
        if centroid is not None:
            crisis_similarity = float(np.dot(embeddings[-1], centroid))
        else:
            crisis_similarity = 0.0

        # 3. Embedding slope: PCA→1D, then linear slope over message index
        if n > 2:
            pca = PCA(n_components=1)
            projected = pca.fit_transform(embeddings).flatten()
            x_norm = np.arange(n, dtype=float) / max(n - 1, 1)
            embedding_slope = float(np.polyfit(x_norm, projected, 1)[0])
        else:
            embedding_slope = 0.0

        # 4. Embedding variance: mean pairwise cosine distance
        if n > 1:
            pair_dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    pair_dists.append(1.0 - float(np.dot(embeddings[i], embeddings[j])))
            embedding_variance = float(np.mean(pair_dists))
        else:
            embedding_variance = 0.0

        return np.array([embedding_drift, crisis_similarity, embedding_slope, embedding_variance])

    except Exception as e:
        logging.warning(f"Embedding feature extraction failed: {e}")
        return np.zeros(4)


def _compute_coherence_features(user_texts: list, messages: list, embeddings=None) -> np.ndarray:
    """
    Compute 3 trajectory-level conversational coherence features:
      [0] short_response_ratio — fraction of user messages with < 5 words (disengagement)
      [1] min_topic_coherence — min cosine similarity between consecutive user embeddings
      [2] question_response_ratio — fraction of assistant questions followed by responsive reply

    Calcule 3 features de cohérence conversationnelle au niveau trajectoire :
      [0] short_response_ratio — fraction de messages utilisateur < 5 mots (désengagement)
      [1] min_topic_coherence — similarité cosinus min entre messages consécutifs
      [2] question_response_ratio — fraction de questions assistant suivies d'une réponse engagée

    Returns np.ndarray(3,)
    """
    n_user = len(user_texts) if user_texts else 0

    # 1. Short response ratio / Ratio de réponses courtes
    if n_user > 0:
        short_count = sum(1 for t in user_texts if len(t.split()) < 5)
        short_response_ratio = short_count / n_user
    else:
        short_response_ratio = 0.0

    # 2. Min topic coherence (using embeddings) / Cohérence thématique minimale
    if embeddings is not None and len(embeddings) > 1:
        consecutive_sims = [float(np.dot(embeddings[i], embeddings[i + 1]))
                            for i in range(len(embeddings) - 1)]
        min_topic_coherence = min(consecutive_sims)
    else:
        min_topic_coherence = 1.0  # Default: no drift detected

    # 3. Question-response ratio / Ratio question-réponse
    if messages and len(messages) > 1:
        assistant_questions = 0
        responsive_replies = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and "?" in msg["content"]:
                assistant_questions += 1
                # Check if next message is a responsive user reply
                if i + 1 < len(messages) and messages[i + 1]["role"] == "user":
                    reply = messages[i + 1]["content"]
                    if len(reply.split()) > 5 or "?" in reply:
                        responsive_replies += 1
        if assistant_questions > 0:
            question_response_ratio = responsive_replies / assistant_questions
        else:
            question_response_ratio = 1.0
    else:
        question_response_ratio = 1.0

    return np.array([short_response_ratio, min_topic_coherence, question_response_ratio])


def extract_trajectory_features(features_array: np.ndarray, user_texts: list = None,
                                 messages: list = None) -> np.ndarray:
    """
    Compute trajectory features from a (n_messages, n_features) matrix.
    Aggregates the temporal dynamics of each base feature.

    Calcule des features de trajectoire globale à partir d'une matrice
    (n_messages, n_features) décrivant la dynamique de la conversation.

    Args:
        features_array: (n_messages, n_features) matrix from extract_features()
        user_texts: optional list of user message strings for embedding features
        messages: optional full message list for coherence features

    Returns:
        1D np.ndarray:  [mean, std, slope, last, max, min] × n_base_features
                        + 4 embedding trajectory features
                        + 3 coherence features
        shape: (n_features * 6 + 4 + 3,) if user_texts provided, else (n_features * 6,)
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

    # Encode user texts once, share between embedding + coherence features
    # Encoder les textes une seule fois, partager entre embedding + cohérence
    if user_texts is not None:
        embeddings = _encode_user_texts(user_texts)

        emb_feats = _compute_embedding_trajectory_features(user_texts, embeddings=embeddings)
        trajectory_feats.extend(emb_feats.tolist())

        coherence_feats = _compute_coherence_features(user_texts, messages, embeddings=embeddings)
        trajectory_feats.extend(coherence_feats.tolist())

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
    "negation_score",
    "identity_conflict_score",
    "somatization_score",
]

EMBEDDING_FEATURE_NAMES = [
    "embedding_drift",
    "crisis_similarity",
    "embedding_slope",
    "embedding_variance",
]

COHERENCE_FEATURE_NAMES = [
    "short_response_ratio",
    "min_topic_coherence",
    "question_response_ratio",
]

TRAJECTORY_FEATURE_NAMES = [
    f"{feat}_{stat}"
    for feat in FEATURE_NAMES
    for stat in ["mean", "std", "slope", "last", "max", "min"]
] + EMBEDDING_FEATURE_NAMES + COHERENCE_FEATURE_NAMES
