# 🧠 DDEC — Détection de Dérive Émotionnelle Conversationnelle

> **Hackathon Mila · Sécurité IA en santé mentale des jeunes · POC v1.0**

DDEC est un système de surveillance en temps réel conçu pour détecter une **dérive émotionnelle progressive** chez des jeunes (16-22 ans) lors de conversations avec un chatbot de soutien. Il combine analyse lexicale, machine learning et modulation adaptative du LLM pour offrir des réponses ajustées à l'état émotionnel détecté.

---

## Table des matières

- [Contexte et motivation](#contexte-et-motivation)
- [Architecture](#architecture)
- [Niveaux d'alerte](#niveaux-dalerte)
- [Modules](#modules)
  - [Feature Extractor](#1-feature-extractor--ddecfeature_extractorpy)
  - [Classifier](#2-classifier--ddecclassifierpy)
  - [Response Modulator](#3-response-modulator--ddecresponse_modulatorpy)
  - [Session Tracker](#4-session-tracker--ddecsession_trackerpy)
  - [Interface Streamlit](#5-interface-streamlit--apppy)
- [Données synthétiques](#données-synthétiques)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Métriques](#métriques)
- [Structure du projet](#structure-du-projet)
- [Limites connues et pistes d'amélioration](#limites-connues-et-pistes-damélioration)

---

## Contexte et motivation

Les chatbots de soutien émotionnel pour les jeunes peuvent, sans système de surveillance, ne pas détecter une dégradation progressive de l'état mental de l'utilisateur. DDEC propose une couche d'analyse orthogonale au LLM : elle surveille la **trajectoire** des messages de l'utilisateur (pas uniquement leur contenu ponctuel) pour identifier un glissement vers la détresse.

La détection repose sur des features **purement lexicales et structurelles** (sans LLM, sans réseau de neurones profond), ce qui garantit :
- une latence nulle à l'inférence,
- une complète explicabilité,
- un fonctionnement hors-ligne.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Interface Streamlit (app.py)              │
│  Chat  │  Jauge alerte  │  Probas  │  Features  │  Longitudinal  │
└────────┬────────────────────────────────────────────────────────┘
         │ messages utilisateur
         ▼
┌────────────────────────┐
│   Feature Extractor    │  ← 7 features/message → 48 features de trajectoire
│  (numpy + regex, 0 LLM)│
└────────────┬───────────┘
             │ vecteur 48D
             ▼
┌────────────────────────┐        ┌──────────────────────────┐
│   DDECClassifier       │───────►│  Règles de sécurité       │
│  (GradientBoosting)    │        │  (override lexical)       │
└────────────┬───────────┘        └──────────────────────────┘
             │ niveau 0-3 + confiance + features dominantes
             ▼
┌────────────────────────┐        ┌──────────────────────────┐
│  Response Modulator    │───────►│  LLM (Claude / Mistral / │
│  (prompt adaptatif)    │        │  Llama / fallback)        │
└────────────────────────┘        └──────────────────────────┘
             │
             ▼
┌────────────────────────┐
│   Session Tracker      │  ← SQLite, historique inter-sessions
│  (longitudinal SQLite) │
└────────────────────────┘
```

---

## Niveaux d'alerte

| Niveau | Couleur | Label  | Description                                    | Mode LLM                          |
|--------|---------|--------|------------------------------------------------|-----------------------------------|
| 0      | 🟢 Vert  | verte  | Conversation normale, jeune en bonne forme     | Standard bienveillant             |
| 1      | 🟡 Jaune | jaune  | Signes préoccupants, fatigue, solitude         | Validation émotionnelle renforcée |
| 2      | 🟠 Orange| orange | Détresse significative, pensées négatives      | Soutien actif + ressources        |
| 3      | 🔴 Rouge | rouge  | Crise potentielle, pensées de finalité         | Crise — orientation urgente       |

---

## Modules

### 1. Feature Extractor — `ddec/feature_extractor.py`

Cœur analytique du système. Pour chaque message de l'utilisateur, extrait **7 features de base** :

| Feature              | Description                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------|
| `longueur_mots`      | Nombre de mots du message                                                                             |
| `ratio_ponctuation`  | Proportion de signes de ponctuation (`.`, `!`, `?`, `;`, `…`) sur le nombre total de caractères     |
| `presence_question`  | Indicateur binaire 0/1 : le message contient-il un `?`                                               |
| `score_negatif`      | Ratio de mots/expressions négatifs (`triste`, `épuisé`, `vide`, `honte`…) sur la longueur du message. Réduit de 50% si un contexte physique est détecté (ex : "j'ai mal au dos") |
| `score_finalite`     | Ratio de mots du champ lexical de finalité/détresse (`disparaître`, `en finir`, `fardeau`, `à quoi bon`…) |
| `score_espoir`       | Ratio de mots du champ lexical de l'espoir/ressources (`demain`, `essayer`, `famille`, `guérir`…)    |
| `delta_longueur`     | Variation relative de longueur par rapport au message précédent — détecte les raccourcissements progressifs |

#### Features de trajectoire (48 features → entrée du classifier)

Pour chaque feature de base, 6 statistiques de trajectoire sont calculées sur l'ensemble de la conversation :

| Statistique | Description                                                          |
|-------------|----------------------------------------------------------------------|
| `_mean`     | Moyenne sur tous les messages                                        |
| `_std`      | Écart-type — mesure la variabilité                                   |
| `_slope`    | Pente de régression linéaire normalisée — capte la tendance          |
| `_last`     | Valeur du dernier message — état le plus récent                      |
| `_max`      | Maximum observé dans la conversation                                 |
| `_min`      | Minimum observé dans la conversation                                 |

Résultat : **vecteur 1D de 42 features** (7 × 6) soumis au classifier.

> **Dictionnaires lexicaux** : trois listes de mots/expressions sont utilisées —
> `FINALITY_WORDS` (35 entrées), `HOPE_WORDS` (24 entrées), `NEGATIVE_WORDS` (37 entrées).
> Aucun embedding, aucune API externe.

---

### 2. Classifier — `ddec/classifier.py`

#### Pipeline sklearn

```
StandardScaler → GradientBoostingClassifier(n_estimators=200, max_depth=3)
```

La normalisation par `StandardScaler` est essentielle car les features sont d'échelles très différentes (longueur en dizaines de mots vs scores en 0-1).

#### Règles de sécurité (override lexical)

Le classifier applique **deux niveaux de règles prioritaires** avant et après la prédiction ML :

**Avant la prédiction (contexte insuffisant < 3 messages) :**
- Si un mot critique est détecté (`mourir`, `suicide`, `en finir`…) → niveau minimum **Orange** (0.70 de confiance)
- Si 2+ mots de détresse sont détectés (`pleure`, `vide`, `seul`…) → niveau minimum **Jaune** (0.65)
- Sinon → **Vert** mode sécuritaire (0.80)

**Après la prédiction ML (3+ messages) :**
- Un `niveau_minimum` est calculé à partir du texte complet (même logique mots critiques/détresse)
- La prédiction ML ne peut **jamais descendre sous ce niveau minimum** : `predicted = max(ml_pred, niveau_minimum)`
- Si la confiance du ML est < 0.45, retourne **Jaune** par défaut (principe de précaution)

#### Interprétabilité

Les **5 features dominantes** sont calculées à chaque prédiction par le score composite :

```
composite_i = feature_importance_i × |valeur_scalée_i|
```

Un dictionnaire de noms lisibles (`_FEATURE_DISPLAY_NAMES`) traduit les noms techniques en signaux compréhensibles (ex. `score_finalite_slope` → "Tendance hausse finalité").

---

### 3. Response Modulator — `ddec/response_modulator.py`

#### Prompts système adaptatifs

Quatre prompts système distincts sont injectés dans le LLM en fonction du niveau d'alerte :

- **Niveau 0** : assistant bienveillant, questions ouvertes, registre positif
- **Niveau 1** : validation émotionnelle prioritaire, une question à la fois, écoute active
- **Niveau 2** : espace sécurisé, ressources mentionnées naturellement (Jeunesse J'écoute : 1-800-668-6868)
- **Niveau 3** : protocole de crise — valider la souffrance, évaluer la sécurité, orienter vers 911/ressources immédiates

#### Hiérarchie de LLM avec fallback automatique

```
claude-haiku (Anthropic API) → mistral (Ollama local) → llama3.2:1b (Ollama local) → fallback statique
```

L'utilisateur peut forcer un modèle spécifique depuis l'interface. Si le modèle choisi est indisponible, le fallback statique répond `"Je suis là pour t'écouter. Peux-tu me dire comment tu te sens ?"`.

| Modèle             | Requis          | Indicateur |
|--------------------|-----------------|------------|
| `claude-haiku`     | `ANTHROPIC_API_KEY` | 🟣         |
| `mistral`          | Ollama local    | 🔵         |
| `llama3.2:1b`      | Ollama local    | ⚪         |
| `fallback-statique`| Aucun           | ⚠️         |

---

### 4. Session Tracker — `ddec/session_tracker.py`

Module de **surveillance longitudinale inter-sessions** utilisant SQLite (inclus dans Python, zéro dépendance externe).

#### Schéma de base de données (`data/ddec_sessions.db`)

**Table `sessions`** : une ligne par session de chat
```sql
id, user_id, session_id (UUID4), started_at, ended_at, max_alert_level, message_count
```

**Table `alert_events`** : un enregistrement par message analysé
```sql
id, user_id, session_id, timestamp, alert_level, confidence, trigger_message (500 chars max)
```

#### Analyse de risque longitudinal

Sur les 7 dernières sessions complétées, calcule :

- **`risk_score`** : moyenne pondérée des niveaux max (sessions récentes = poids plus élevé), normalisée sur [0, 1]
- **`trend`** : tendance calculée en comparant la moyenne des 3 dernières sessions aux 3 précédentes
  - `improving` : baisse > 0.3
  - `worsening` : hausse > 0.3
  - `stable` : sinon
- **`consecutive_high_sessions`** : nombre de sessions consécutives récentes avec niveau ≥ Orange
- **`recommendation`** : action suggérée au professionnel de santé

| Seuil                                    | Recommandation                          |
|------------------------------------------|-----------------------------------------|
| consecutive_high ≥ 3 **ou** score > 0.8  | Intervention prioritaire recommandée    |
| score > 0.6                              | Consultation professionnelle suggérée   |
| score > 0.3                              | Attention soutenue recommandée          |
| sinon                                    | Suivi normal                            |

---

### 5. Interface Streamlit — `app.py`

Interface en deux colonnes avec mise à jour en temps réel après chaque message.

#### Colonne gauche : Chat

- Bulles de conversation avec CSS personnalisé (style iMessage)
- Zone de saisie via `st.form` (soumission sur Entrée)
- Analyse DDEC déclenchée **avant** la génération de la réponse LLM

#### Colonne droite : Dashboard DDEC

| Composant                   | Description                                                                       |
|-----------------------------|-----------------------------------------------------------------------------------|
| **Jauge circulaire**        | Indicateur Plotly 0-3 avec barre de confiance                                     |
| **Barres de probabilité**   | Probabilités par classe (verte/jaune/orange/rouge) avec code couleur              |
| **Signaux actifs**          | Features dominantes affichées en pills colorés                                    |
| **Évolution du niveau**     | Graphique ligne Plotly : historique des niveaux de la session courante            |
| **Historique longitudinal** | Graphique barres par session + tendance + recommandation (données SQLite)         |
| **Sélecteur LLM**           | 4 boutons pour choisir/forcer le modèle conversationnel                           |
| **Prompt système actif**    | Description du mode courant + expander pour voir le prompt complet                |
| **Stats de session**        | Compteurs : messages, échanges, pic d'alerte                                      |

---

## Données synthétiques

**`data/synthetic_conversations.json`** — 24 conversations initiales (6 par classe).

Chaque conversation contient 24 messages (12 user + 12 assistant), en français québécois authentique, générées via `generate_synthetic_data.py` à partir de Claude Haiku.

### Archetypes de génération

| Archetype | Caractéristiques                                                                                              |
|-----------|---------------------------------------------------------------------------------------------------------------|
| `verte`   | Projets futurs, humour, amis/famille mentionnés, messages variés, émotions normales (stress d'examen)         |
| `jaune`   | Fatigue persistante, solitude croissante, doutes sur soi, dérive graduelle sur 12 messages                    |
| `orange`  | Sentiment de vide, pleurs, sentiment d'être un fardeau, messages de plus en plus courts, inutilité            |
| `rouge`   | Désir de disparaître, isolement total, ton de finalité, messages courts et intenses, plus de projets          |

### Génération de données supplémentaires

```bash
export ANTHROPIC_API_KEY="sk-..."
python generate_synthetic_data.py   # génère 80 conversations supplémentaires (20 par classe)
```

Le script gère les erreurs JSON et le rate limiting automatiquement, avec jusqu'à 25 tentatives par classe.

---

## Installation

### Prérequis

- Python 3.9+
- (Optionnel) Ollama pour les LLM locaux : https://ollama.ai

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd ddec-hackathon

# 2. Installer les dépendances
pip install streamlit plotly scikit-learn numpy joblib requests anthropic

# 3. (Optionnel) Configurer la clé API Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. (Optionnel) Installer les modèles Ollama
ollama pull mistral
ollama pull llama3.2:1b

# 5. Entraîner le modèle
python train.py

# 6. Lancer l'interface
streamlit run app.py
```

---

## Utilisation

### Entraînement

```bash
python train.py
```

Affiche :
- Validation croisée stratifiée (k=4 folds)
- Rapport de classification et matrice de confusion
- Top 10 des features les plus importantes
- Sauvegarde dans `models/ddec_model.joblib`
- Test de rechargement avec une conversation de crise

### Interface web

```bash
streamlit run app.py
```

Ouvre `http://localhost:8501`. Pour démarrer une nouvelle session de surveillance, cliquer sur **Réinitialiser** (la session courante est clôturée en base avant le reset).

---

## Métriques

Résultats sur le dataset initial de 24 conversations :

| Métrique                  | Valeur                 |
|---------------------------|------------------------|
| Train accuracy            | 100% (overfitting attendu) |
| CV accuracy (k=4)         | 66.7% ± 26.4%          |
| Nombre de features        | 42 (7 × 6 stats)       |
| Top feature               | `longueur_mots_std`    |
| 2e feature                | `score_negatif_mean`   |

> **Note** : le dataset de 24 conversations est trop petit pour des métriques fiables. L'utilisation de `generate_synthetic_data.py` pour générer ~100 conversations par classe améliore significativement la CV accuracy.

---

## Structure du projet

```
ddec-hackathon/
├── app.py                          # Interface Streamlit
├── train.py                        # Script d'entraînement
├── generate_synthetic_data.py      # Génération de données via Claude API
│
├── ddec/                           # Package Python principal
│   ├── __init__.py                 # Version (0.1.0)
│   ├── feature_extractor.py        # Extraction features lexicales + trajectoire
│   ├── classifier.py               # DDECClassifier (GradientBoosting + règles)
│   ├── response_modulator.py       # Prompts adaptatifs + appels LLM
│   └── session_tracker.py          # Suivi inter-sessions SQLite
│
├── data/
│   ├── synthetic_conversations.json  # Dataset d'entraînement
│   └── ddec_sessions.db             # Base de données SQLite (créée auto)
│
└── models/
    └── ddec_model.joblib            # Modèle entraîné (créé par train.py)
```

---

## Limites connues et pistes d'amélioration

| Limite                                                              | Piste d'amélioration                                             |
|---------------------------------------------------------------------|------------------------------------------------------------------|
| Dataset de 24 conversations — overfitting élevé                    | Générer 100+ conversations par classe avec `generate_synthetic_data.py` |
| Features de trajectoire peu fiables avant 3-4 messages             | Règles de sécurité lexicales déjà actives pour les premiers messages |
| Un seul `user_id` "demo_user" dans l'interface de démo             | Ajouter un système d'authentification léger                      |
| Lexiques en français uniquement                                     | Étendre au français québécois/joual et à l'anglais               |
| Aucune validation clinique des seuils                              | Collaboration avec professionnels en santé mentale               |
| LLM non fine-tuné pour le contexte de crise                        | Fine-tuning sur conversations d'intervenants certifiés           |

---

## Ressources d'urgence (Canada)

> Ces ressources sont intégrées dans les prompts de niveau Orange et Rouge.

- **Jeunesse J'écoute** : 1-800-668-6868 (24h/24, gratuit, confidentiel) — texto : 686868
- **Urgences** : 911
- **Médecin de famille / service de soutien scolaire**
