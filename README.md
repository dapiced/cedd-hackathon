# 🧠 CEDD — Conversational Emotional Drift Detection

> **Mila Hackathon · AI Safety in Youth Mental Health · POC v1.0**

[🇬🇧 English](#english-documentation) | [🇫🇷 Français](#documentation-en-français)

---

## English Documentation

CEDD is a real-time monitoring system designed to detect **progressive emotional drift** in youth (16–22 years old) during conversations with an AI support chatbot. It combines lexical analysis, machine learning, and adaptive LLM modulation to deliver responses calibrated to the detected emotional state.

---

### Table of Contents

- [Context & Motivation](#context--motivation)
- [Architecture](#architecture)
- [Alert Levels](#alert-levels)
- [Modules](#modules)
  - [Feature Extractor](#1-feature-extractor--ceddfeature_extractorpy)
  - [Classifier](#2-classifier--ceddclassifierpy)
  - [Response Modulator](#3-response-modulator--ceddresponse_modulatorpy)
  - [Session Tracker](#4-session-tracker--ceddsession_trackerpy)
  - [Streamlit Interface](#5-streamlit-interface--apppy)
- [Bilingual Support](#bilingual-support)
- [Synthetic Dataset](#synthetic-dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Metrics](#metrics)
- [Project Structure](#project-structure)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Emergency Resources](#emergency-resources)

---

### Context & Motivation

Emotional support chatbots for youth can, without a monitoring layer, fail to detect a user's gradual mental deterioration. CEDD adds an orthogonal analysis layer to the LLM: it monitors the **trajectory** of user messages (not just their instant content) to identify a drift toward distress.

Detection relies on **purely lexical and structural features** (no LLM, no deep neural network), which ensures:
- Zero inference latency,
- Full explainability,
- Offline operation.

---

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Streamlit Interface (app.py) — FR / EN             │
│  Chat  │  Alert gauge  │  Probas  │  Features  │  Longitudinal  │
└────────┬────────────────────────────────────────────────────────┘
         │ user messages
         ▼
┌────────────────────────┐
│   Feature Extractor    │  ← 7 features/message → 42 trajectory features
│  (numpy + regex, 0 LLM)│    Bilingual lexicons (FR + EN)
└────────────┬───────────┘
             │ 42D vector
             ▼
┌────────────────────────┐        ┌──────────────────────────┐
│   CEDDClassifier       │───────►│  Safety rules            │
│  (GradientBoosting)    │        │  (lexical override)      │
└────────────┬───────────┘        └──────────────────────────┘
             │ level 0-3 + confidence + top features
             ▼
┌────────────────────────┐        ┌──────────────────────────┐
│  Response Modulator    │───────►│  LLM Claude / Mistral /  │
│  (adaptive prompt)     │        │  Llama / without llm     │
│  FR or EN system prompt│        └──────────────────────────┘
└────────────────────────┘
             │
             ▼
┌────────────────────────┐
│   Session Tracker      │  ← SQLite, cross-session history
│  (longitudinal SQLite) │
└────────────────────────┘
```

---

### Alert Levels

| Level | Color     | Label  | Description                                | LLM Mode                         |
|-------|-----------|--------|--------------------------------------------|----------------------------------|
| 0     | 🟢 Green  | green  | Normal conversation, youth doing well      | Supportive standard              |
| 1     | 🟡 Yellow | yellow | Concerning signs, fatigue, loneliness      | Enhanced emotional validation    |
| 2     | 🟠 Orange | orange | Significant distress, negative thoughts    | Active support + resources       |
| 3     | 🔴 Red    | red    | Potential crisis, finality thoughts        | Crisis — urgent referral         |

---

### Modules

#### 1. Feature Extractor — `cedd/feature_extractor.py`

Analytical core of the system. Extracts **7 base features** per user message:

| Feature              | Description                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------|
| `word_count`         | Word count of the message                                                                             |
| `punctuation_ratio`  | Proportion of punctuation characters (`.`, `!`, `?`, `;`, `…`) over total characters                 |
| `question_presence`  | Binary 0/1 indicator: does the message contain a `?`                                                  |
| `negative_score`     | Ratio of negative words/phrases over message length. Halved when a physical context is detected       |
| `finality_score`     | Ratio of finality/distress vocabulary (`disappear`, `end it`, `burden`, `what's the point`…)         |
| `hope_score`         | Ratio of hope/resources vocabulary (`tomorrow`, `try`, `family`, `heal`…)                            |
| `length_delta`       | Relative length change vs previous message — detects progressive shortening                          |

**Bilingual lexicons**: `FINALITY_WORDS`, `HOPE_WORDS`, and `NEGATIVE_WORDS` each contain both French and English terms, enabling the system to analyse conversations in either language.

#### Trajectory Features (42 total → classifier input)

For each base feature, 6 trajectory statistics computed over the full conversation:

| Statistic | Description                                                        |
|-----------|--------------------------------------------------------------------|
| `_mean`   | Average across all messages                                        |
| `_std`    | Standard deviation — measures variability                          |
| `_slope`  | Normalised linear regression slope — captures the trend            |
| `_last`   | Value of the last message — most recent state                      |
| `_max`    | Maximum observed in the conversation                               |
| `_min`    | Minimum observed in the conversation                               |

---

#### 2. Classifier — `cedd/classifier.py`

**sklearn pipeline:**
```
StandardScaler → GradientBoostingClassifier(n_estimators=200, max_depth=3)
```

**Safety rules (lexical override):**

*Before ML (< 3 user messages):*
- Crisis keyword detected (`gun`, `knife`, `suicide`, `kill myself`, `want to die`, etc.) → immediate **Red** (0.90)
- Critical word detected → minimum **Orange** (0.70 confidence)
- 2+ distress words → minimum **Yellow** (0.65)
- Otherwise → **Green** safe mode (0.80)

*After ML (3+ messages):*
- A `minimum_level` is computed from the full text (same keywords as above)
- ML prediction can **never go below this minimum**: `predicted = max(ml_pred, minimum_level)`
- If ML confidence < 0.45 → returns **Yellow** by default (precautionary principle)
- **Short-conversation cap**: ML trained on 12-message conversations — for < 6 user messages, ML is capped at **Orange** max. Red via ML only fires with sufficient conversational context.
- **Safety override display**: when safety rules raise the level above ML prediction, class probability bars are replaced by a "crisis word detected" badge to avoid misleading output.

**Feature display names** are available in both French and English, selectable via the `lang` parameter.

---

#### 3. Response Modulator — `cedd/response_modulator.py`

**Adaptive system prompts** (French and English) injected into the LLM based on alert level:

- **Level 0**: Warm supportive assistant, open questions, positive register
- **Level 1**: Emotional validation priority, one question at a time, active listening
- **Level 2**: Safe space, resources mentioned naturally (Kids Help Phone: 1-800-668-6868)
- **Level 3**: Crisis protocol — validate suffering, assess safety, refer to 911/immediate resources

**LLM hierarchy with automatic fallback:**
```
claude-haiku (Anthropic API) → mistral (local Ollama) → llama3.2:1b (local Ollama) → without llm
```

| Model              | Requires            | Indicator |
|--------------------|---------------------|-----------|
| `claude-haiku`     | `ANTHROPIC_API_KEY` | 🟣        |
| `mistral`          | Local Ollama        | 🔵        |
| `llama3.2:1b`      | Local Ollama        | ⚪        |
| `without llm`      | None                | ⚠️        |

---

#### 4. Session Tracker — `cedd/session_tracker.py`

**Cross-session longitudinal monitoring** using SQLite.

**Schema:**
- `sessions`: one row per chat session
- `alert_events`: one record per analysed message

**Longitudinal risk analysis** over the last 7 sessions:
- `risk_score`: weighted average of max levels (recent = higher weight), normalised to [0, 1]
- `trend`: `improving` / `stable` / `worsening`
- `consecutive_high_sessions`: consecutive recent sessions with level ≥ Orange
- `recommendation`: action suggested to a healthcare professional

---

#### 5. Streamlit Interface — `app.py`

Two-column interface with real-time updates after each message.

**Language toggle** in the header: switch between 🇬🇧 English and 🇫🇷 Français at any time. The UI, system prompts, and LLM responses all switch to the selected language.

| Component                  | Description                                                                |
|----------------------------|----------------------------------------------------------------------------|
| **Circular gauge**         | Plotly 0-3 indicator with confidence bar                                   |
| **Probability bars**       | Per-class probabilities (green/yellow/orange/red) with colour coding       |
| **Active signals**         | Dominant features displayed as colour-coded pills                          |
| **Level history**          | Plotly line chart: alert level history for the current session             |
| **Longitudinal history**   | Per-session bar chart + trend + recommendation (SQLite data)               |
| **LLM selector**           | 4 buttons to choose/force the conversational model                         |
| **Active system prompt**   | Description of the current mode + expander showing the full prompt         |
| **Session stats**          | Counters: messages, exchanges, alert peak                                  |

---

### Bilingual Support

| Layer                  | English                        | French                          |
|------------------------|--------------------------------|---------------------------------|
| Web UI                 | Full (toggle button in header) | Full (default)                  |
| LLM system prompts     | All 4 levels                   | All 4 levels                    |
| Lexical analysis       | EN words in all lexicons       | FR words in all lexicons        |
| Feature display names  | Via `lang="en"` parameter      | Via `lang="fr"` parameter       |
| Synthetic data         | `--lang en` flag               | `--lang fr` (default)           |
| History simulation     | `--lang en` flag               | `--lang fr` (default)           |

---

### Synthetic Dataset

**`data/synthetic_conversations.json`** — 24 seed conversations (6 per class).

Each conversation contains 24 messages (12 user + 12 assistant), in authentic Canadian French, generated via `generate_synthetic_data.py` using Claude Haiku.

#### Generation Archetypes

| Archetype | Characteristics                                                                                           |
|-----------|-----------------------------------------------------------------------------------------------------------|
| `verte`   | Future projects, humour, friends/family mentioned, varied messages, normal emotions (exam stress)         |
| `jaune`   | Persistent fatigue, growing loneliness, self-doubt, gradual drift over 12 messages                       |
| `orange`  | Feeling of emptiness, crying, feeling like a burden, shorter messages, sense of uselessness               |
| `rouge`   | Desire to disappear, total isolation, tone of finality, short intense messages, no future plans           |

#### Generating Additional Data (FR and EN)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Generate 80 French conversations (20 per class) / 80 conversations en français
python generate_synthetic_data.py --lang fr --count 20

# Generate 80 English conversations (20 per class) / 80 conversations en anglais
python generate_synthetic_data.py --lang en --count 20
```

---

### Installation

**Prerequisites:**
- Python 3.9+
- (Optional) Ollama for local LLMs

```bash
# 1. Clone the repository
git clone <repo-url>
cd cedd-hackathon

# 2. Create a Python env.
cedd-hackathon$ python3 -m venv venv
cedd-hackathon$ source venv/bin/activate
(venv) /cedd-hackathon$ 

# 3. Install dependencies
(venv) /cedd-hackathon$ pip install streamlit plotly scikit-learn numpy joblib requests anthropic

# 4. (Optional) Configure Claude API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 5. (Optional) Install Ollama models
ollama pull mistral
ollama pull llama3.2:1b

# 6. Train the model
python train.py

# 7. Launch the interface
streamlit run app.py
```

---

### Usage

```bash
# Train the classifier
python train.py

# Launch the bilingual web interface
streamlit run app.py

# Simulate session history for demo (French)
python simulate_history.py --lang fr

# Simulate session history for demo (English)
python simulate_history.py --lang en

# Generate additional French synthetic data
python generate_synthetic_data.py --lang fr --count 20

# Generate additional English synthetic data
python generate_synthetic_data.py --lang en --count 20
```

Opens at `http://localhost:8501`. Use the language toggle (🇫🇷 / 🇬🇧) in the header to switch languages. Click **Reset / Réinitialiser** to start a new monitoring session.

---

### Metrics

Results on the initial 24-conversation dataset:

| Metric                    | Value                     |
|---------------------------|---------------------------|
| Train accuracy            | 100% (expected overfitting) |
| CV accuracy (k=4)         | 66.7% ± 26.4%             |
| Number of features        | 42 (7 × 6 stats)          |
| Top feature               | `word_count_std`          |
| 2nd feature               | `negative_score_mean`     |

> **Note**: 24 conversations is too small for reliable metrics. Running `generate_synthetic_data.py` to produce ~100 conversations per class significantly improves CV accuracy.

---

### Project Structure

```
cedd-hackathon/
├── app.py                          # Bilingual Streamlit interface / Interface Streamlit bilingue
├── train.py                        # Training script / Script d'entraînement
├── generate_synthetic_data.py      # Data generation via Claude API (FR + EN)
├── simulate_history.py             # Demo history simulation (FR + EN)
│
├── cedd/                           # Main Python package
│   ├── __init__.py
│   ├── feature_extractor.py        # Bilingual lexical + trajectory feature extraction
│   ├── classifier.py               # CEDDClassifier (GradientBoosting + rules)
│   ├── response_modulator.py       # Adaptive prompts (FR + EN) + LLM calls
│   └── session_tracker.py          # Cross-session SQLite tracking
│
├── data/
│   ├── synthetic_conversations.json  # Training dataset (FR + EN)
│   └── cedd_sessions.db             # SQLite database (auto-created)
│
└── models/
    └── cedd_model.joblib            # Trained model (created by train.py)
```

---

### Known Limitations & Future Work

| Limitation                                                          | Potential Improvement                                               |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| 24-conversation dataset — high overfitting                         | Generate 100+ conversations per class with `generate_synthetic_data.py` |
| ML unreliable for short conversations (< 6 messages)               | ML capped at Orange for < 6 messages; crisis keywords trigger Red instantly at any point |
| Single `demo_user` ID in demo interface                            | Add lightweight authentication system                               |
| ML model trained on French data only                               | Retrain on combined FR+EN dataset                                   |
| No clinical validation of thresholds                               | Collaborate with mental health professionals                        |
| LLM not fine-tuned for crisis contexts                             | Fine-tune on certified counsellor conversations                     |

---

### Emergency Resources

> Integrated into Orange and Red alert level prompts.

- **Kids Help Phone**: 1-800-668-6868 (24/7, free, confidential) — text: 686868
- **Emergency services**: 911
- **Family doctor / school counselling service**

---
---

## Documentation en Français

CEDD est un système de surveillance en temps réel conçu pour détecter une **dérive émotionnelle progressive** chez des jeunes (16-22 ans) lors de conversations avec un chatbot de soutien. Il combine analyse lexicale, machine learning et modulation adaptative du LLM pour offrir des réponses ajustées à l'état émotionnel détecté.

---

### Table des matières

- [Contexte et motivation](#contexte-et-motivation)
- [Architecture](#architecture-1)
- [Niveaux d'alerte](#niveaux-dalerte)
- [Modules](#modules-1)
- [Support bilingue](#support-bilingue)
- [Données synthétiques](#données-synthétiques)
- [Installation](#installation-1)
- [Utilisation](#utilisation)
- [Métriques](#métriques)
- [Structure du projet](#structure-du-projet)
- [Limites connues et pistes d'amélioration](#limites-connues-et-pistes-damélioration)

---

### Contexte et motivation

Les chatbots de soutien émotionnel pour les jeunes peuvent, sans système de surveillance, ne pas détecter une dégradation progressive de l'état mental de l'utilisateur. CEDD propose une couche d'analyse orthogonale au LLM : elle surveille la **trajectoire** des messages de l'utilisateur (pas uniquement leur contenu ponctuel) pour identifier un glissement vers la détresse.

La détection repose sur des features **purement lexicales et structurelles** (sans LLM, sans réseau de neurones profond), ce qui garantit :
- une latence nulle à l'inférence,
- une complète explicabilité,
- un fonctionnement hors-ligne.

---

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│            Interface Streamlit (app.py) — FR / EN               │
│  Chat  │  Jauge alerte  │  Probas  │  Features  │  Longitudinal │
└────────┬────────────────────────────────────────────────────────┘
         │ messages utilisateur
         ▼
┌────────────────────────┐
│   Feature Extractor    │  ← 7 features/message → 42 features de trajectoire
│  (numpy + regex, 0 LLM)│    Lexiques bilingues (FR + EN)
└────────────┬───────────┘
             │ vecteur 42D
             ▼
┌────────────────────────┐        ┌──────────────────────────┐
│   CEDDClassifier       │───────►│  Règles de sécurité      │
│  (GradientBoosting)    │        │  (override lexical)      │
└────────────┬───────────┘        └──────────────────────────┘
             │ niveau 0-3 + confiance + features dominantes
             ▼
┌────────────────────────┐        ┌──────────────────────────┐
│  Response Modulator    │───────►│  LLM Claude / Mistral /  │
│  (prompt adaptatif)    │        │  Llama / sans llm        │
│  Prompt FR ou EN       │        └──────────────────────────┘
└────────────────────────┘
             │
             ▼
┌────────────────────────┐
│   Session Tracker      │  ← SQLite, historique inter-sessions
│  (longitudinal SQLite) │
└────────────────────────┘
```

---

### Niveaux d'alerte

| Niveau | Couleur  | Label  | Description                                    | Mode LLM                          |
|--------|----------|--------|------------------------------------------------|-----------------------------------|
| 0      | 🟢 Vert   | verte  | Conversation normale, jeune en bonne forme     | Standard bienveillant             |
| 1      | 🟡 Jaune  | jaune  | Signes préoccupants, fatigue, solitude         | Validation émotionnelle renforcée |
| 2      | 🟠 Orange | orange | Détresse significative, pensées négatives      | Soutien actif + ressources        |
| 3      | 🔴 Rouge  | rouge  | Crise potentielle, pensées de finalité         | Crise — orientation urgente       |

---

### Modules

#### 1. Feature Extractor — `cedd/feature_extractor.py`

Cœur analytique du système. Extrait **7 features de base** par message utilisateur. Les **dictionnaires lexicaux** contiennent désormais des termes français **et** anglais : `FINALITY_WORDS`, `HOPE_WORDS`, `NEGATIVE_WORDS`.

#### Features de trajectoire (42 features → entrée du classifier)

Pour chaque feature de base, 6 statistiques de trajectoire (mean, std, slope, last, max, min).

#### 2. Classifier — `cedd/classifier.py`

Pipeline : `StandardScaler → GradientBoostingClassifier(n_estimators=200, max_depth=3)`

Règles de sécurité prioritaires (override lexical) avant et après la prédiction ML. Mots-clés de crise étendus (arme, pistolet, couteau, gun, knife, shoot…) déclenchent **Rouge immédiatement** à tout moment. Pour < 6 messages utilisateur, le ML est plafonné à Orange. En cas d'override de sécurité, les barres de probabilité sont remplacées par un badge "mot de crise détecté". Les **noms lisibles des features** sont disponibles en français et en anglais, sélectionnables via le paramètre `lang`.

#### 3. Response Modulator — `cedd/response_modulator.py`

Quatre prompts système distincts, disponibles en **français et en anglais**, injectés dans le LLM en fonction du niveau d'alerte et de la langue de l'interface.

- **Niveau 0** : assistant bienveillant, questions ouvertes
- **Niveau 1** : validation émotionnelle prioritaire, écoute active
- **Niveau 2** : espace sécurisé, ressources (Jeunesse J'écoute : 1-800-668-6868)
- **Niveau 3** : protocole de crise — valider la souffrance, évaluer la sécurité, orienter

Hiérarchie LLM : `claude-haiku → mistral → llama3.2:1b → sans llm`

#### 4. Session Tracker — `cedd/session_tracker.py`

Surveillance longitudinale inter-sessions via SQLite. Calcule `risk_score`, `trend`, `consecutive_high_sessions` et `recommendation` sur les 7 dernières sessions.

#### 5. Interface Streamlit — `app.py`

Interface bilingue en deux colonnes. **Bouton de langue** dans l'en-tête : bascule entre 🇫🇷 Français et 🇬🇧 English. L'interface complète, les prompts système et les réponses LLM basculent vers la langue choisie.

---

### Support bilingue

| Couche                  | Anglais                              | Français                              |
|-------------------------|--------------------------------------|---------------------------------------|
| Interface web           | Complète (bouton dans l'en-tête)     | Complète (langue par défaut)          |
| Prompts système LLM     | 4 niveaux                            | 4 niveaux                             |
| Analyse lexicale        | Mots EN dans les lexiques            | Mots FR dans les lexiques             |
| Noms des features       | Via paramètre `lang="en"`            | Via paramètre `lang="fr"`             |
| Données synthétiques    | Drapeau `--lang en`                  | Drapeau `--lang fr` (défaut)          |
| Simulation d'historique | Drapeau `--lang en`                  | Drapeau `--lang fr` (défaut)          |

---

### Données synthétiques

**`data/synthetic_conversations.json`** — 24 conversations initiales (6 par classe) en français québécois authentique.

#### Génération de données supplémentaires (FR et EN)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# 80 conversations en français (20 par classe)
python generate_synthetic_data.py --lang fr --count 20

# 80 conversations en anglais (20 par classe)
python generate_synthetic_data.py --lang en --count 20
```

---

### Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd cedd-hackathon

# 2. Creer a Python env.
cedd-hackathon$ python3 -m venv venv
cedd-hackathon$ source venv/bin/activate
(venv) /cedd-hackathon$

# 3. Installer les dépendances
(venv) /cedd-hackathon$ pip install streamlit plotly scikit-learn numpy joblib requests anthropic

# 4. (Optionnel) Configurer la clé API Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# 5. (Optional) Installation Ollama models
ollama pull mistral
ollama pull llama3.2:1b

# 6. Entraîner le modèle
python train.py

# 7. Lancer l'interface
streamlit run app.py
```

---

### Utilisation

```bash
# Entraîner le classifieur
python train.py

# Lancer l'interface web bilingue
streamlit run app.py

# Simuler l'historique pour la démo (français)
python simulate_history.py --lang fr

# Simuler l'historique pour la démo (anglais)
python simulate_history.py --lang en

# Générer des données synthétiques supplémentaires
python generate_synthetic_data.py --lang fr --count 20
python generate_synthetic_data.py --lang en --count 20
```

Ouvre `http://localhost:8501`. Utiliser le bouton de langue (🇫🇷 / 🇬🇧) dans l'en-tête pour basculer. Cliquer sur **Réinitialiser / Reset** pour démarrer une nouvelle session de surveillance.

---

### Métriques

| Métrique                  | Valeur                 |
|---------------------------|------------------------|
| Train accuracy            | 100% (overfitting attendu) |
| CV accuracy (k=4)         | 66.7% ± 26.4%          |
| Nombre de features        | 42 (7 × 6 stats)       |
| Top feature               | `word_count_std`       |
| 2e feature                | `negative_score_mean`  |

---

### Structure du projet

```
cedd-hackathon/
├── app.py                          # Interface Streamlit bilingue
├── train.py                        # Script d'entraînement (sortie bilingue)
├── generate_synthetic_data.py      # Génération FR + EN via Claude API
├── simulate_history.py             # Simulation d'historique FR + EN
│
├── cedd/
│   ├── __init__.py
│   ├── feature_extractor.py        # Extraction lexicale bilingue + trajectoire
│   ├── classifier.py               # CEDDClassifier (GradientBoosting + règles)
│   ├── response_modulator.py       # Prompts adaptatifs FR + EN + appels LLM
│   └── session_tracker.py          # Suivi inter-sessions SQLite
│
├── data/
│   ├── synthetic_conversations.json  # Dataset FR (+EN générable)
│   └── cedd_sessions.db             # Base SQLite (créée automatiquement)
│
└── models/
    └── cedd_model.joblib            # Modèle entraîné (créé par train.py)
```

---

### Limites connues et pistes d'amélioration

| Limite                                                              | Piste d'amélioration                                             |
|---------------------------------------------------------------------|------------------------------------------------------------------|
| Dataset de 24 conversations — overfitting élevé                    | Générer 100+ conversations par classe                            |
| ML peu fiable pour les conversations courtes (< 6 messages)        | ML plafonné à Orange pour < 6 messages ; mots-clés de crise déclenchent Rouge immédiatement |
| Un seul `user_id` "demo_user" dans l'interface de démo             | Ajouter un système d'authentification léger                      |
| Modèle ML entraîné uniquement sur données françaises               | Réentraîner sur un dataset FR+EN combiné                         |
| Aucune validation clinique des seuils                              | Collaboration avec professionnels en santé mentale               |
| LLM non fine-tuné pour le contexte de crise                        | Fine-tuning sur conversations d'intervenants certifiés           |

---

### Ressources d'urgence

> Ces ressources sont intégrées dans les prompts de niveau Orange et Rouge.

- **Jeunesse J'écoute** : 1-800-668-6868 (24h/24, gratuit, confidentiel) — texto : 686868
- **Kids Help Phone** : 1-800-668-6868
- **Urgences / Emergency** : 911
- **Médecin de famille / school counselling service**
