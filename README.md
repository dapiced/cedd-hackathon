# CEDD — Conversational Emotional Drift Detection

---

## English Documentation

CEDD is a real-time monitoring system designed to detect **progressive emotional drift** in youth (16–22 years old) during conversations with an AI support chatbot. It combines lexical analysis, sentence embeddings, machine learning, and adaptive LLM modulation to deliver responses calibrated to the detected emotional state.

**Hackathon**: Mila x Bell x Jeunesse, J'ecoute (Kids Help Phone) — March 16–23, 2026
**Team**: 404HarmNotFound

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
- [Adversarial Testing](#adversarial-testing)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Emergency Resources](#emergency-resources)

---

### Context & Motivation

Emotional support chatbots for youth can, without a monitoring layer, fail to detect a user's gradual mental deterioration. CEDD adds an orthogonal analysis layer to the LLM: it monitors the **trajectory** of user messages (not just their instant content) to identify a drift toward distress.

Detection relies on a hybrid approach:
- **Lexical and structural features** (10 per-message features with bilingual lexicons) — zero latency, full explainability
- **Multilingual sentence embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`) — semantic understanding that catches synonyms, paraphrases, and sarcasm
- **Conversational coherence features** — behavioral withdrawal patterns like short responses and topic avoidance
- **GradientBoosting classifier** with 6-gate safety logic — safety rules that can never be overridden by ML

---

### Architecture

```
+-----------------------------------------------------------------+
|              Streamlit Interface (app.py) -- FR / EN             |
|  Chat  |  Alert gauge  |  Probas  |  Features  |  Longitudinal  |
+--------+----------------------------------------------------+---+
         | user messages
         v
+------------------------+
|   Feature Extractor    |  <- 10 features/message -> 67 trajectory features
|  (numpy + regex +      |     Bilingual lexicons (FR + EN)
|   sentence embeddings) |     + embedding drift + coherence
+------------+-----------+
             | 67D vector
             v
+------------------------+        +--------------------------+
|   CEDDClassifier       |------->|  Safety rules            |
|  (GradientBoosting)    |        |  (lexical override)      |
+------------+-----------+        +--------------------------+
             | level 0-3 + confidence + top features
             v
+------------------------+        +--------------------------+
|  Response Modulator    |------->|  LLM Claude / Mistral /  |
|  (adaptive prompt)     |        |  Llama / static fallback |
|  FR or EN system prompt|        +--------------------------+
+------------------------+
             |
             v
+------------------------+
|   Session Tracker      |  <- SQLite, cross-session history
|  (longitudinal SQLite) |
+------------------------+
```

---

### Alert Levels

| Level | Color     | Label  | Description                                | LLM Mode                         |
|-------|-----------|--------|--------------------------------------------|----------------------------------|
| 0     | Green     | green  | Normal conversation, youth doing well      | Supportive standard              |
| 1     | Yellow    | yellow | Concerning signs, fatigue, loneliness      | Enhanced emotional validation    |
| 2     | Orange    | orange | Significant distress, negative thoughts    | Active support + resources       |
| 3     | Red       | red    | Potential crisis, finality thoughts        | Crisis -- urgent referral        |

---

### Modules

#### 1. Feature Extractor -- `cedd/feature_extractor.py`

Analytical core of the system. Extracts **10 base features** per user message:

| # | Feature                  | Description                                                                                           |
|---|--------------------------|-------------------------------------------------------------------------------------------------------|
| 0 | `word_count`             | Word count of the message                                                                             |
| 1 | `punctuation_ratio`      | Proportion of punctuation characters over total characters                                            |
| 2 | `question_presence`      | Binary 0/1 indicator: does the message contain a `?`                                                  |
| 3 | `negative_score`         | Ratio of negative words/phrases over message length                                                   |
| 4 | `finality_score`         | Ratio of finality/distress vocabulary (`disappear`, `end it`, `burden`, `mourir`...)                 |
| 5 | `hope_score`             | Ratio of hope/resources vocabulary (`tomorrow`, `try`, `family`, `demain`...)                        |
| 6 | `length_delta`           | Relative length change vs previous message -- detects progressive shortening                          |
| 7 | `negation_score`         | Detects negated positive states: `"je ne me sens pas bien"`, `"can't cope"`, `"no hope"`            |
| 8 | `identity_conflict_score`| Detects 2SLGBTQ+ and cultural identity distress: `"my family won't accept me"`, `"je dois me cacher"` |
| 9 | `somatization_score`     | Detects emotional distress co-occurring with physical complaints (pure physical = 0.0)                |

**Bilingual lexicons** (all contain FR + EN terms):
- `FINALITY_WORDS` -- crisis/ending language (36 terms)
- `HOPE_WORDS` -- resilience/future (24 terms)
- `NEGATIVE_WORDS` -- distress sentiment (40 terms)
- `PHYSICAL_CONTEXT_WORDS` -- body-related complaints (14 terms)
- `IDENTITY_CONFLICT_WORDS` -- 2SLGBTQ+ / cultural identity (13+ phrases)
- `SOMATIZATION_EMOTIONAL_WORDS` -- emotional co-occurrence with physical (18 terms)
- `NEGATION_PATTERNS_FR` / `NEGATION_PATTERNS_EN` -- regex patterns for negation structures

#### Trajectory Features (60 = 10 x 6 stats)

For each of the 10 base features, 6 trajectory statistics are computed over the full conversation:

| Statistic | Description                                                        |
|-----------|--------------------------------------------------------------------|
| `_mean`   | Average across all messages                                        |
| `_std`    | Standard deviation -- measures variability                         |
| `_slope`  | Normalised linear regression slope -- captures the trend           |
| `_last`   | Value of the last message -- most recent state                     |
| `_max`    | Maximum observed in the conversation                               |
| `_min`    | Minimum observed in the conversation                               |

#### Embedding Features (4)

Computed using `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, lazy-loaded):

| Feature              | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `embedding_drift`    | Mean cosine distance between consecutive user messages              |
| `crisis_similarity`  | Cosine similarity of last message to a crisis language centroid     |
| `embedding_slope`    | PCA->1D slope over message order (directional semantic drift)      |
| `embedding_variance` | Mean pairwise cosine distance (overall conversation coherence)     |

#### Coherence Features (3)

Behavioral withdrawal patterns computed at conversation level:

| Feature                   | Description                                                      |
|---------------------------|------------------------------------------------------------------|
| `short_response_ratio`    | Fraction of user messages with < 5 words (disengagement signal)  |
| `min_topic_coherence`     | Min cosine similarity between consecutive user messages           |
| `question_response_ratio` | Fraction of assistant questions followed by a responsive reply    |

**Total features: 67** = 10 x 6 trajectory + 4 embedding + 3 coherence

---

#### 2. Classifier -- `cedd/classifier.py`

**sklearn pipeline:**
```
StandardScaler -> GradientBoostingClassifier(n_estimators=200, max_depth=3)
```

**6-Gate Safety Logic:**

| Gate | Condition | Action |
|------|-----------|--------|
| 1 | < 3 user messages | Return Green (insufficient context) + keyword check only |
| 2 | Crisis keyword detected (`suicide`, `gun`, `kill myself`, etc.) | Force Red (confidence 0.90) |
| 3 | ML prediction | Run GradientBoosting on 67D feature vector |
| 4 | ML confidence < 0.45 | Default to Yellow (precautionary principle) |
| 5 | < 6 user messages | Cap ML at Orange max (trajectory features noisy on short convos) |
| 6 | ML < safety minimum | Enforce safety floor: ML can never go below keyword-based level |

**Safety rules (lexical override):**

*Before ML (< 3 user messages):*
- Crisis keyword -> immediate **Red** (0.90 confidence)
- Critical word -> minimum **Orange** (0.70)
- 2+ distress words -> minimum **Yellow** (0.65)
- Otherwise -> **Green** safe mode (0.80)

*After ML (3+ messages):*
- Same keyword scan sets a `minimum_level`
- ML prediction can **never go below this minimum**: `predicted = max(ml_pred, minimum_level)`
- **Safety override display**: when rules raise the level above ML, probability bars are replaced by a "crisis word detected" badge

**Feature display names** are available in both French and English (30+ entries), selectable via the `lang` parameter.

---

#### 3. Response Modulator -- `cedd/response_modulator.py`

**Adaptive system prompts** (French and English) injected into the LLM based on alert level:

- **Level 0**: Warm supportive assistant, open questions, positive register
- **Level 1**: Emotional validation priority, one question at a time, active listening
- **Level 2**: Safe space, resources mentioned naturally (Kids Help Phone: 1-800-668-6868)
- **Level 3**: Crisis protocol with **5-step warm handoff**:
  1. Empathetic validation (no resources yet)
  2. Permission-based transition ("Would it be okay if I connected you with someone?")
  3. Resource presentation (KHP 1-800-668-6868, text 686868, 9-8-8, 911)
  4. Encouragement to connect (normalize hesitation, suggest text-first)
  5. Continued presence ("I'm still here if you want to keep talking")

**LLM hierarchy with automatic fallback:**
```
claude-haiku (Anthropic API) -> mistral (local Ollama) -> llama3.2:1b (local Ollama) -> static text
```

| Model              | Requires            | Indicator |
|--------------------|---------------------|-----------|
| `claude-haiku`     | `ANTHROPIC_API_KEY` | Purple    |
| `mistral`          | Local Ollama        | Blue      |
| `llama3.2:1b`      | Local Ollama        | White     |
| `static fallback`  | None                | Warning   |

---

#### 4. Session Tracker -- `cedd/session_tracker.py`

**Cross-session longitudinal monitoring** using SQLite.

**Schema:**
- `sessions`: one row per chat session
- `alert_events`: one record per analysed message
- `handoff_events`: warm handoff step transitions (step, alert_level)
- `last_activity`: per-user last message timestamp for withdrawal detection

**Silence/withdrawal detection:**
- Tracks `last_activity` per user with `had_closing` flag
- `check_withdrawal_risk()` flags users returning after >24h without closing
- Surfaces welcome-back banner and withdrawal badge in the dashboard

**Longitudinal risk analysis** over the last 7 sessions:
- `risk_score`: weighted average of max levels (recent = higher weight), normalised to [0, 1]
- `trend`: `improving` / `stable` / `worsening`
- `consecutive_high_sessions`: consecutive recent sessions with level >= Orange
- `recommendation`: action suggested to a healthcare professional

---

#### 5. Streamlit Interface -- `app.py`

Two-column interface with real-time updates after each message.

**Language toggle** in the header: switch between English and Francais at any time. The UI, system prompts, and LLM responses all switch to the selected language.

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
| Training data          | 240 EN conversations           | 240 FR conversations            |
| Adversarial tests      | 6 EN + mixed                   | 7 FR + mixed                    |

---

### Synthetic Dataset

**`data/synthetic_conversations.json`** -- 480 balanced bilingual conversations (60 per class x 4 classes x 2 languages).

Each conversation contains ~12 user + 12 assistant messages, generated via `generate_synthetic_data.py` using Claude Haiku in authentic Canadian French and English.

#### Generation Archetypes

| Archetype | Characteristics                                                                                           |
|-----------|-----------------------------------------------------------------------------------------------------------|
| `green`   | Future projects, humour, friends/family mentioned, varied messages, normal emotions (exam stress)         |
| `yellow`  | Persistent fatigue, growing loneliness, self-doubt, gradual drift over 12 messages                       |
| `orange`  | Feeling of emptiness, crying, feeling like a burden, shorter messages, sense of uselessness               |
| `red`     | Desire to disappear, total isolation, tone of finality, short intense messages, no future plans           |

#### Generating Additional Data (FR and EN)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Generate 80 French conversations (20 per class)
python generate_synthetic_data.py --lang fr --count 20

# Generate 80 English conversations (20 per class)
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

# 2. Create a Python env
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

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

# Simulate session history for demo
python simulate_history.py --lang fr   # French
python simulate_history.py --lang en   # English

# Generate additional synthetic data
python generate_synthetic_data.py --lang fr --count 20
python generate_synthetic_data.py --lang en --count 20

# Run adversarial tests
python tests/adversarial_suite.py --verbose
```

Opens at `http://localhost:8501`. Use the language toggle in the header to switch languages. Click **Reset / Reinitialiser** to start a new monitoring session.

---

### Metrics

Results on the 480-conversation balanced bilingual dataset:

| Metric                    | Value                         |
|---------------------------|-------------------------------|
| CV accuracy (k=4)         | **91.7% +/- 4.4%**           |
| Train accuracy            | 100% (expected overfitting)   |
| Number of features        | **67** (10x6 + 4 emb + 3 coh)|
| Training conversations    | **480** (60/class x FR + EN)  |
| Sample:feature ratio      | **7.2:1** (improved from 4.8) |
| Top feature               | `word_count_slope` (0.367)    |
| 2nd feature               | `word_count_max` (0.249)      |
| 3rd feature               | `length_delta_mean` (0.126)   |
| 4th feature               | `finality_score_mean` (0.119) |
| Adversarial tests         | **13/13 passing**             |
| Critical misses           | **0**                         |
| Languages                 | French + English (bilingual)  |

#### Metrics History

| Date | Event | CV Accuracy | Adversarial |
|------|-------|-------------|-------------|
| March 10 | Baseline (24 FR convos, 42 features) | 66.7% +/- 26.4% | 7/10 |
| March 12 | Data expansion (320 bilingual convos) | ~91.2% +/- 1.5% | 9/10 |
| March 12 | Crisis keyword expansion | ~91.2% +/- 1.5% | 10/10 |
| March 12 | +Negation + Embeddings (52 features) | ~92.2% +/- 1.8% | 9/10 |
| March 12 | +Identity + Somatization + Coherence (67 features) | 92.5% +/- 1.5% | 13/13 |
| March 12 | Data expansion to 480 convos (60/class) | **91.7% +/- 4.4%** | **13/13** |

---

### Project Structure

```
cedd-hackathon/
+-- app.py                          # Bilingual Streamlit interface
+-- train.py                        # Training: load -> cross-validate -> fit -> save
+-- generate_synthetic_data.py      # Data generation via Claude API (FR + EN)
+-- simulate_history.py             # Demo history simulation (FR + EN)
+-- annotate_data.py                # Quality annotation tool (Claude-based)
+-- requirements.txt                # Python dependencies
|
+-- cedd/                           # Main Python package
|   +-- __init__.py
|   +-- feature_extractor.py        # 10 features/msg + embeddings + coherence -> 67D
|   +-- classifier.py               # CEDDClassifier (GradientBoosting + 6-gate safety)
|   +-- response_modulator.py       # Adaptive prompts (FR + EN) + LLM fallback chain
|   +-- session_tracker.py          # Cross-session SQLite longitudinal tracking
|
+-- tests/                          # Adversarial test suite (Track 1)
|   +-- adversarial_suite.py        # CLI test runner (--verbose, --category, --export)
|   +-- test_cases_adversarial.json # 13 adversarial test cases (FR + EN)
|   +-- results/
|       +-- baseline_v1.json        # Original: 7/10 passed
|       +-- post_data_expansion.json# 320 convos: 9/10 passed
|       +-- post_keyword_fix.json   # Crisis keywords: 10/10 passed
|       +-- post_negation_embeddings.json  # +Negation +Embeddings
|       +-- post_features_456.json  # 67 features: 13/13 passed, 0 critical misses
|       +-- post_480_convos.json   # Current: 480 convos, 13/13 passed
|
+-- data/
|   +-- synthetic_conversations.json  # 480 labeled conversations (balanced FR + EN)
|   +-- annotated_conversations.json  # Quality-annotated subset
|   +-- filtered_conversations.json   # Post-annotation filtered
|   +-- cedd_sessions.db              # SQLite database (auto-created)
|
+-- models/
|   +-- cedd_model.joblib            # Trained model (created by train.py)
|
+-- demo/
    +-- demo_scenario.md             # FR demo: Felix, CEGEP, Green->Yellow->Orange
    +-- demo_scenario_en.md          # EN demo: Alex, university, Green->Yellow->Orange
```

---

### Adversarial Testing

The `tests/` directory provides a systematic red-teaming suite to validate CEDD robustness against real-world edge cases.

#### Test categories (13 tests)

| Category | Description | Count |
|---|---|---|
| `false_positive_physical` | Physical complaints that should NOT trigger alerts (back pain, nausea) | 2 |
| `sarcasm` | Sarcastic language masking real distress | 1 |
| `negation` | Negation of positive states (`"je ne me sens pas bien"`) | 1 |
| `code_switching` | French/English mixing (Quebec franglais) | 1 |
| `quebecois_slang` | Quebec slang (`"chu pu capable"`, `"en criss"`) | 1 |
| `gradual_drift_no_keywords` | Slow emotional deterioration with no crisis keywords | 1 |
| `direct_crisis` | Explicit crisis language -- **must always be Red** | 1 |
| `hidden_intent` | Indirect suicidal ideation framed as hypothetical | 1 |
| `manipulation_downplay` | Distress followed by minimisation -- must NOT drop to Green | 1 |
| `somatization` | Physical pain + emotional decline (somatized distress) | 1 |
| `identity_conflict` | 2SLGBTQ+ identity crisis and family rejection (EN + FR) | 2 |

#### Running the suite

```bash
# Run all tests
python tests/adversarial_suite.py

# Verbose output (probabilities + top features per test)
python tests/adversarial_suite.py --verbose

# Filter by category
python tests/adversarial_suite.py --category identity_conflict

# Export results to JSON for tracking
python tests/adversarial_suite.py --export tests/results/run_001.json
```

#### Exit codes

| Code | Meaning |
|---|---|
| `0` | All tests passed |
| `1` | Some tests failed (non-critical) |
| `2` | **Critical miss** -- crisis predicted as Green/Yellow (safety regression, blocks merge) |

> **Current (v7):** 13/13 passed, 0 critical misses -- see `tests/results/post_480_convos.json`
> **Original baseline (v1):** 7/10 passed -- see `tests/results/baseline_v1.json`

---

### Known Limitations & Future Work

| Limitation                                                          | Potential Improvement                                               |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| ML unreliable for short conversations (< 6 messages)               | ML capped at Orange for < 6 messages; crisis keywords trigger Red instantly |
| No clinical validation of thresholds                               | Collaborate with mental health professionals                        |
| Single `demo_user` ID in demo interface                            | Add lightweight authentication system                               |
| Identity conflict detection is phrase-based, not contextual         | Fine-tune embeddings on identity-distress corpus                    |
| Withdrawal detection is threshold-based (>24h), not intra-session   | Track intra-session message timing and progressive disengagement    |
| Somatization relies on word co-occurrence, not clinical reasoning   | Add validated somatization scales as complementary signal            |
| LLM not fine-tuned for crisis contexts                              | Fine-tune on certified counsellor conversations                     |

---

### Emergency Resources

> Integrated into Orange and Red alert level prompts.

- **Kids Help Phone**: 1-800-668-6868 (24/7, free, confidential) -- text: 686868
- **Suicide Crisis Helpline**: 9-8-8 (988.ca)
- **Multi-Ecoute**: 514-378-3430 (multiecoute.org)
- **Tracom**: 514-483-3033 (tracom.ca)
- **Emergency services**: 911
- **Family doctor / school counselling service**

---

## Documentation en Francais

CEDD est un systeme de surveillance en temps reel concu pour detecter une **derive emotionnelle progressive** chez des jeunes (16-22 ans) lors de conversations avec un chatbot de soutien. Il combine analyse lexicale, embeddings de phrases multilingues, machine learning et modulation adaptative du LLM pour offrir des reponses ajustees a l'etat emotionnel detecte.

**Hackathon** : Mila x Bell x Jeunesse, J'ecoute -- 16-23 mars 2026
**Equipe** : 404HarmNotFound

---

### Table des matieres

- [Contexte et motivation](#contexte-et-motivation)
- [Architecture](#architecture-1)
- [Niveaux d'alerte](#niveaux-dalerte)
- [Modules](#modules-1)
- [Support bilingue](#support-bilingue)
- [Donnees synthetiques](#donnees-synthetiques)
- [Installation](#installation-1)
- [Utilisation](#utilisation)
- [Metriques](#metriques)
- [Structure du projet](#structure-du-projet)
- [Tests adversariaux](#tests-adversariaux)
- [Limites connues et pistes d'amelioration](#limites-connues-et-pistes-damelioration)

---

### Contexte et motivation

Les chatbots de soutien emotionnel pour les jeunes peuvent, sans systeme de surveillance, ne pas detecter une degradation progressive de l'etat mental de l'utilisateur. CEDD propose une couche d'analyse orthogonale au LLM : elle surveille la **trajectoire** des messages de l'utilisateur (pas uniquement leur contenu ponctuel) pour identifier un glissement vers la detresse.

La detection repose sur une approche hybride :
- **Features lexicales et structurelles** (10 features par message avec lexiques bilingues) -- latence nulle, explicabilite complete
- **Embeddings de phrases multilingues** (`paraphrase-multilingual-MiniLM-L12-v2`) -- comprehension semantique qui detecte synonymes, paraphrases et sarcasme
- **Features de coherence conversationnelle** -- patterns de retrait comportemental (reponses courtes, evitement thematique)
- **Classifieur GradientBoosting** avec logique de securite a 6 portes -- les regles de securite ne peuvent jamais etre outrepassees par le ML

---

### Architecture

```
+-----------------------------------------------------------------+
|            Interface Streamlit (app.py) -- FR / EN               |
|  Chat  |  Jauge alerte  |  Probas  |  Features  |  Longitudinal |
+--------+----------------------------------------------------+---+
         | messages utilisateur
         v
+------------------------+
|   Feature Extractor    |  <- 10 features/message -> 67 features de trajectoire
|  (numpy + regex +      |     Lexiques bilingues (FR + EN)
|   embeddings phrases)  |     + derive semantique + coherence
+------------+-----------+
             | vecteur 67D
             v
+------------------------+        +--------------------------+
|   CEDDClassifier       |------->|  Regles de securite      |
|  (GradientBoosting)    |        |  (override lexical)      |
+------------+-----------+        +--------------------------+
             | niveau 0-3 + confiance + features dominantes
             v
+------------------------+        +--------------------------+
|  Response Modulator    |------->|  LLM Claude / Mistral /  |
|  (prompt adaptatif)    |        |  Llama / sans llm        |
|  Prompt FR ou EN       |        +--------------------------+
+------------------------+
             |
             v
+------------------------+
|   Session Tracker      |  <- SQLite, historique inter-sessions
|  (longitudinal SQLite) |
+------------------------+
```

---

### Niveaux d'alerte

| Niveau | Couleur  | Label  | Description                                    | Mode LLM                          |
|--------|----------|--------|------------------------------------------------|-----------------------------------|
| 0      | Vert     | verte  | Conversation normale, jeune en bonne forme     | Standard bienveillant             |
| 1      | Jaune    | jaune  | Signes preoccupants, fatigue, solitude         | Validation emotionnelle renforcee |
| 2      | Orange   | orange | Detresse significative, pensees negatives      | Soutien actif + ressources        |
| 3      | Rouge    | rouge  | Crise potentielle, pensees de finalite         | Crise -- orientation urgente      |

---

### Modules

#### 1. Feature Extractor -- `cedd/feature_extractor.py`

Coeur analytique du systeme. Extrait **10 features de base** par message utilisateur :

| # | Feature | Description |
|---|---------|-------------|
| 0 | `word_count` | Nombre de mots du message |
| 1 | `punctuation_ratio` | Ratio de ponctuation par rapport au total de caracteres |
| 2 | `question_presence` | Indicateur binaire 0/1 : le message contient-il un `?` |
| 3 | `negative_score` | Ratio de mots negatifs sur la longueur du message |
| 4 | `finality_score` | Ratio de vocabulaire de finalite/detresse |
| 5 | `hope_score` | Ratio de vocabulaire d'espoir/ressources |
| 6 | `length_delta` | Variation relative de longueur vs message precedent |
| 7 | `negation_score` | Detecte les negations d'etats positifs : `"ne...pas bien"`, `"can't cope"` |
| 8 | `identity_conflict_score` | Detecte la detresse identitaire 2SLGBTQ+ : `"ma famille ne m'accepte pas"` |
| 9 | `somatization_score` | Detecte la detresse emotionnelle co-occurrente avec des plaintes physiques |

**Lexiques bilingues** (tous contiennent des termes FR + EN) :
- `FINALITY_WORDS` -- langage de crise/fin (36 termes)
- `HOPE_WORDS` -- resilience/avenir (24 termes)
- `NEGATIVE_WORDS` -- sentiment de detresse (40 termes)
- `PHYSICAL_CONTEXT_WORDS` -- plaintes corporelles (14 termes)
- `IDENTITY_CONFLICT_WORDS` -- identite 2SLGBTQ+ / culturelle (13+ phrases)
- `SOMATIZATION_EMOTIONAL_WORDS` -- co-occurrence emotionnelle avec physique (18 termes)
- `NEGATION_PATTERNS_FR` / `NEGATION_PATTERNS_EN` -- patterns regex de negation

#### Features de trajectoire (60 = 10 x 6 stats)

Pour chaque feature de base, 6 statistiques de trajectoire (mean, std, slope, last, max, min).

#### Features d'embedding (4)

Calculees avec `paraphrase-multilingual-MiniLM-L12-v2` :
- `embedding_drift` -- derive cosinus moyenne entre messages consecutifs
- `crisis_similarity` -- similarite cosinus du dernier message avec le centroide de crise
- `embedding_slope` -- pente PCA->1D (derive semantique directionnelle)
- `embedding_variance` -- distance cosinus moyenne par paires (coherence)

#### Features de coherence (3)

Patterns de retrait comportemental au niveau conversation :
- `short_response_ratio` -- fraction de messages < 5 mots (desengagement)
- `min_topic_coherence` -- similarite cosinus min entre messages consecutifs
- `question_response_ratio` -- fraction de questions assistant suivies d'une reponse engagee

**Total : 67 features** = 10 x 6 trajectoire + 4 embedding + 3 coherence

---

#### 2. Classifier -- `cedd/classifier.py`

Pipeline : `StandardScaler -> GradientBoostingClassifier(n_estimators=200, max_depth=3)`

**Logique de securite a 6 portes :**

| Porte | Condition | Action |
|-------|-----------|--------|
| 1 | < 3 messages utilisateur | Retourner Vert (contexte insuffisant) + verification mots-cles |
| 2 | Mot-cle de crise detecte | Forcer Rouge (confiance 0.90) |
| 3 | Prediction ML | Executer GradientBoosting sur vecteur 67D |
| 4 | Confiance ML < 0.45 | Defaut a Jaune (principe de precaution) |
| 5 | < 6 messages utilisateur | Plafonner ML a Orange max |
| 6 | ML < minimum securite | Appliquer plancher de securite |

Mots-cles de crise etendus (arme, pistolet, couteau, gun, knife, shoot...) declenchent **Rouge immediatement** a tout moment. En cas d'override de securite, les barres de probabilite sont remplacees par un badge "mot de crise detecte".

Les **noms lisibles des features** sont disponibles en francais et en anglais (30+ entrees), selectionnables via le parametre `lang`.

#### 3. Response Modulator -- `cedd/response_modulator.py`

Quatre niveaux de prompts systeme distincts, disponibles en **francais et en anglais**. Le niveau Rouge utilise un **transfert accompagne en 5 etapes** :
1. Validation empathique (pas de ressources encore)
2. Transition accompagnee (demande de permission)
3. Presentation des ressources (JJE, 9-8-8, 911)
4. Encouragement a se connecter
5. Presence continue

Hierarchie LLM : `claude-haiku -> mistral -> llama3.2:1b -> sans llm`

#### 4. Session Tracker -- `cedd/session_tracker.py`

Surveillance longitudinale inter-sessions via SQLite. Calcule `risk_score`, `trend`, `consecutive_high_sessions` et `recommendation` sur les 7 dernieres sessions.

**Detection d'abandon/retrait :** suit le `last_activity` par utilisateur. Si un utilisateur revient apres >24h sans avoir ferme sa session, une banniere de bienvenue et un badge d'abandon s'affichent.

#### 5. Interface Streamlit -- `app.py`

Interface bilingue en deux colonnes. **Bouton de langue** dans l'en-tete pour basculer entre Francais et English.

---

### Support bilingue

| Couche                  | Anglais                              | Francais                              |
|-------------------------|--------------------------------------|---------------------------------------|
| Interface web           | Complete (bouton dans l'en-tete)     | Complete (langue par defaut)          |
| Prompts systeme LLM     | 4 niveaux                            | 4 niveaux                             |
| Analyse lexicale        | Mots EN dans les lexiques            | Mots FR dans les lexiques             |
| Noms des features       | Via parametre `lang="en"`            | Via parametre `lang="fr"`             |
| Donnees d'entrainement  | 160 conversations EN                 | 160 conversations FR                  |
| Tests adversariaux      | 6 EN + mixtes                        | 7 FR + mixtes                         |

---

### Donnees synthetiques

**`data/synthetic_conversations.json`** -- 480 conversations equilibrees bilingues (60 par classe x 4 classes x 2 langues).

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# 80 conversations en francais (20 par classe)
python generate_synthetic_data.py --lang fr --count 20

# 80 conversations en anglais (20 par classe)
python generate_synthetic_data.py --lang en --count 20
```

---

### Installation

```bash
# 1. Cloner le depot
git clone <url-du-repo>
cd cedd-hackathon

# 2. Creer un environnement Python
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dependances
pip install -r requirements.txt

# 4. (Optionnel) Configurer la cle API Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# 5. (Optionnel) Installer les modeles Ollama
ollama pull mistral
ollama pull llama3.2:1b

# 6. Entrainer le modele
python train.py

# 7. Lancer l'interface
streamlit run app.py
```

```powershell
# Windows OS uniquement : voir setup-cedd-for-win.ps1
```

---

### Utilisation

```bash
# Entrainer le classifieur
python train.py

# Lancer l'interface web bilingue
streamlit run app.py

# Simuler l'historique pour la demo
python simulate_history.py --lang fr
python simulate_history.py --lang en

# Generer des donnees synthetiques supplementaires
python generate_synthetic_data.py --lang fr --count 20
python generate_synthetic_data.py --lang en --count 20

# Lancer les tests adversariaux
python tests/adversarial_suite.py --verbose
```

Ouvre `http://localhost:8501`. Utiliser le bouton de langue dans l'en-tete pour basculer. Cliquer sur **Reinitialiser / Reset** pour demarrer une nouvelle session.

---

### Metriques

Resultats sur le dataset de 480 conversations equilibrees bilingues :

| Metrique                  | Valeur                         |
|---------------------------|--------------------------------|
| CV accuracy (k=4)         | **91.7% +/- 4.4%**            |
| Train accuracy            | 100% (overfitting attendu)     |
| Nombre de features        | **67** (10x6 + 4 emb + 3 coh) |
| Conversations             | **480** (60/classe x FR + EN)  |
| Ratio echantillons:features | **7.2:1** (ameliore de 4.8)  |
| Top feature               | `word_count_slope` (0.367)     |
| 2e feature                | `word_count_max` (0.249)       |
| 3e feature                | `length_delta_mean` (0.126)    |
| 4e feature                | `finality_score_mean` (0.119)  |
| Tests adversariaux        | **13/13 reussis**              |
| Crises manquees           | **0**                          |

#### Historique des metriques

| Date | Evenement | CV Accuracy | Adversarial |
|------|-----------|-------------|-------------|
| Mars 10 | Baseline (24 convos FR, 42 features) | 66.7% +/- 26.4% | 7/10 |
| Mars 12 | Expansion donnees (320 convos bilingues) | ~91.2% +/- 1.5% | 9/10 |
| Mars 12 | Expansion mots-cles de crise | ~91.2% +/- 1.5% | 10/10 |
| Mars 12 | +Negation + Embeddings (52 features) | ~92.2% +/- 1.8% | 9/10 |
| Mars 12 | +Identite + Somatisation + Coherence (67 features) | 92.5% +/- 1.5% | 13/13 |
| Mars 12 | Expansion donnees a 480 convos (60/classe) | **91.7% +/- 4.4%** | **13/13** |

---

### Structure du projet

```
cedd-hackathon/
+-- app.py                          # Interface Streamlit bilingue
+-- train.py                        # Entrainement : chargement -> CV -> fit -> sauvegarde
+-- generate_synthetic_data.py      # Generation via Claude API (FR + EN)
+-- simulate_history.py             # Simulation d'historique FR + EN
+-- annotate_data.py                # Outil d'annotation qualite (Claude)
+-- requirements.txt                # Dependances Python
|
+-- cedd/                           # Package Python principal
|   +-- __init__.py
|   +-- feature_extractor.py        # 10 features/msg + embeddings + coherence -> 67D
|   +-- classifier.py               # CEDDClassifier (GradientBoosting + 6 portes securite)
|   +-- response_modulator.py       # Prompts adaptatifs FR + EN + chaine LLM
|   +-- session_tracker.py          # Suivi longitudinal inter-sessions SQLite
|
+-- tests/                          # Suite de tests adversariaux (Track 1)
|   +-- adversarial_suite.py        # Runner CLI (--verbose, --category, --export)
|   +-- test_cases_adversarial.json # 13 cas de test adversariaux (FR + EN)
|   +-- results/                    # Historique des resultats
|
+-- data/
|   +-- synthetic_conversations.json  # 480 conversations etiquetees (FR + EN)
|   +-- cedd_sessions.db              # Base SQLite (creee automatiquement)
|
+-- models/
|   +-- cedd_model.joblib            # Modele entraine (cree par train.py)
|
+-- demo/
    +-- demo_scenario.md             # Scenario FR : Felix, CEGEP
    +-- demo_scenario_en.md          # Scenario EN : Alex, universite
```

---

### Tests adversariaux

Le repertoire `tests/` fournit une suite de tests systematiques pour valider la robustesse de CEDD face a des cas reels difficiles.

#### Categories de tests (13 tests)

| Categorie | Description | Nb |
|---|---|---|
| `false_positive_physical` | Plaintes physiques qui NE doivent PAS declencher d'alerte | 2 |
| `sarcasm` | Langage sarcastique masquant une detresse reelle | 1 |
| `negation` | Negation d'etats positifs (`"je ne me sens pas bien"`) | 1 |
| `code_switching` | Alternance francais/anglais (franglais quebecois) | 1 |
| `quebecois_slang` | Expressions quebecoises (`"chu pu capable"`, `"en criss"`) | 1 |
| `gradual_drift_no_keywords` | Deterioration emotionnelle lente sans mots-cles de crise | 1 |
| `direct_crisis` | Langage de crise explicite -- **doit toujours etre Rouge** | 1 |
| `hidden_intent` | Ideation suicidaire indirecte presentee comme hypothetique | 1 |
| `manipulation_downplay` | Detresse suivie de minimisation -- ne doit PAS redescendre a Vert | 1 |
| `somatization` | Douleur physique + declin emotionnel (detresse somatisee) | 1 |
| `identity_conflict` | Crise identitaire 2SLGBTQ+ et rejet familial (EN + FR) | 2 |

#### Codes de sortie

| Code | Signification |
|---|---|
| `0` | Tous les tests reussis |
| `1` | Certains tests echoues (non critique) |
| `2` | **Crise manquee** -- crise predite comme Vert/Jaune (regression de securite) |

> **Actuel (v7) :** 13/13 reussis, 0 crise manquee -- voir `tests/results/post_480_convos.json`

---

### Limites connues et pistes d'amelioration

| Limite                                                              | Piste d'amelioration                                             |
|---------------------------------------------------------------------|------------------------------------------------------------------|
| ML peu fiable pour conversations courtes (< 6 messages)            | ML plafonne a Orange; mots-cles de crise declenchent Rouge       |
| Aucune validation clinique des seuils                              | Collaboration avec professionnels en sante mentale               |
| Detection identitaire basee sur des phrases, pas le contexte       | Fine-tuner les embeddings sur un corpus detresse identitaire     |
| Detection d'abandon basee sur seuil (>24h), pas intra-session     | Suivre le delai intra-session et le desengagement progressif     |
| LLM non fine-tune pour le contexte de crise                       | Fine-tuning sur conversations d'intervenants certifies           |

---

### Ressources d'urgence

> Ces ressources sont integrees dans les prompts de niveau Orange et Rouge.

- **Jeunesse J'ecoute / Kids Help Phone** : 1-800-668-6868 (24h/24, gratuit, confidentiel) -- texto : 686868
- **Ligne de crise suicide** : 9-8-8 (988.ca)
- **Multi-Ecoute** : 514-378-3430 (multiecoute.org)
- **Tracom** : 514-483-3033 (tracom.ca)
- **Urgences / Emergency** : 911
- **Medecin de famille / service de consultation scolaire**
