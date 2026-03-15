# CEDD — Conversational Emotional Drift Detection

---

## English Documentation (Francais par la suite)

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
- **GradientBoosting classifier** with 7-gate safety logic — safety rules that can never be overridden by ML

---

### Architecture

```
+-----------------------------------------------------------------+
|              Streamlit Interface (app.py) -- FR / EN            |
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
|  Response Modulator    |------->|  LLM Cohere / Groq /     |
|  (adaptive prompt)     |        |  Gemini / Claude / static|
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

**7-Gate Safety Logic:**

| Gate | Condition | Action |
|------|-----------|--------|
| 1 | < 3 user messages | Return Green (insufficient context) + keyword check only |
| 2 | Crisis keyword detected (`suicide`, `gun`, `kill myself`, etc.) | Force Red (confidence 0.90) |
| 3 | ML prediction | Run GradientBoosting on 67D feature vector |
| 4 | ML confidence < 0.45 | Default to Yellow (precautionary principle) |
| 5 | < 6 user messages | Cap ML at Orange max (trajectory features noisy on short convos) |
| 6 | ML < safety minimum | Enforce safety floor: ML can never go below keyword-based level |
| 7 | Response delay >= 300s AND Yellow+ | Bump +1 level (cap at Red). 120s+ AND Orange+ also bumps. Green never bumped. |

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

**Feature importance output**: `get_alert_level()` returns `feature_scores` — top 5 features by composite score (`model_importance × |scaled_value|`), each with display name, raw name, and score. Available for both ML predictions and safety overrides, displayed as a horizontal bar chart in the dashboard.

---

#### 3. Response Modulator -- `cedd/response_modulator.py`

**Adaptive system prompts** (French and English) injected into the LLM based on alert level:

- **Level 0**: Warm supportive assistant, open questions, positive register
- **Level 1**: Emotional validation priority, one question at a time, active listening
- **Level 2**: Safe space, resources mentioned naturally (Kids Help Phone: 1-800-668-6868)
- **Level 3**: Crisis protocol with **5-step warm handoff** + **simulated counselor "Alex"**:
  1. Empathetic validation (no resources yet)
  2. Permission-based transition ("Would it be okay if I connected you with someone?")
  3. Resource presentation (KHP 1-800-668-6868, text 686868, 9-8-8, 911)
  4. Encouragement to connect (normalize hesitation, suggest text-first)
  5. Continued presence ("I'm still here if you want to keep talking")

  At Red, CEDD also offers to connect with **Alex**, a simulated KHP counselor using ASIST active listening techniques. If the user accepts, the chat switches to a counselor persona with distinct visual styling (blue bubbles, 🧑‍⚕️ avatar, counselor banner). The counselor mode bypasses CEDD classification and uses `HUMAN_COUNSELOR_PROMPT` via the same LLM fallback chain. Only Reset exits counselor mode.

**LLM hierarchy with automatic fallback (15s timeout per model):**
```
cohere -> groq (Llama 3.3 70B) -> gemini-flash (Gemini 2.5 Flash) -> claude-haiku -> static text
```
Each LLM client has a 25-second timeout. If a model hangs or is slow, the chain automatically falls through to the next provider without freezing the UI. The full conversation history is passed to each model, so switching mid-conversation is seamless.

| Model              | Requires            | Indicator |
|--------------------|---------------------|-----------|
| `cohere`           | `COHERE_API_KEY`    | Blue      |
| `groq`             | `GROQ_API_KEY`      | Orange    |
| `gemini-flash`     | `GEMINI_API_KEY`    | Blue      |
| `claude-haiku`     | `ANTHROPIC_API_KEY` | Purple    |
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

Two-column interface with real-time updates after each message. Professional UI powered by **Inter** font (Google Fonts), **CSS custom properties** (design tokens for spacing, radii, typography scale), **flexbox** chat layout, and **micro-interactions** (hover lift + shadow transitions on bubbles and pills).

**Profile selector** in the header: 5 demo profiles with bilingual trajectory labels — e.g. "Dominic (escalating)" in EN, "Dominic (escalade)" in FR. Switching profiles ends the current session and loads the selected user's history.

**Language toggle** in the header: switch between English and Francais at any time. The UI, system prompts, and LLM responses all switch to the selected language.

| Component                  | Description                                                                |
|----------------------------|----------------------------------------------------------------------------|
| **Welcome card**           | Branded card with brain emoji, title, description, CTA, and profile legend showing all 5 demo trajectories. Uses CSS classes (`.welcome-card`) for clean styling |
| **Chat bubbles**           | Flexbox layout with `align-self` alignment (no floats), subtle box-shadow, hover lift animation, custom 6px scrollbar, `line-height: 1.5` for readability |
| **Chat timestamps**        | HH:MM timestamp below each message bubble (right-aligned for user, left for assistant) |
| **LLM source badge**       | Small coloured badge on each assistant bubble showing which LLM generated it (e.g. 🔵 Cohere) |
| **Alert level badge**      | Coloured alert dot (e.g. 🟢 Green) on each assistant message showing CEDD classification at that point |
| **Demo autopilot**         | "Play Demo" button auto-plays the Félix/Alex scenario (9 messages). Judges watch drift unfold live |
| **About CEDD panel**       | Collapsible explainer: what CEDD does, how it works, what the dashboard shows. Bilingual |
| **Export transcript**      | Download button exports conversation + alert history as JSON file                          |
| **Alert transition toast** | CSS-animated notification when alert level increases (3s fade-in/out). Red-level toasts include a pulsing glow animation (`pulse-red` keyframes) |
| **Compare mode**           | Side-by-side: raw LLM (no instructions) vs LLM with CEDD adaptive prompts. Toggle via 🔀 button |
| **Feature radar**          | Plotly spider chart: 10 per-message features normalized 0-1, latest msg vs Msg 1 ghost overlay   |
| **Counselor handoff**      | At RED, offers to connect with simulated KHP counselor "Alex" (ASIST persona). Blue bubbles with stronger shadow, 🧑‍⚕️ avatar, counselor banner (`.counselor-banner` CSS class). Bypasses CEDD in counselor mode |
| **Circular gauge**         | Plotly 0-3 indicator with confidence bar                                   |
| **Probability bars**       | Per-class probabilities (green/yellow/orange/red) with theme-aware bar tracks (`.proba-bar-track` — no more dark-mode bug) |
| **Active signals**         | Dominant features displayed as colour-coded pills with hover lift effect   |
| **Feature importance**     | Collapsible Plotly horizontal bar chart: top 5 features by composite score (model importance × scaled value), 6 colour categories. Visible at Yellow+ including safety overrides |
| **Level history**          | Plotly line chart: alert level history for the current session             |
| **Emotional flow (streamgraph)** | Stacked area chart (`go.Scatter` stackgroup) showing Green/Yellow/Orange/Red class probabilities evolving over messages. Safety overrides synthesize 100% at detected level. 50% alpha fills, percentage Y-axis, bilingual legend |
| **Longitudinal history**   | Per-session bar chart + trend + recommendation (SQLite data)               |
| **LLM selector**           | 5 buttons to choose/force the conversational model                         |
| **Active system prompt**   | Description of the current mode + expander showing the full prompt (word-wrapped, theme-styled). Expander summaries use forced `background-color` + JS MutationObserver to prevent Streamlit's default dark header in light mode |
| **Status cards**           | Reusable `.status-card` CSS class for response mode, recommendations, and handoff progress indicators |
| **Session stats**          | Counters: messages, exchanges, alert peak (bold 1.6rem values, uppercase labels) |

---

### Bilingual Support

| Layer                  | English                        | French                          |
|------------------------|--------------------------------|---------------------------------|
| Web UI                 | Full (toggle button in header) | Full (default)                  |
| LLM system prompts     | All 4 levels                   | All 4 levels                    |
| Lexical analysis       | EN words in all lexicons       | FR words in all lexicons        |
| Feature display names  | Via `lang="en"` parameter      | Via `lang="fr"` parameter       |
| Synthetic data         | `--lang en` flag               | `--lang fr` (default)           |
| Training data          | 300 EN conversations           | 300 FR conversations            |
| Adversarial tests      | 14 EN + mixed                  | 16 FR + mixed                   |

---

### Synthetic Dataset

**`data/synthetic_conversations.json`** -- 600 bilingual conversations (480 standard + 120 adversarial).

Each conversation contains ~12 user + 12 assistant messages, generated via `generate_synthetic_data.py` using Claude Haiku in authentic Canadian French and English.

#### Standard Generation Archetypes (480 conversations)

| Archetype | Characteristics                                                                                           |
|-----------|-----------------------------------------------------------------------------------------------------------|
| `green`   | Future projects, humour, friends/family mentioned, varied messages, normal emotions (exam stress)         |
| `yellow`  | Persistent fatigue, growing loneliness, self-doubt, gradual drift over 12 messages                       |
| `orange`  | Feeling of emptiness, crying, feeling like a burden, shorter messages, sense of uselessness               |
| `red`     | Desire to disappear, total isolation, tone of finality, short intense messages, no future plans           |

#### Adversarial Archetypes (120 conversations, `--adversarial` flag)

| Archetype | Label | Characteristics |
|-----------|-------|-----------------|
| `physical_only` | Green | Pure physical complaints, zero emotional distress, some short messages |
| `sarcasm_distress` | Yellow | Dark humour masking real isolation/fatigue, no crisis keywords |
| `adversarial_bypass` | Yellow | Reveal-minimize-reveal pattern, active deflection |
| `identity_distress` | Orange | 2SLGBTQ+/cultural rejection, identity-specific language |
| `neurodivergent_flat` | Orange | Flat affect, topic jumps, concerning situations described factually |
| `crisis_with_deflection` | Red | Crisis language followed by "I'm fine" — still Red |

#### Generating Additional Data (FR and EN)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Standard archetypes (20 per class)
python generate_synthetic_data.py --lang fr --count 20
python generate_synthetic_data.py --lang en --count 20

# Adversarial archetypes (10 per archetype)
python generate_synthetic_data.py --adversarial --lang fr --count 10
python generate_synthetic_data.py --adversarial --lang en --count 10
```

---

### Installation

**Prerequisites:**
- Python 3.9+
- At least one LLM API key (Cohere, Groq, Gemini, or Anthropic)

```bash
# 1. Clone the repository
git clone <repo-url>
cd cedd-hackathon

# 2. Create a Python env
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure LLM API keys (at least one required for live chat)
export COHERE_API_KEY="..."            # Primary: Cohere (default)
export GROQ_API_KEY="gsk_..."          # Secondary: Llama 3.3 70B via Groq (fastest)
export GEMINI_API_KEY="AI..."          # Tertiary: Gemini 2.5 Flash
export ANTHROPIC_API_KEY="sk-ant-..."  # Quaternary: Claude Haiku + data generation

# 5. Train the model
python train.py

# 6. Launch the interface
streamlit run app.py
```
---

### Usage

```bash
# Train the classifier
python train.py

# Launch the bilingual web interface
streamlit run app.py

# Simulate session history for demo (4 user profiles x 7 sessions)
python simulate_history.py --lang fr   # French
python simulate_history.py --lang en   # English

# Generate additional synthetic data
python generate_synthetic_data.py --lang fr --count 20
python generate_synthetic_data.py --lang en --count 20

# Generate adversarial training data
python generate_synthetic_data.py --adversarial --lang fr --count 10
python generate_synthetic_data.py --adversarial --lang en --count 10

# Run adversarial tests
python tests/adversarial_suite.py --verbose

# Run unit tests (feature extractor, classifier, response modulator, session tracker)
pytest tests/test_unit.py -v

# Run integration tests (demo scenarios, cross-language, bilingual UI, edge cases)
pytest tests/test_integration.py -v

# Run all tests (unit + integration)
pytest tests/ -v
```

Opens at `http://localhost:8501`. Use the **profile selector** in the header to switch between demo users — each shows its trajectory label (e.g. "Shuchita (stable green)", "Dominic (escalating)"). Labels switch language with the toggle. Click **Reset / Reinitialiser** to start a new monitoring session.

---

### Metrics

Results on the 600-conversation bilingual dataset (480 standard + 120 adversarial):

| Metric                    | Value                         |
|---------------------------|-------------------------------|
| CV accuracy (k=4)         | **90.0% +/- 1.6%**           |
| Train accuracy            | 100% (expected overfitting)   |
| Number of features        | **67** (10x6 + 4 emb + 3 coh)|
| Training conversations    | **600** (480 standard + 120 adversarial) |
| Sample:feature ratio      | **9.0:1** (improved from 7.2) |
| Top feature               | `word_count_max` (0.192)      |
| 2nd feature               | `word_count_slope` (0.179)    |
| 3rd feature               | `word_count_last` (0.138)     |
| 4th feature               | `finality_score_mean` (0.066) |
| Adversarial tests         | **36/36 passing**             |
| Unit tests (pytest)       | **94/94 passing**             |
| Integration tests (pytest)| **39/39 passing**             |
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
| March 12 | Data expansion to 480 convos (60/class) | 91.7% +/- 4.4% | 13/13 |
| March 13 | Adversarial augmentation to 600 convos (6 new archetypes) | 90.5% +/- 1.5% | 30/30 |
| March 14 | Word-boundary fix + 6 new tests (regex `\b`, context-aware "personne", feminine forms) | 90.0% +/- 1.6% | 36/36 |
| March 15 | +Gate 7 response delay bump (67 features, 7 gates) | **90.0% +/- 1.6%** | **36/36** |

---

### Project Structure

```
cedd-hackathon/
+-- app.py                          # Bilingual Streamlit interface
+-- train.py                        # Training: load -> cross-validate -> fit -> save
+-- generate_synthetic_data.py      # Data generation via Claude API (FR + EN)
+-- simulate_history.py             # Demo history simulation per user profile (FR + EN)
+-- requirements.txt                # Python dependencies
|
+-- cedd/                           # Main Python package
|   +-- __init__.py
|   +-- feature_extractor.py        # 10 features/msg + embeddings + coherence -> 67D
|   +-- classifier.py               # CEDDClassifier (GradientBoosting + 7-gate safety)
|   +-- response_modulator.py       # Adaptive prompts (FR + EN) + LLM fallback chain
|   +-- session_tracker.py          # Cross-session SQLite longitudinal tracking
|
+-- tests/                          # Test suites (Track 1)
|   +-- adversarial_suite.py        # CLI test runner (--verbose, --category, --export)
|   +-- test_unit.py                # 94 pytest unit tests (4 modules: features, classifier, modulator, tracker)
|   +-- test_integration.py         # 39 pytest integration tests (demo, cross-language, bilingual UI, edge cases)
|   +-- test_cases_adversarial.json # 36 adversarial test cases across 20 categories (FR + EN)
|
+-- data/
|   +-- synthetic_conversations.json  # 600 labeled conversations (480 standard + 120 adversarial, FR + EN)
|   +-- cedd_sessions.db              # SQLite database (auto-created)
|
+-- models/
|   +-- cedd_model.joblib            # Trained model (created by train.py)
|
+-- report.md                        # Formal hackathon report (required deliverable)
```

---

### Adversarial Testing

The `tests/` directory provides a systematic red-teaming suite to validate CEDD robustness against real-world edge cases.

#### Test categories (36 tests across 20 categories)

| Category | Description | Count |
|---|---|---|
| `false_positive_physical` | Physical complaints that should NOT trigger alerts (back pain, nausea) | 2 |
| `sarcasm` | Sarcastic language masking real distress | 1 |
| `negation` | Negation of positive states (`"je ne me sens pas bien"`) | 1 |
| `code_switching` | French/English mixing (Quebec franglais) | 1 |
| `quebecois_slang` | Quebec slang (`"chu pu capable"`, `"en criss"`, positive joual) | 3 |
| `gradual_drift_no_keywords` | Slow emotional deterioration with no crisis keywords (EN + FR) | 2 |
| `direct_crisis` | Explicit crisis language -- **must always be Red** (EN + FR) | 2 |
| `hidden_intent` | Indirect suicidal ideation framed as hypothetical | 1 |
| `manipulation_downplay` | Distress followed by minimisation -- must NOT drop to Green | 1 |
| `somatization` | Physical pain + emotional decline (somatized distress) | 1 |
| `identity_conflict` | 2SLGBTQ+ identity crisis and family rejection (EN + FR) | 2 |
| `sudden_escalation` | Normal conversation then sudden crisis escalation | 3 |
| `active_bypass` | Crisis language then retraction ("I was joking") | 2 |
| `rapid_recovery_manipulation` | Deep crisis then "I feel better" -- safety floor must persist | 2 |
| `cultural_false_positive` | "Mort de rire", "killed it", "personne" in neutral contexts | 3 |
| `neurodivergent_pattern` | Literal/flat communication, ADHD bursts, topic jumps | 3 |
| `emoji_only` | Very short messages with ellipses and emoji | 1 |
| `repeated_word` | Repeated words and brief frustration messages | 1 |
| `short_recovery` | Brief crisis then rapid recovery (short conversation) | 1 |
| `long_message` | Single long venting message without crisis words | 1 |
| `neutral_personne_fr` | Neutral use of "personne" (= person) in French | 1 |
| `emoji_crisis` | Crisis words mixed with emoji -- **must always be Red** | 1 |

#### Running the suite

```bash
# Run all tests
python tests/adversarial_suite.py

# Verbose output (probabilities + top features per test)
python tests/adversarial_suite.py --verbose

# Filter by category
python tests/adversarial_suite.py --category identity_conflict
```

#### Exit codes

| Code | Meaning |
|---|---|
| `0` | All tests passed |
| `1` | Some tests failed (non-critical) |
| `2` | **Critical miss** -- crisis predicted as Green/Yellow (safety regression, blocks merge) |

> **Current (v9):** 36/36 passed, 0 critical misses
> **Original baseline (v1):** 7/10 passed

#### Unit Tests (pytest)

94 automated tests covering the 4 core modules:

| Module | Tests | Coverage |
|--------|------:|----------|
| Feature Extractor | 34 | All 10 features (FR + EN), edge cases, trajectory shapes, slope direction |
| Classifier | 16 | All 7 safety gates including Gate 7 response delay, crisis keywords (FR + EN), safety floor, delay_bumped flag |
| Response Modulator | 23 | Prompt selection, crisis resources in Orange/Red, handoff steps 1-5, counselor Alex, static fallback |
| Session Tracker | 21 | Session lifecycle, withdrawal detection, longitudinal risk (trends, consecutive high) |

```bash
pytest tests/test_unit.py -v                    # All 94 tests
pytest tests/test_unit.py -v -k "feature"       # Feature extractor only
pytest tests/test_unit.py -v -k "Gate"           # Classifier gates only
```

> **Known gap documented in tests:** conjugated crisis words ("killing myself") do not match keyword list entry ("kill myself") -- Gate 2 safety floor does not fire for conjugated forms.

#### Integration Tests (pytest)

39 tests validating presentation-readiness across 7 categories:

| Category | Tests | What it catches |
|----------|------:|-----------------|
| Demo Scenarios | 7 | Demo freezing, wrong alert levels, crash during autopilot |
| Cross-Language Consistency | 6 | FR and EN producing wildly different results |
| Bilingual String Completeness | 5 | Missing translations, broken format placeholders |
| End-to-End Integration | 6 | Pipeline producing wrong results, model file issues  |
| Edge Cases | 7 | Crash on emoji, long messages, whitespace, mixed languages |
| Feature Scores Output | 5 | Explainability chart showing empty/broken data |
| Session Tracker Integration | 3 | Session logging failing with real classifier results |

```bash
pytest tests/test_integration.py -v             # All 39 integration tests
pytest tests/test_integration.py -v -k "Demo"   # Demo scenario validation
pytest tests/ -v                                 # All 133 tests (unit + integration)
```

---

### Known Limitations & Future Work

| Limitation                                                          | Potential Improvement                                               |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| ML unreliable for short conversations (< 6 messages)               | ML capped at Orange for < 6 messages; crisis keywords trigger Red instantly |
| No clinical validation of thresholds                               | Collaborate with mental health professionals                        |
| No real authentication (demo profiles only)                        | Add lightweight authentication system                               |
| Identity conflict detection is phrase-based, not contextual         | Fine-tune embeddings on identity-distress corpus                    |
| Withdrawal detection is threshold-based (>24h), not intra-session   | Track intra-session message timing and progressive disengagement    |
| Somatization relies on word co-occurrence, not clinical reasoning   | Add validated somatization scales as complementary signal            |
| LLM not fine-tuned for crisis contexts                              | Fine-tune on certified counsellor conversations                     |
| Word-boundary matching improved but not perfect (idioms like "mort de rire") | Context-aware phrase exclusion or idiom detection                    |

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
- **Classifieur GradientBoosting** avec logique de securite a 7 portes -- les regles de securite ne peuvent jamais etre outrepassees par le ML

---

### Architecture

```
+-----------------------------------------------------------------+
|            Interface Streamlit (app.py) -- FR / EN              |
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
|  Response Modulator    |------->|  LLM Cohere / Groq /     |
|  (prompt adaptatif)    |        |  Gemini / Claude / sans  |
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

**Logique de securite a 7 portes :**

| Porte | Condition | Action |
|-------|-----------|--------|
| 1 | < 3 messages utilisateur | Retourner Vert (contexte insuffisant) + verification mots-cles |
| 2 | Mot-cle de crise detecte | Forcer Rouge (confiance 0.90) |
| 3 | Prediction ML | Executer GradientBoosting sur vecteur 67D |
| 4 | Confiance ML < 0.45 | Defaut a Jaune (principe de precaution) |
| 5 | < 6 messages utilisateur | Plafonner ML a Orange max |
| 6 | ML < minimum securite | Appliquer plancher de securite |
| 7 | Delai de reponse >= 300s ET Jaune+ | Hausse +1 niveau (plafond Rouge). 120s+ ET Orange+ aussi. Vert jamais hausse. |

Mots-cles de crise etendus (arme, pistolet, couteau, gun, knife, shoot...) declenchent **Rouge immediatement** a tout moment. En cas d'override de securite, les barres de probabilite sont remplacees par un badge "mot de crise detecte".

Les **noms lisibles des features** sont disponibles en francais et en anglais (30+ entrees), selectionnables via le parametre `lang`.

**Sortie d'importance des features** : `get_alert_level()` retourne `feature_scores` — top 5 features par score composite (`importance_modele × |valeur_normalisee|`), chacune avec nom affichable, nom brut et score. Disponible pour les predictions ML et les overrides de securite, affiche en barres horizontales dans le dashboard.

#### 3. Response Modulator -- `cedd/response_modulator.py`

Quatre niveaux de prompts systeme distincts, disponibles en **francais et en anglais**. Le niveau Rouge utilise un **transfert accompagne en 5 etapes** + **intervenant simule « Alex »** :
1. Validation empathique (pas de ressources encore)
2. Transition accompagnee (demande de permission)
3. Presentation des ressources (JJE, 9-8-8, 911)
4. Encouragement a se connecter
5. Presence continue

Au niveau Rouge, CEDD propose aussi de connecter l'utilisateur avec **Alex**, un·e intervenant·e simule·e de JJE utilisant les techniques d'ecoute active ASIST. Si l'utilisateur accepte, le chat bascule vers un persona d'intervenant avec un style visuel distinct (bulles bleues, avatar 🧑‍⚕️, banniere d'intervenant). Le mode intervenant contourne le classificateur CEDD et utilise `HUMAN_COUNSELOR_PROMPT` via la meme chaine de fallback LLM. Seul le bouton Reinitialiser quitte le mode intervenant.

Hierarchie LLM (timeout de 15s par modele) : `cohere -> groq (Llama 3.3 70B) -> gemini-flash (Gemini 2.5 Flash) -> claude-haiku -> sans llm`

Chaque client LLM a un timeout de 25 secondes. Si un modele est lent ou bloque, la chaine passe automatiquement au fournisseur suivant sans geler l'interface. L'historique complet de la conversation est transmis a chaque modele, donc le changement en cours de conversation est transparent.

#### 4. Session Tracker -- `cedd/session_tracker.py`

Surveillance longitudinale inter-sessions via SQLite. Calcule `risk_score`, `trend`, `consecutive_high_sessions` et `recommendation` sur les 7 dernieres sessions.

**Detection d'abandon/retrait :** suit le `last_activity` par utilisateur. Si un utilisateur revient apres >24h sans avoir ferme sa session, une banniere de bienvenue et un badge d'abandon s'affichent.

#### 5. Interface Streamlit -- `app.py`

Interface bilingue en deux colonnes avec UI professionnelle : police **Inter** (Google Fonts), **proprietes CSS personnalisees** (tokens de design pour espacement, rayons, echelle typographique), mise en page **flexbox** pour le chat, et **micro-interactions** (elevation au survol + transitions d'ombre sur les bulles et les pills).

**Selecteur de profil** dans l'en-tete : 5 profils demo (Shuchita, Priyanka, Amanda, Dominic, Guest) avec des historiques longitudinaux distincts. **Bouton de langue** pour basculer entre Francais et English.

Composants du chat : **carte d'accueil** (classes CSS `.welcome-card`, emoji cerveau, titre, description, CTA quand le chat est vide), **bulles de chat** (mise en page flexbox avec `align-self`, ombre subtile, animation d'elevation au survol, barre de defilement personnalisee 6px, `line-height: 1.5`), **horodatages** (HH:MM sous chaque bulle), **badge LLM** (source du modele sur chaque reponse assistant), **badge niveau d'alerte** (point colore sur chaque reponse assistant), **demo autopilote** (bouton Play Demo joue le scenario Felix/Alex en 9 messages), **panneau A propos** (explication de CEDD repliable), **export JSON** (telecharge la conversation + historique d'alertes), **toast de transition** (notification animee CSS quand le niveau augmente ; animation de pulsation rouge au niveau Rouge), **mode comparaison** (LLM brut vs LLM guide par CEDD cote a cote, toggle via bouton 🔀), **radar des features** (graphique araignee Plotly des 10 features par message, dernier message vs Msg 1 en overlay), **transfert vers intervenant** (au niveau Rouge, propose de connecter avec « Alex », intervenant·e simule·e JJE utilisant ASIST ; bulles bleues avec ombre renforcee, avatar 🧑‍⚕️, banniere d'intervenant `.counselor-banner` ; contourne le classificateur CEDD en mode intervenant). Composants du dashboard : jauge circulaire, probabilites par classe (pistes `.proba-bar-track` adaptees au theme), signaux actifs (pills avec effet d'elevation au survol), **graphique d'importance des features** (barres horizontales Plotly, top 5 par score composite, 6 categories de couleurs, visible a partir du Jaune y compris lors des overrides de securite), historique du niveau, **flux emotionnel** (graphique en aires empilees des probabilites par classe au fil des messages, avec synthese 100% pour les overrides de securite), historique longitudinal, selecteur LLM, prompt systeme, cartes de statut (classe CSS `.status-card` reutilisable), statistiques de session (valeurs en gras 1.6rem, labels en majuscules).

---

### Support bilingue

| Couche                  | Anglais                              | Francais                              |
|-------------------------|--------------------------------------|---------------------------------------|
| Interface web           | Complete (bouton dans l'en-tete)     | Complete (langue par defaut)          |
| Prompts systeme LLM     | 4 niveaux                            | 4 niveaux                             |
| Analyse lexicale        | Mots EN dans les lexiques            | Mots FR dans les lexiques             |
| Noms des features       | Via parametre `lang="en"`            | Via parametre `lang="fr"`             |
| Donnees d'entrainement  | 300 conversations EN                 | 300 conversations FR                  |
| Tests adversariaux      | 14 EN + mixtes                       | 16 FR + mixtes                        |

---

### Donnees synthetiques

**`data/synthetic_conversations.json`** -- 600 conversations bilingues (480 standard + 120 adversariaux).

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Archetypes standard (20 par classe)
python generate_synthetic_data.py --lang fr --count 20
python generate_synthetic_data.py --lang en --count 20

# Archetypes adversariaux (10 par archetype)
python generate_synthetic_data.py --adversarial --lang fr --count 10
python generate_synthetic_data.py --adversarial --lang en --count 10
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

# 4. Configurer les cles API LLM (au moins une requise pour le chat)
export COHERE_API_KEY="..."            # Primaire : Cohere (par defaut)
export GROQ_API_KEY="gsk_..."          # Secondaire : Llama 3.3 70B via Groq (le plus rapide)
export GEMINI_API_KEY="AI..."          # Tertiaire : Gemini 2.5 Flash
export ANTHROPIC_API_KEY="sk-ant-..."  # Quaternaire : Claude Haiku + generation de donnees

# 5. Entrainer le modele
python train.py

# 6. Lancer l'interface
streamlit run app.py
```
---

### Utilisation

```bash
# Entrainer le classifieur
python train.py

# Lancer l'interface web bilingue
streamlit run app.py

# Simuler l'historique pour la demo (4 profils x 7 sessions)
python simulate_history.py --lang fr
python simulate_history.py --lang en

# Generer des donnees synthetiques supplementaires
python generate_synthetic_data.py --lang fr --count 20
python generate_synthetic_data.py --lang en --count 20

# Generer des donnees adversariales
python generate_synthetic_data.py --adversarial --lang fr --count 10
python generate_synthetic_data.py --adversarial --lang en --count 10

# Lancer les tests adversariaux
python tests/adversarial_suite.py --verbose

# Lancer les tests unitaires (feature extractor, classifier, response modulator, session tracker)
pytest tests/test_unit.py -v

# Lancer les tests d'integration (scenarios demo, cross-langue, UI bilingue, cas limites)
pytest tests/test_integration.py -v

# Lancer tous les tests (unitaires + integration)
pytest tests/ -v
```

Ouvre `http://localhost:8501`. Utiliser le **selecteur de profil** dans l'en-tete pour choisir un utilisateur demo (Shuchita, Priyanka, Amanda, Dominic, Guest). Utiliser le bouton de langue pour basculer. Cliquer sur **Reinitialiser / Reset** pour demarrer une nouvelle session.

---

### Metriques

Resultats sur le dataset de 600 conversations bilingues (480 standard + 120 adversariaux) :

| Metrique                  | Valeur                         |
|---------------------------|--------------------------------|
| CV accuracy (k=4)         | **90.0% +/- 1.6%**            |
| Train accuracy            | 100% (overfitting attendu)     |
| Nombre de features        | **67** (10x6 + 4 emb + 3 coh) |
| Conversations             | **600** (480 standard + 120 adversariaux) |
| Ratio echantillons:features | **9.0:1** (ameliore de 7.2)  |
| Top feature               | `word_count_max` (0.192)       |
| 2e feature                | `word_count_slope` (0.179)     |
| 3e feature                | `word_count_last` (0.138)      |
| 4e feature                | `finality_score_mean` (0.066)  |
| Tests adversariaux        | **36/36 reussis**              |
| Tests unitaires (pytest)  | **94/94 reussis**              |
| Tests d'integration (pytest) | **39/39 reussis**           |
| Crises manquees           | **0**                          |

#### Historique des metriques

| Date | Evenement | CV Accuracy | Adversarial |
|------|-----------|-------------|-------------|
| Mars 10 | Baseline (24 convos FR, 42 features) | 66.7% +/- 26.4% | 7/10 |
| Mars 12 | Expansion donnees (320 convos bilingues) | ~91.2% +/- 1.5% | 9/10 |
| Mars 12 | Expansion mots-cles de crise | ~91.2% +/- 1.5% | 10/10 |
| Mars 12 | +Negation + Embeddings (52 features) | ~92.2% +/- 1.8% | 9/10 |
| Mars 12 | +Identite + Somatisation + Coherence (67 features) | 92.5% +/- 1.5% | 13/13 |
| Mars 12 | Expansion donnees a 480 convos (60/classe) | 91.7% +/- 4.4% | 13/13 |
| Mars 13 | Augmentation adversariale a 600 convos (6 nouveaux archetypes) | 90.5% +/- 1.5% | 30/30 |
| Mars 14 | Correctif frontieres de mots + 6 nouveaux tests (regex `\b`, "personne" contextuel, formes feminines) | 90.0% +/- 1.6% | 36/36 |
| Mars 15 | +Porte 7 delai de reponse (67 features, 7 portes) | **90.0% +/- 1.6%** | **36/36** |

---

### Structure du projet

```
cedd-hackathon/
+-- app.py                          # Interface Streamlit bilingue
+-- train.py                        # Entrainement : chargement -> CV -> fit -> sauvegarde
+-- generate_synthetic_data.py      # Generation via Claude API (FR + EN)
+-- simulate_history.py             # Simulation d'historique par profil utilisateur (FR + EN)
+-- requirements.txt                # Dependances Python
|
+-- cedd/                           # Package Python principal
|   +-- __init__.py
|   +-- feature_extractor.py        # 10 features/msg + embeddings + coherence -> 67D
|   +-- classifier.py               # CEDDClassifier (GradientBoosting + 7 portes securite)
|   +-- response_modulator.py       # Prompts adaptatifs FR + EN + chaine LLM
|   +-- session_tracker.py          # Suivi longitudinal inter-sessions SQLite
|
+-- tests/                          # Suites de tests (Track 1)
|   +-- adversarial_suite.py        # Runner CLI (--verbose, --category, --export)
|   +-- test_unit.py                # 94 tests unitaires pytest (4 modules : features, classifier, modulator, tracker)
|   +-- test_integration.py         # 39 tests d'integration pytest (demo, cross-langue, UI bilingue, cas limites)
|   +-- test_cases_adversarial.json # 36 cas de test adversariaux, 20 categories (FR + EN)
|
+-- data/
|   +-- synthetic_conversations.json  # 600 conversations etiquetees (480 standard + 120 adversariaux)
|   +-- cedd_sessions.db              # Base SQLite (creee automatiquement)
|
+-- models/
|   +-- cedd_model.joblib            # Modele entraine (cree par train.py)
|
+-- report.md                        # Rapport formel de hackathon (livrable requis)
```

---

### Tests adversariaux

Le repertoire `tests/` fournit une suite de tests systematiques pour valider la robustesse de CEDD face a des cas reels difficiles.

#### Categories de tests (36 tests, 20 categories)

| Categorie | Description | Nb |
|---|---|---|
| `false_positive_physical` | Plaintes physiques qui NE doivent PAS declencher d'alerte | 2 |
| `sarcasm` | Langage sarcastique masquant une detresse reelle | 1 |
| `negation` | Negation d'etats positifs (`"je ne me sens pas bien"`) | 1 |
| `code_switching` | Alternance francais/anglais (franglais quebecois) | 1 |
| `quebecois_slang` | Expressions quebecoises (`"chu pu capable"`, joual positif) | 3 |
| `gradual_drift_no_keywords` | Deterioration emotionnelle lente sans mots-cles (EN + FR) | 2 |
| `direct_crisis` | Langage de crise explicite -- **doit toujours etre Rouge** (EN + FR) | 2 |
| `hidden_intent` | Ideation suicidaire indirecte presentee comme hypothetique | 1 |
| `manipulation_downplay` | Detresse suivie de minimisation -- ne doit PAS redescendre a Vert | 1 |
| `somatization` | Douleur physique + declin emotionnel (detresse somatisee) | 1 |
| `identity_conflict` | Crise identitaire 2SLGBTQ+ et rejet familial (EN + FR) | 2 |
| `sudden_escalation` | Conversation normale puis escalade soudaine | 3 |
| `active_bypass` | Langage de crise puis retractation ("c'etait une blague") | 2 |
| `rapid_recovery_manipulation` | Crise profonde puis "ca va mieux" -- le plancher doit persister | 2 |
| `cultural_false_positive` | "Mort de rire", "killed it", "personne" en contexte neutre | 3 |
| `neurodivergent_pattern` | Communication litterale/plate, explosions TDAH, sauts de sujet | 3 |
| `emoji_only` | Messages tres courts avec ellipses et emoji | 1 |
| `repeated_word` | Mots repetes et messages de frustration brefs | 1 |
| `short_recovery` | Crise breve puis recuperation rapide (conversation courte) | 1 |
| `long_message` | Long message de ventilation sans mots de crise | 1 |
| `neutral_personne_fr` | Utilisation neutre de "personne" (= quelqu'un) en francais | 1 |
| `emoji_crisis` | Mots de crise melanges avec emoji -- **doit etre Rouge** | 1 |

#### Codes de sortie

| Code | Signification |
|---|---|
| `0` | Tous les tests reussis |
| `1` | Certains tests echoues (non critique) |
| `2` | **Crise manquee** -- crise predite comme Vert/Jaune (regression de securite) |

> **Actuel (v9) :** 36/36 reussis, 0 crise manquee

#### Tests unitaires (pytest)

94 tests automatises couvrant les 4 modules principaux :

| Module | Tests | Couverture |
|--------|------:|------------|
| Feature Extractor | 34 | Les 10 features (FR + EN), cas limites, forme trajectoire, direction pente |
| Classifier | 16 | Les 7 portes de securite dont Porte 7 delai reponse, mots-cles de crise (FR + EN), plancher de securite, flag delay_bumped |
| Response Modulator | 23 | Selection de prompts, ressources de crise Orange/Rouge, etapes transfert 1-5, intervenant Alex, fallback statique |
| Session Tracker | 21 | Cycle de vie session, detection de retrait, risque longitudinal (tendances, sessions consecutives) |

```bash
pytest tests/test_unit.py -v                    # Les 94 tests
pytest tests/test_unit.py -v -k "feature"       # Feature extractor seulement
pytest tests/test_unit.py -v -k "Gate"           # Portes du classifier seulement
```

> **Lacune documentee dans les tests :** les mots de crise conjugues ("killing myself") ne correspondent pas a l'entree du lexique ("kill myself") -- le plancher de securite Gate 2 ne se declenche pas pour les formes conjuguees.

#### Tests d'integration (pytest)

39 tests validant la preparation a la presentation, repartis en 7 categories :

| Categorie | Tests | Ce que ca detecte |
|-----------|------:|-------------------|
| Scenarios demo | 7 | Demo qui plante, mauvais niveaux d'alerte, crash en autopilote |
| Coherence cross-langue | 6 | FR et EN produisant des resultats tres differents |
| Completude bilingue | 5 | Traductions manquantes, placeholders de format brises |
| Integration bout en bout | 6 | Pipeline produisant de mauvais resultats, problemes de modele |
| Cas limites | 7 | Crash sur emoji, messages longs, espaces, langues melangees |
| Scores de features | 5 | Graphique d'explicabilite vide ou brise |
| Integration session tracker | 3 | Echec de journalisation avec resultats reels du classifieur |

```bash
pytest tests/test_integration.py -v             # Les 39 tests d'integration
pytest tests/test_integration.py -v -k "Demo"   # Validation scenarios demo
pytest tests/ -v                                 # Les 133 tests (unitaires + integration)
```

---

### Limites connues et pistes d'amelioration

| Limite                                                              | Piste d'amelioration                                             |
|---------------------------------------------------------------------|------------------------------------------------------------------|
| ML peu fiable pour conversations courtes (< 6 messages)            | ML plafonne a Orange; mots-cles de crise declenchent Rouge       |
| Aucune validation clinique des seuils                              | Collaboration avec professionnels en sante mentale               |
| Detection identitaire basee sur des phrases, pas le contexte       | Fine-tuner les embeddings sur un corpus detresse identitaire     |
| Detection d'abandon basee sur seuil (>24h), pas intra-session     | Suivre le delai intra-session et le desengagement progressif     |
| LLM non fine-tune pour le contexte de crise                       | Fine-tuning sur conversations d'intervenants certifies           |
| Correspondance par frontieres de mots amelioree mais pas parfaite (idiomes comme "mort de rire") | Exclusion contextuelle de phrases ou detection d'idiomes |

---

### Ressources d'urgence

> Ces ressources sont integrees dans les prompts de niveau Orange et Rouge.

- **Jeunesse J'ecoute / Kids Help Phone** : 1-800-668-6868 (24h/24, gratuit, confidentiel) -- texto : 686868
- **Ligne de crise suicide** : 9-8-8 (988.ca)
- **Multi-Ecoute** : 514-378-3430 (multiecoute.org)
- **Tracom** : 514-483-3033 (tracom.ca)
- **Urgences / Emergency** : 911
- **Medecin de famille / service de consultation scolaire**
