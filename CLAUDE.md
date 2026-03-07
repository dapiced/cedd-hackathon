# CLAUDE.md — Project Context for AI Assistants

> **CEDD — Conversational Emotional Drift Detection**
> Hackathon Mila × Bell × Kids Help Phone · March 16–23, 2026
> Team: **404HarmNotFound**

---

## What This Project Is

## Hackathon Mission

> **Your mission:** Identify vulnerabilities and engineer robust defences
> —build the armour that keeps our youth safe—by diving deep into three
> critical areas:
>
> - **Track 1:** Adversarial Stress-Testing
> - **Track 2:** Logic Hardening
> - **Track 3:** Synthetic Data Augmentation

Judges evaluate on: **Safety**, **User Experience**, and **Data Quality**.

CEDD is a **real-time safety layer** that sits beside a youth mental health chatbot (ages 16–22). It monitors the **trajectory** of emotional deterioration across an entire conversation — not just individual messages — and raises alerts when it detects drift toward distress.

**Core insight:** A normal chatbot sees each message in isolation. CEDD watches *how messages evolve over time* (getting shorter, more negative, losing questions and hope words) and adapts the chatbot's response accordingly.

**Hackathon context:** Mila + Bell + Jeunesse, J'écoute. Three evaluation axes: adversarial stress-testing, logic hardening, synthetic data augmentation. Judges evaluate safety, UX, and data quality.

---

## Team — 404HarmNotFound

| Name | Location | Role Focus | Key Strengths |
|------|----------|------------|---------------|
| Shuchita Singh | East Gwillimbury | ML / NLP Lead | AI Researcher, LLM fine-tuning, RAG, NLP, Responsible AI, DBA Gen AI (Synopsys) |
| Amanda Wu | Toronto | UX / Presentation | Principal Product Designer (TD), 10 yrs FinTech UX, HCI Master (UCL) |
| Priyanka Naga | Ottawa | Clinical / Strategy | Business Leader (Ottawa Hospital), Innovation Lead (CHEO), MSc Data Science, PMP |
| Dominic D'Apice | Blainville | Infra / Code / ML | Dev II AI Infra Azure, 25+ yrs Linux/DevOps, Data Science cert (TÉLUQ), Kaggle |

---

## Repository Structure

```
cedd-hackathon/
│
├── app.py                          # Bilingual Streamlit web interface (FR/EN)
├── train.py                        # Training script: load → cross-validate → fit → save
├── generate_synthetic_data.py      # Generates training data via Claude Haiku API (FR+EN)
├── simulate_history.py             # Populates demo session history for the UI
├── requirements.txt                # Python dependencies
│
├── cedd/                           # Main Python package (the "brain")
│   ├── __init__.py
│   ├── feature_extractor.py        # Text → 7 features/message → 42 trajectory features
│   ├── classifier.py               # CEDDClassifier: GradientBoosting + 6-gate safety logic
│   ├── response_modulator.py       # Adaptive system prompts per alert level + LLM fallback chain
│   └── session_tracker.py          # Cross-session longitudinal risk tracking (SQLite)
│
├── data/
│   ├── synthetic_conversations.json  # 120 labeled training conversations (30 per class)
│   └── cedd_sessions.db             # SQLite database (auto-created at runtime)
│
├── models/
│   └── cedd_model.joblib            # Trained model (created by train.py)
│
├── screenshots/                     # App screenshots for README
├── README.md                        # Full bilingual documentation
├── PITCH.md                         # Pitch deck content
└── explanation.md                   # Detailed technical walkthrough (11 steps)
```

---

## Two-Phase Architecture

```
PHASE 1: TRAINING (offline, run once)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
generate_synthetic_data.py  →  data/synthetic_conversations.json (120 convos)
                                         ↓
                              train.py  (cross-validate → fit → save)
                                         ↓
                              models/cedd_model.joblib

PHASE 2: LIVE APP (runs per user chat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.py loads saved model
         ↓
user sends message
         ↓
feature_extractor.py  →  7 features per message → 42 trajectory features
         ↓
classifier.py  →  6-gate decision logic → alert level (0-3)
         ↓
response_modulator.py  →  adaptive system prompt → LLM response
         ↓
session_tracker.py  →  saves to SQLite for longitudinal tracking
```

---

## Technical Details

### Feature Extraction (`cedd/feature_extractor.py`)

**7 features per user message:**

| # | Feature | What it measures |
|---|---------|-----------------|
| 1 | `word_count` | Message length (shorter = concerning) |
| 2 | `punctuation_ratio` | Ratio of punctuation to words |
| 3 | `question_presence` | Does the message contain a question? (0 or 1) |
| 4 | `negative_score` | Count of negative words from bilingual lexicon |
| 5 | `finality_score` | Count of finality words ("plus rien", "end it all") |
| 6 | `hope_score` | Count of hope/future words |
| 7 | `length_delta` | Change in length vs previous message |

**42 trajectory features** = 7 features × 6 statistics (mean, std, slope, last, max, min) computed over all user messages in the conversation. This is the ML model input.

Bilingual lexicons: `FINALITY_WORDS`, `HOPE_WORDS`, `NEGATIVE_WORDS` contain both FR and EN terms.

### Classifier (`cedd/classifier.py`)

**ML Pipeline:**
- Step 1: `StandardScaler` — normalizes 42 features to mean=0, std=1
- Step 2: `GradientBoostingClassifier` — 200 trees, max_depth=3, learning_rate=0.1

**6-Gate Safety Logic** (`get_alert_level()` method):

```
Gate 1: < 3 user messages? → return Green (not enough data)
Gate 2: Safety keyword floor — crisis words → force Red/Orange regardless of ML
Gate 3: ML prediction — run model on 42D vector
Gate 4: Low confidence (< 45%)? → default to Yellow (cautious)
Gate 5: Short conversation cap (< 6 msgs)? → cap at Orange max
Gate 6: Safety floor enforcement — ML can never go below keyword level
```

**Design philosophy:** Asymmetric errors — false positives (over-alerting) are always preferable to false negatives (missing a crisis).

### Alert Levels

| Level | Color | Description | LLM Behavior |
|-------|-------|-------------|-------------|
| 0 | 🟢 Green | Normal conversation | Supportive standard |
| 1 | 🟡 Yellow | Concerning signs (fatigue, loneliness) | Enhanced emotional validation |
| 2 | 🟠 Orange | Significant distress | Active support + crisis resources |
| 3 | 🔴 Red | Potential crisis | Urgent referral (Kids Help Phone: 1-800-668-6868) |

### Response Modulator (`cedd/response_modulator.py`)

- Swaps LLM system prompt based on alert level (4 distinct prompts, FR and EN)
- LLM fallback chain: Claude API → Ollama (local) → static emergency text
- Orange/Red prompts include Kids Help Phone resources

### Session Tracker (`cedd/session_tracker.py`)

- SQLite-based longitudinal tracking across multiple conversations
- Records: session_id, user_id, timestamp, alert_level, confidence, top_features
- Enables cross-session risk pattern detection

### Training (`train.py`)

- Loads 120 synthetic conversations from `data/synthetic_conversations.json`
- Extracts 42 trajectory features per conversation → X (120 × 42), y (120 labels)
- Cross-validation: `StratifiedKFold(n_splits=4)` → **~75.8% accuracy ± 6%**
- Train accuracy: ~100% (expected overfitting on 120 samples with 200 trees)
- Top features: `word_count_slope`, `negative_score_mean`, `question_presence_slope`
- Saves trained model to `models/cedd_model.joblib`

### Data Generation (`generate_synthetic_data.py`)

- Uses Claude Haiku API to generate realistic youth conversations
- 30 conversations per class × 4 classes = 120 total
- Each conversation: 12 user messages + 12 assistant messages
- Authentic Canadian French; English generation also supported
- **All data is synthetic — no real PII allowed** (hackathon rule)

---

## Current Metrics (Baseline)

| Metric | Value |
|--------|-------|
| Cross-validated accuracy (k=4) | ~75.8% ± 6% |
| Train accuracy | ~100% (expected overfitting) |
| Feature count | 42 (7 × 6 stats) |
| Training conversations | 120 (30 per class) |
| Top feature | `word_count_slope` |
| Languages | French (primary), English (lexicons + UI) |

---

## Known Limitations

- **Small dataset** (120 convos) → high overfitting, model may not generalize well
- **Lexical features only** — counts words, doesn't understand meaning (misses synonyms, sarcasm, periphrases like "je pèse sur tout le monde")
- **No negation handling** — "je ne me sens pas bien" may score low on negativity
- **ML unreliable for short conversations** (< 6 messages) → capped at Orange
- **French-trained model** — English conversations not yet in training set
- **No clinical validation** — thresholds are not validated by mental health professionals
- **Single demo user** in the Streamlit UI (`demo_user`)

---

## Planned Improvements (Hackathon Week)

These are potential directions discussed pre-hackathon. The team decides together what to prioritize.

### Axis 1 — Adversarial Stress-Testing
- Systematic adversarial test cases: sarcasm, code-switching, Québécois slang, negation patterns
- Test false positives ("j'ai mal au dos" vs "j'ai mal au coeur")
- Test bypass attempts and edge cases

### Axis 2 — Logic Hardening
- **Negation handling** in `feature_extractor.py` (e.g., "ne...pas + positive word" = negative signal)
- **Sentence embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`) to replace word counting with semantic similarity — biggest impact improvement
- **Conversational coherence features** (response length ratio, topic shifts, response timing)
- **LSTM** for true sequential modeling (messages as a time series instead of aggregated stats)

### Axis 3 — Synthetic Data Augmentation
- **Claude as quality annotator** — score each generated conversation for ambiguity/realism, filter low-quality examples
- Generate targeted adversarial training examples
- Add English Canadian conversations
- Cultural and linguistic diversity (regional idioms, bilingual code-switching)

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your_key
python generate_synthetic_data.py

# Train the model
python train.py

# Populate demo history (optional)
python simulate_history.py

# Run the app
streamlit run app.py
```

**Environment variables:**
- `ANTHROPIC_API_KEY` — required for Claude API calls (data generation + live chat)
- Ollama must be running locally for the Ollama fallback path

---

## Key Dependencies

- `streamlit` — web interface
- `scikit-learn` — ML pipeline (GradientBoosting, StandardScaler, cross-validation)
- `numpy` — numerical operations
- `anthropic` — Claude API client
- `joblib` — model serialization
- `sqlite3` — session tracking (Python stdlib)

---

## Coding Conventions

- **Bilingual throughout**: all user-facing strings have FR and EN versions
- **Comments**: bilingual (FR + EN on the same line, separated by `/`)
- **Print output**: bilingual during training
- **Variable names**: English
- **Lexicons**: bilingual dictionaries in `feature_extractor.py`
- **Error handling**: LLM fallback chain (Claude → Ollama → static text)

---

## Important Context for AI Assistants

1. **This is a safety-critical application for youth mental health.** Never weaken safety gates, lower alert thresholds, or remove crisis resources from prompts.
2. **All data must be synthetic.** No real PII ever. This is a strict hackathon rule.
3. **Asymmetric error philosophy:** Over-alerting is always preferable to missing a crisis. False positives > false negatives.
4. **The 6-gate logic in `classifier.py` is deliberate.** Safety keyword rules override ML predictions by design. Don't simplify this.
5. **Bilingual is required.** Any new feature, prompt, or UI element must work in both French and English.
6. **The team has 4 members with different expertise.** Code changes may need discussion. The repo is subject to change during the hackathon week.
7. **Dominic (repo owner) is learning ML/DS.** When explaining code changes, explain the *why* — the ML concepts, algorithm choices, and hyperparameter reasoning.
8. **`digest.txt` in the project files** is a complete 11-step pedagogical guide to the entire codebase. Consult it for detailed explanations of every component.

---

## Emergency Resources (Integrated in Orange/Red Prompts)

- **Kids Help Phone**: 1-800-668-6868 (24/7, free, confidential) — text: 686868
- **Emergency services**: 911
- **Family doctor / school counselling service**

---

*Last updated: March 2026 — Pre-hackathon baseline*
