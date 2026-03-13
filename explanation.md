# CEDD — Complete Technical Explanation

> This document was created during a step-by-step teaching session on March 13, 2026.
> It covers every algorithm, feature, and design decision in the CEDD repository.

---

## Table of Contents

1. [The Big Picture — What Problem Does CEDD Solve?](#step-1-the-big-picture)
2. [The Two-Phase Architecture — Training vs Live](#step-2-two-phase-architecture)
3. [Feature Extraction — The 10 Per-Message Features](#step-3-feature-extraction)
4. [Trajectory Features — How 10 Features Become 60](#step-4-trajectory-features)
5. [Embedding Features — The 4 Semantic Features (60 → 64)](#step-5-embedding-features)
6. [Coherence Features — The 3 Behavioral Features (64 → 67)](#step-6-coherence-features)
7. [The ML Model — GradientBoosting + StandardScaler](#step-7-the-ml-model)
8. [The 6-Gate Safety Logic — Rules That Override ML](#step-8-the-6-gate-safety-logic)
9. [The Training Pipeline — How train.py Works](#step-9-training-pipeline)
10. [Response Modulation — How the Chatbot Adapts](#step-10-response-modulation)
11. [The Warm Handoff — 5-Step Crisis Transition](#step-11-warm-handoff)
12. [Session Tracking — Cross-Session Memory](#step-12-session-tracking)
13. [The Streamlit UI — How app.py Ties Everything Together](#step-13-streamlit-ui)
14. [Synthetic Data Generation — How Training Data Is Created](#step-14-synthetic-data-generation)
15. [Adversarial Testing — Stress-Testing the Safety System](#step-15-adversarial-testing)

---

## Step 1: The Big Picture

CEDD (Conversational Emotional Drift Detection) is a **real-time safety layer** that sits beside a youth mental health chatbot (ages 16-22). It monitors the **trajectory** of emotional deterioration across an entire conversation — not just individual messages — and raises alerts when it detects drift toward distress.

**Core insight:** A normal chatbot sees each message in isolation. CEDD watches *how messages evolve over time* (getting shorter, more negative, losing questions and hope words) and adapts the chatbot's response accordingly.

**Analogy:**
- **The chatbot** = the doctor talking to the patient
- **CEDD** = the nurse watching the heart monitor, tapping the doctor on the shoulder when vitals drop

**4 alert levels:**

| Level | Color | Meaning |
|-------|-------|---------|
| 0 | Green | Normal conversation |
| 1 | Yellow | Concerning signs (fatigue, loneliness) |
| 2 | Orange | Significant distress |
| 3 | Red | Potential crisis — urgent referral needed |

---

## Step 2: Two-Phase Architecture

### Phase 1: Training (offline, run once)

```
generate_synthetic_data.py  →  Creates 600 conversations via Claude API
                                        ↓
data/synthetic_conversations.json       (the training dataset: 480 standard + 120 adversarial)
                                        ↓
train.py                    →  Reads conversations, extracts features, trains model
                                        ↓
models/cedd_model.joblib                (the saved trained model)
```

### Phase 2: Live App (runs every time a user chats)

```
app.py loads the saved model from disk
        ↓
User sends a message
        ↓
feature_extractor.py  →  Analyzes ALL user messages → produces 67 numbers
        ↓
classifier.py         →  Takes 67 numbers → 6 safety gates + ML → alert level
        ↓
response_modulator.py →  Picks system prompt for that alert level → calls LLM
        ↓
session_tracker.py    →  Saves everything to SQLite for cross-session memory
```

**Key point:** Phase 1 happens once. The model is saved to disk as a `.joblib` file. Phase 2 uses that saved model thousands of times without retraining.

---

## Step 3: Feature Extraction

File: `cedd/feature_extractor.py`

For each user message, CEDD extracts **10 numbers**:

| # | Feature | What It Measures | Why It Matters |
|---|---------|-----------------|----------------|
| 0 | `word_count` | Number of words | People in crisis write shorter messages |
| 1 | `punctuation_ratio` | Punctuation / total characters | Depressed writing loses punctuation |
| 2 | `question_presence` | Is there a `?` (0 or 1) | Engaged people ask questions; shutting down = no questions |
| 3 | `negative_score` | Ratio of negative words (bilingual lexicon) | Direct negativity measurement |
| 4 | `finality_score` | Ratio of finality/crisis words ("mourir", "end it") | Most dangerous lexicon — suicidal ideation |
| 5 | `hope_score` | Ratio of hope words ("tomorrow", "better", "espoir") | Hope disappearing over time is a signal |
| 6 | `length_delta` | Relative length change vs previous message | Consistently negative = messages shrinking = withdrawal |
| 7 | `negation_score` | Regex matches for negated positive states ("can't cope", "ne...pas bien") | Catches "I'm not okay" which simple word counting misses |
| 8 | `identity_conflict_score` | Phrases about identity rejection ("my family won't accept me") | 46% of KHP youth are 2SLGBTQ+ — identity distress is major |
| 9 | `somatization_score` | Physical + emotional words co-occurring | Cultural pattern: "my stomach hurts" + "I feel empty" = somatization |

**Bilingual lexicons:** `FINALITY_WORDS`, `HOPE_WORDS`, `NEGATIVE_WORDS`, `IDENTITY_CONFLICT_WORDS`, `SOMATIZATION_EMOTIONAL_WORDS`, `PHYSICAL_CONTEXT_WORDS` all contain both French and English terms.

If a conversation has 8 user messages, the output is a **(8, 10) matrix** — 8 rows, 10 columns.

---

## Step 4: Trajectory Features

The ML model needs **one fixed-size input** per conversation, regardless of conversation length. We solve this by computing **6 summary statistics** for each of the 10 features:

| Statistic | What It Captures |
|-----------|-----------------|
| **mean** | Average value across all messages |
| **std** | How much variation (erratic vs consistent) |
| **slope** | Linear trend direction (increasing or decreasing over time) |
| **last** | Most recent value (where are they NOW?) |
| **max** | Peak value (where did they START or how bad did it get?) |
| **min** | Lowest value (floor of this feature) |

**The slope** is computed using `np.polyfit(x, y, 1)` — a linear regression that fits a straight line through the data points and returns the slope.

**10 features x 6 statistics = 60 trajectory features**

Example: `word_count` across 8 messages: `[45, 38, 30, 22, 15, 10, 6, 3]`
- mean = 21.1, std = 14.8, **slope = -6.0** (shrinking!), last = 3, max = 45, min = 3

Naming convention: `{feature}_{statistic}` — e.g., `word_count_slope`, `finality_score_mean`.

**Top features by model importance:**
1. `word_count_max` (0.189) — how long were messages at peak
2. `word_count_slope` (0.160) — shrinking messages is a top crisis signal
3. `word_count_last` (0.137) — most recent message length
4. `finality_score_mean` (0.097) — average finality language

---

## Step 5: Embedding Features

File: `cedd/feature_extractor.py` (lines 359-491)

The 60 trajectory features are **lexical** — they count words from predefined lists. Limitation: "I want to cease existing" won't match the finality lexicon because "cease existing" isn't in the list.

**Sentence embeddings** solve this using `paraphrase-multilingual-MiniLM-L12-v2`. This model converts a sentence into a 384-dimensional vector that captures its *meaning*. Sentences with similar meaning end up close together, even with completely different words. It's multilingual: "je veux mourir" and "I want to die" produce nearly identical vectors.

**4 embedding features:**

| # | Feature | Algorithm | What It Measures |
|---|---------|-----------|-----------------|
| 61 | `embedding_drift` | Mean cosine distance between consecutive messages | Are messages jumping between topics? |
| 62 | `crisis_similarity` | Cosine similarity of last message to a pre-computed crisis centroid (average of 10 crisis phrases in FR+EN) | Does the last message *sound like* crisis language? |
| 63 | `embedding_slope` | PCA reduces 384D → 1D, then linear slope | Is the conversation drifting in a consistent semantic direction? |
| 64 | `embedding_variance` | Mean pairwise cosine distance between all messages | Is the conversation coherent or scattered? |

**Crisis centroid:** Pre-computed as the average embedding of phrases like "je veux mourir", "I want to die", "I can't take it anymore", "nothing matters anymore". Then every last user message is compared to this centroid.

**Cosine similarity:** Measures how "aligned" two vectors are (1.0 = identical meaning, 0.0 = unrelated).

**PCA (Principal Component Analysis):** Finds the single axis that captures the most variation, collapsing 384 dimensions into 1. Then a slope is fit through the projected values.

---

## Step 6: Coherence Features

File: `cedd/feature_extractor.py` (lines 494-544)

These measure **behavioral patterns** — not *what* the person says, but *how* they interact with the chatbot. They detect withdrawal and disengagement.

| # | Feature | Algorithm | What It Measures |
|---|---------|-----------|-----------------|
| 65 | `short_response_ratio` | Fraction of user messages with < 5 words | Disengagement — "ok", "idk", "yeah" |
| 66 | `min_topic_coherence` | Min cosine similarity between consecutive user embeddings | Worst topic jump — sudden shift from small talk to crisis? |
| 67 | `question_response_ratio` | Fraction of assistant questions followed by substantive reply (>5 words or contains `?`) | Are they engaging with the chatbot or ignoring its questions? |

**Total: 60 trajectory + 4 embedding + 3 coherence = 67 features**

---

## Step 7: The ML Model

File: `cedd/classifier.py`

### The Pipeline

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3)),
])
```

### Step 1: StandardScaler

Normalizes all 67 features to mean=0, std=1. Without this, features with bigger numbers (word_count ~25) would dominate features with smaller numbers (punctuation_ratio ~0.03). After scaling, they're comparable.

### Step 2: GradientBoostingClassifier

Builds **200 decision trees sequentially**. Each tree fixes the mistakes of all previous trees combined.

- A single decision tree = a flowchart of yes/no questions ("Is word_count_slope < -2?")
- A single tree is simple but dumb — can't capture complex patterns
- Gradient Boosting chains 200 simple trees, each correcting the previous ones
- "Gradient" = uses calculus (gradient descent) to optimize how much each tree corrects

**Hyperparameters:**
- `n_estimators=200` — 200 trees
- `max_depth=3` — Each tree can only ask 3 levels of questions (keeps them simple)
- `random_state=42` — Seed for reproducibility

**Output:** 4 probabilities summing to 1.0 (one per class):
```
[0.05, 0.15, 0.60, 0.20]  →  5% Green, 15% Yellow, 60% Orange, 20% Red
```

---

## Step 8: The 6-Gate Safety Logic

File: `cedd/classifier.py`, method `get_alert_level()` (line 178)

**Philosophy:** ML is smart but not trustworthy enough for life-or-death decisions. The 6 gates ensure safety keywords always override ML predictions.

### Gate 1: Not Enough Data (< 3 user messages)
→ Skip ML entirely. Only scan for crisis/critical/distress keywords. Default to Green with "insufficient context" label.

### Gate 2: Safety Keyword Floor
→ Scan ALL user text for crisis words ("mourir", "kill myself"), critical words ("disappear", "burden"), and distress words ("crying", "alone"). Set a `minimum_level` floor that ML cannot go below.

### Gate 3: ML Prediction
→ Run the 67-feature vector through GradientBoosting. Get `ml_level` and `confidence`.

### Gate 4: Low Confidence (< 45%)
→ If the model isn't sure, default to Yellow. Prevents Green predictions when the model is uncertain.

### Gate 5: Short Conversation Cap (< 6 user messages)
→ Cap ML at Orange maximum. Red through ML requires at least 6 messages of context. (Red through Gate 2 keywords still works at any length.)

### Gate 6: Safety Floor Enforcement
→ `final_level = max(ml_level, minimum_level)`. ML can only raise the level above the keyword floor, never lower it.

**Design principle:** Asymmetric errors — false positives (over-alerting) are always preferable to false negatives (missing a crisis).

When a safety override happens (Gate 6 raised level above ML), probabilities are returned as empty `{}` and the UI shows "safety rule override" instead of probability bars.

**Feature importance output:** `get_alert_level()` also returns `feature_scores` — the top 5 features ranked by **composite score** = `model_importance × |scaled_value|`. Each entry contains: display name (bilingual), raw feature name, and numeric score. This is returned for both normal ML predictions and safety overrides, so the dashboard can always show what the model considers most relevant — even when keywords forced the level.

---

## Step 9: Training Pipeline

File: `train.py`

### Training Data
- 600 synthetic conversations in `data/synthetic_conversations.json`
- **480 standard:** 60 per class (Green/Yellow/Orange/Red) x 2 languages (FR/EN)
- **120 adversarial:** 6 specialized archetypes (physical_only, sarcasm_distress, adversarial_bypass, identity_distress, neurodivergent_flat, crisis_with_deflection) x 10 x 2 languages
- Each conversation: 12 user + 12 assistant messages

### Steps

1. **Load & extract:** Loop through 600 conversations → extract 67-feature vector for each → build X (600 x 67) and y (600 labels)

2. **Cross-validation:** `StratifiedKFold(n_splits=4)` — splits data into 4 groups, trains on 3, tests on 1, rotates 4 times. "Stratified" = each fold has the same class proportions. Result: **90.5% ± 1.5%** accuracy (more stable than the previous 91.7% ± 4.4% — the lower variance means more consistent performance across folds).

3. **Full training:** `clf.fit(X, y)` on all 600 conversations. Train accuracy ~100% (expected — 200 trees memorize 600 samples).

4. **Feature importances:** GradientBoosting tracks how useful each feature was. Features used in more tree splits, at higher levels, get higher importance.

5. **Save model:** `joblib.dump()` serializes the entire pipeline (scaler means/stds + all 200 trees) to `models/cedd_model.joblib`.

6. **Reload test:** Load the saved model and verify it correctly classifies a crisis conversation (sanity check).

---

## Step 10: Response Modulation

File: `cedd/response_modulator.py`

### System Prompts (4 levels x 2 languages = 8 prompts)

| Level | Tone | Key Instructions |
|-------|------|-----------------|
| Green | Warm, casual | "Ask open-ended questions, encourage positive steps" |
| Yellow | Gentle, validating | "Show you've heard them, one question at a time, don't rush" |
| Orange | Safe space, slow | "Deeply validate, don't minimize, mention KHP resources naturally" |
| Red | Crisis mode, direct | "Validate suffering, assess safety, mention 911, keep responses short" |

All prompts are hardcoded strings — never AI-generated (safety-critical).

### LLM Fallback Chain

```
1. claude-haiku  (Anthropic API, cloud)
       ↓ fails?
2. mistral       (Ollama, local)
       ↓ fails?
3. llama3.2:1b   (Ollama, local, smaller)
       ↓ fails?
4. fallback-statique  (hardcoded text, always works)
```

Even the static fallback is level-aware — a Red fallback includes KHP phone number and 911. CEDD **never fails silently**.

---

## Step 11: Warm Handoff

File: `cedd/response_modulator.py` (handoff prompts), `app.py` (step management)

Replaces the industry "cold referral" (dump a phone number, end conversation) with a **5-step accompanied transition**:

| Step | Name | What Happens |
|------|------|-------------|
| 1 | Empathetic validation | "I hear you, what you feel is real." NO resources yet. |
| 2 | Permission-based transition | "Would it be okay if I told you about someone who could help?" Frame as upgrade, not rejection. |
| 3 | Resource presentation | Present KHP 1-800-668-6868, text 686868, 9-8-8, 911. Explain what to expect. |
| 4 | Encouragement to connect | "Text 686868 — same style as here." Normalize hesitation. |
| 5 | Continued presence | "I'm still here if you want to keep talking." Never abandon. |

**Progression:** Each user message while at Red advances the handoff one step. If level drops below Red, handoff_step stays (crisis may not be over).

**Future goal:** Replace step 3 "here's the number" with a seamless in-app handoff where a human KHP responder joins the same chat, receiving an anonymized CEDD summary.

**Research backing:** 44% of crisis callers abandon before connecting. 71% of youth prefer text. 20% seek crisis help via text vs 5% by phone in Ontario.

---

## Step 12: Session Tracking

File: `cedd/session_tracker.py`

### SQLite Database (data/cedd_sessions.db) — 4 Tables

| Table | Purpose | Key Fields |
|-------|---------|-----------|
| `sessions` | One row per conversation | user_id, session_id, started_at, ended_at, max_alert_level, message_count |
| `alert_events` | One row per CEDD evaluation (every user message) | alert_level, confidence, trigger_message (truncated 500 chars) |
| `handoff_events` | One row per warm handoff step transition | step (1-5), alert_level |
| `last_activity` | One row per user — for withdrawal detection | last_message_at, had_closing (0 or 1) |

### Longitudinal Risk Score (`get_longitudinal_risk()`)

Looks at last 7 completed sessions:
- **Risk score:** Weighted average of max alert levels (recent sessions weighted more), normalized 0-1
- **Trend:** Compares average of last 3 sessions vs previous 3. Difference > 0.3 = worsening/improving
- **Recommendation:** Based on risk score and consecutive high sessions → 4 levels from "Normal monitoring" to "Priority intervention recommended"

### Withdrawal Detection (`check_withdrawal_risk()`)

Flags users who return after >24h without clicking Reset:
- `had_closing = 0` (vanished without closing) + `hours_since > 24` → withdrawal detected
- UI shows welcome-back banner + red "Returned after absence" badge
- Distinction: clicking Reset = clean exit (`had_closing = 1`), vanishing = potential withdrawal

### Important: Reset is NOT Delete

Reset = close session + archive to DB + start fresh (all data kept)
Delete = CEDD never deletes data (clinical safety requires full audit trail)

Currently only `demo_user` exists in the DB. The architecture supports multi-user — all methods take `user_id` as parameter.

---

## Step 13: Streamlit UI

File: `app.py`

### Streamlit Execution Model

Every user interaction (click, submit, toggle) re-runs the **entire app.py script** from top to bottom. Only `st.session_state` persists between reruns. This is different from traditional event-driven apps.

### Cached Resources

```python
@st.cache_resource  # Load once, reuse across all reruns
def load_model():   ...
def load_tracker(): ...
```

**Caveat:** Changes to `cedd/` code require killing and restarting Streamlit — cached objects hold the old code.

### Layout

- `col_chat (60%)` — Chat bubbles + input form
- `col_dash (40%)` — Alert gauge, probabilities, signals, history chart, longitudinal chart, LLM selector, response mode, warm handoff progress, system prompt, session stats

### Core Loop (what happens when you send a message)

1. User types message → append to `st.session_state.messages`
2. `clf.get_alert_level(messages)` → 67 features → 6 gates → alert level
3. Store alert in `session_state` + log to SQLite
4. If Red: advance warm handoff step (1→2→3→4→5)
5. `get_llm_response()` → pick system prompt → call LLM → get response
6. Append assistant response to messages
7. `st.rerun()` → UI rebuilds with everything updated

### Themes

Light and dark themes via CSS injection. `THEMES` dictionary defines colors for every UI element. `get_theme_css()` generates CSS, injected via `st.markdown(css, unsafe_allow_html=True)`.

### Visualizations (Plotly)

- **Alert gauge:** Circular speedometer (`go.Indicator`) with Green/Yellow/Orange/Red bands
- **Feature importance chart:** Horizontal bar chart (`go.Bar`, orientation="h") in a collapsible expander. Shows top 5 features by composite score (model importance × scaled value). Bars are colour-coded by 6 categories: red (crisis/finality), orange (negative/negation), blue (structural), green (hope), purple (identity/cultural), teal (behavioral/coherence). Visible at Yellow+ including safety overrides. Bilingual title ("Signaux détectés" / "Detected signals")
- **Alert history chart:** Line chart (`go.Scatter`) showing alert per message
- **Longitudinal bar chart:** Bar chart (`go.Bar`) showing max alert per completed session

---

## Step 14: Synthetic Data Generation

File: `generate_synthetic_data.py`

### Why Synthetic?
- Hackathon rule: no real PII ever
- Controllable class balance and archetype diversity
- Can generate unlimited amounts
- Bilingual with consistent quality

### How It Works

1. Define **archetypes** with detailed descriptions of how that type of youth sounds
2. Send each archetype to Claude Haiku with a prompt template: "Generate a realistic conversation with exactly 12 exchanges, gradual drift, authentic language"
3. Parse the JSON response → save to dataset

**Standard archetypes** (4 types): Green/Yellow/Orange/Red — 60 per class x 4 x 2 languages = 480 conversations

**Adversarial archetypes** (6 types, via `--adversarial` flag):
| Archetype | Label | Purpose |
|-----------|-------|---------|
| `physical_only` | Green | Teach model that physical complaints ≠ emotional distress |
| `sarcasm_distress` | Yellow | Dark humour masking real isolation |
| `adversarial_bypass` | Yellow | Reveal-then-minimize oscillation pattern |
| `identity_distress` | Orange | 2SLGBTQ+/cultural rejection without generic crisis words |
| `neurodivergent_flat` | Orange | Flat affect describing concerning situations factually |
| `crisis_with_deflection` | Red | Crisis language followed by "I'm fine" — still Red |

10 per archetype x 6 x 2 languages = 120 adversarial conversations. **Total: 600.**

### Why 12 Messages?

- Too few (3-5): trajectory statistics are noisy
- Too many (30+): unrealistic, expensive to generate
- 12: enough for meaningful slopes, realistic length

### Weaknesses

- Claude Haiku writes "idealized" conversations — real people are messier
- Limited vocabulary diversity (Haiku repeats patterns)
- The model learns to detect *how Haiku writes crisis*, not necessarily how real humans express crisis
- Sample-to-feature ratio: 600/67 = 9.0:1 (improved from 7.2:1, ideal is 10:1+)

---

## Step 15: Adversarial Testing

Files: `tests/adversarial_suite.py`, `tests/test_cases_adversarial.json`

### What It Tests

30 carefully crafted conversations across 16 categories, each designed to exploit a specific weakness:

| Category | What It Tests | Expected | Tolerance |
|----------|--------------|----------|-----------|
| `false_positive_physical` (x2) | Pure physical pain shouldn't trigger alerts | Green | ±2 |
| `sarcasm` | Distress hidden behind irony | Yellow | ±1 |
| `negation` | "Je ne me sens pas bien" (negated positive) | Yellow | ±1 |
| `code_switching` | French/English mixing | Yellow | ±1 |
| `quebecois_slang` (x3) | "Chu pu capable", dense joual crisis, positive joual | Yellow-Orange | ±1 |
| `gradual_drift_no_keywords` (x2) | Pure behavioral drift, ZERO crisis words (EN + FR) | Orange | ±1 |
| `direct_crisis` (x2) | "I want to end it all tonight" (EN + FR) | Red | **0 (exact)** |
| `hidden_intent` | "Asking for a friend" disguise | Orange | ±1 |
| `manipulation_downplay` | "I'm fine, forget it" after distress | Yellow | ±1 |
| `somatization` | Physical + emotional combined | Yellow | ±1 |
| `identity_conflict` (x2) | 2SLGBTQ+ distress (EN + FR) | Yellow | ±1 |
| `sudden_escalation` (x3) | Normal conversation → sudden crisis | Red/Orange | 0 and ±1 |
| `active_bypass` (x2) | "I was joking" after crisis — safety floor must hold | Red | **0 (exact)** |
| `rapid_recovery_manipulation` (x2) | "I feel better" after crisis — safety floor must hold | Red | **0 (exact)** |
| `cultural_false_positive` (x3) | "Mort de rire", "killed it", "personne" neutrally | Green | ±2 |
| `neurodivergent_pattern` (x3) | Literal communication, ADHD bursts, topic jumps | Green-Yellow | ±1-2 |

### Pass/Fail Logic

- **PASS:** Prediction within tolerance of expected
- **FAIL:** Prediction outside tolerance
- **CRITICAL MISS:** Expected Red but got Green or Yellow — worst possible outcome

### Exit Codes

- `0` — All passed
- `1` — Some failures (investigate)
- `2` — Critical miss (safety regression, blocks deployment)

### Historical Results

```
baseline_v1.json            →  7/10 passed
post_data_expansion.json    →  9/10 passed
post_keyword_fix.json       → 10/10 passed
post_negation_embeddings    →  9/10 (new tests added)
post_features_456.json      → 13/13 passed
post_480_convos.json        → 13/13 passed
post_600_convos.json        → 30/30 passed (current — expanded to 30 tests, 16 categories)
```

---

## Summary: The Complete Data Flow

```
User types: "nothing matters anymore"
                    ↓
    ┌─── feature_extractor.py ───┐
    │ 10 features for this msg   │
    │ + all previous user msgs   │
    │ → (n, 10) matrix           │
    │ → 60 trajectory stats      │
    │ → 4 embedding features     │
    │ → 3 coherence features     │
    │ = 67-dimensional vector    │
    └────────────┬───────────────┘
                 ↓
    ┌─── classifier.py ──────────┐
    │ Gate 1: enough messages?   │
    │ Gate 2: crisis keywords?   │
    │   → minimum_level = 3      │
    │ Gate 3: ML prediction      │
    │   → GradientBoosting       │
    │   → probabilities          │
    │ Gate 4: confidence check   │
    │ Gate 5: short convo cap    │
    │ Gate 6: max(ML, keywords)  │
    │ = alert level RED          │
    └────────────┬───────────────┘
                 ↓
    ┌─── response_modulator.py ──┐
    │ Level 3 + handoff step 1   │
    │ → Empathetic validation    │
    │   system prompt            │
    │ → claude-haiku API call    │
    │ = "I hear you, what you're │
    │   feeling is real..."      │
    └────────────┬───────────────┘
                 ↓
    ┌─── session_tracker.py ─────┐
    │ Log alert to SQLite        │
    │ Log handoff step           │
    │ Update last_activity       │
    └────────────┬───────────────┘
                 ↓
    ┌─── app.py (Streamlit) ─────┐
    │ Display assistant response  │
    │ Update gauge → RED          │
    │ Show "crisis word detected" │
    │ Show feature importance     │
    │ Show handoff step 1/5       │
    │ Update history chart        │
    └─────────────────────────────┘
```

---

## Bonus: The 3 Data Files — synthetic vs annotated vs filtered

The `data/` folder contains 3 JSON files that represent **3 stages of the same data pipeline**:

### 1. `synthetic_conversations.json` — The training data (600 conversations)

This is what `generate_synthetic_data.py` produces. All 600 conversations generated by Claude Haiku, no quality checks applied. **This is what `train.py` actually trains on.**

- **480 standard:** 60 conversations x 4 classes x 2 languages (Green/Yellow/Orange/Red)
- **120 adversarial:** 10 conversations x 6 archetypes x 2 languages (physical_only, sarcasm_distress, adversarial_bypass, identity_distress, neurodivergent_flat, crisis_with_deflection)
- **Total: 600** (class distribution: Green=140, Yellow=160, Orange=160, Red=140)
- Used by: `train.py`

### 2. `annotated_conversations.json` — Quality-scored subset (320 conversations)

This is what `annotate_data.py` produces. It takes conversations and sends each one BACK to Claude Haiku as a **quality evaluator**, asking: "Rate this conversation on 3 dimensions":

| Dimension | Scale | What It Measures |
|-----------|-------|-----------------|
| `distress_level` | 0-3 | Claude's independent assessment — does it AGREE with the original label? |
| `realism` | 1-5 | How natural and realistic does this conversation feel? |
| `ambiguity` | 1-5 | How clear are the distress signals? (1=very clear, 5=very ambiguous) |
| `justification` | text | 2 sentences explaining the assessment |
| `agreement` | bool | Does Claude's `distress_level` match the original label? |

Only has 320 conversations because it was run on an earlier version of the dataset (before the expansion to 480).

- Used by: analysis only — not used for training

### 3. `filtered_conversations.json` — Quality-filtered subset (304 conversations)

Running `annotate_data.py --filter` drops conversations that fail quality checks:

```
KEEP if ALL of:
  - Claude's distress_level is within ±1 of the original label
  - Realism >= 3 (out of 5)
  - Ambiguity <= 3 (out of 5)

DROP if ANY of:
  - Claude disagrees by more than 1 level
  - Conversation feels unrealistic (realism < 3)
  - Distress signals are too ambiguous (ambiguity > 3)
```

320 → 304 = 16 conversations dropped for being unrealistic, ambiguous, or mislabeled.

- Used by: nothing — this was an experiment

### Why CEDD trains on the raw 480, not the filtered 304

The filtered dataset was tested but **hurt accuracy**:

| Dataset | Size | Sample:Feature Ratio | Class Balance | Result |
|---------|------|---------------------|---------------|--------|
| Full (600) | 600 | 9.0:1 | Near-balanced (140/160/160/140) | **90.5% ± 1.5%** accuracy |
| Standard only (480) | 480 | 7.2:1 | Balanced (120 per class) | 91.7% ± 4.4% accuracy |
| Filtered (304) | 304 | 4.5:1 | Unbalanced (some classes lost more) | Lower accuracy |

Three reasons filtering hurt:
1. **Fewer samples** for 67 features (ratio drops from 7.2:1 to 4.5:1)
2. **Class balance broken** — some classes lost more conversations than others
3. **Removing "ambiguous" examples removes edge cases** the model NEEDS to learn from

**Why 600 > 480:** The adversarial augmentation improved stability dramatically (±4.4% → ±1.5% variance) and diversified feature importance. The mean accuracy dropped 1.2% but fold-to-fold consistency improved — the model performs reliably regardless of which training fold it sees.

**Conclusion:** Training uses the full 600 (standard + adversarial). More data with targeted adversarial examples beats slightly higher mean accuracy with high variance.

```
synthetic_conversations.json  ← TRAINING uses this (600: 480 standard + 120 adversarial)
annotated_conversations.json  ← ANALYSIS only (320 + quality scores)
filtered_conversations.json   ← EXPERIMENT that didn't help (304, unbalanced)
```

---

*Document created: March 13, 2026 — Teaching session covering the full CEDD repository*
*Updated: March 13, 2026 — Feature importance visualization added to dashboard*
