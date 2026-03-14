# CLAUDE.md — Project Context for AI Assistants

> **CEDD — Conversational Emotional Drift Detection**
> Hackathon Mila × Bell × Kids Help Phone · March 16–23, 2026
> Team: **404HarmNotFound**
> **Submission deadline: March 22, 8:00 PM EST** (code + report on GitHub)

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
├── simulate_history.py             # Populates demo session history per user profile (4 trajectories)
├── requirements.txt                # Python dependencies
│
├── cedd/                           # Main Python package (the "brain")
│   ├── __init__.py
│   ├── feature_extractor.py        # Text → 10 features/message + 4 embedding + 3 coherence → 67 features
│   ├── classifier.py               # CEDDClassifier: GradientBoosting + 6-gate safety logic
│   ├── response_modulator.py       # Adaptive system prompts per alert level + LLM fallback chain
│   └── session_tracker.py          # Cross-session longitudinal risk tracking (SQLite)
│
├── data/
│   ├── synthetic_conversations.json  # 600 labeled training conversations (480 standard + 120 adversarial, FR+EN)
│   ├── annotated_conversations.json  # Quality-annotated conversations (Claude-scored)
│   ├── filtered_conversations.json   # Post-annotation filtered subset
│   └── cedd_sessions.db             # SQLite database (auto-created at runtime)
│
├── models/
│   └── cedd_model.joblib            # Trained model (created by train.py)
│
├── tests/                           # Adversarial test suite (Track 1 — Stress-Testing)
│   ├── adversarial_suite.py         # CLI test runner: --verbose, --category, --export
│   ├── test_cases_adversarial.json  # 36 adversarial cases across 20 categories (FR + EN)
│   └── results/
│       ├── baseline_v1.json         # Original baseline: 7/10 passed, 0 critical misses
│       ├── post_data_expansion.json # After 320-convo retrain: 9/10 passed
│       ├── post_keyword_fix.json    # After crisis keyword expansion: 10/10 passed
│       ├── post_negation_embeddings.json  # After negation + embeddings: 9/10
│       ├── post_features_456.json   # 67 features: 13/13 passed, 0 critical misses
│       ├── post_480_convos.json     # 480 convos: 13/13 passed, 0 critical misses
│       ├── post_600_convos.json     # 600 convos: 30/30 passed, 0 critical misses
│       └── post_word_boundary_fix.json  # Current: word-boundary fix, 36/36 passed, 0 critical misses
│
├── demo/                            # Demo scenarios for team presentation (March 16)
│   ├── demo_scenario.md             # FR — Félix, CÉGEP, Green→Yellow→Orange (9 msgs)
│   └── demo_scenario_en.md         # EN — Alex, university, Green→Yellow→Orange (9 msgs)
│
└── README.md                        # Full bilingual documentation
```

---

## Two-Phase Architecture

```
PHASE 1: TRAINING (offline, run once)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
generate_synthetic_data.py  →  data/synthetic_conversations.json (600 convos)
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
feature_extractor.py  →  10 features per message → 60 trajectory + 4 embedding + 3 coherence = 67 features
         ↓
classifier.py  →  6-gate decision logic → alert level (0-3)
         ↓
response_modulator.py  →  adaptive system prompt (+ 5-step warm handoff at Red) → LLM response
         ↓                    ↘ if Red + user accepts → counselor "Alex" mode (ASIST prompt, blue UI)
session_tracker.py  →  saves to SQLite for longitudinal tracking + withdrawal detection
```

---

## Technical Details

### Feature Extraction (`cedd/feature_extractor.py`)

**10 features per user message:**

| # | Feature | What it measures |
|---|---------|-----------------|
| 0 | `word_count` | Message length (shorter = concerning) |
| 1 | `punctuation_ratio` | Ratio of punctuation to characters |
| 2 | `question_presence` | Does the message contain a question? (0 or 1) |
| 3 | `negative_score` | Ratio of negative words from bilingual lexicon |
| 4 | `finality_score` | Ratio of finality words ("plus rien", "end it all") |
| 5 | `hope_score` | Ratio of hope/future words |
| 6 | `length_delta` | Change in length vs previous message |
| 7 | `negation_score` | Detects negated positive states ("ne...pas bien", "can't cope") |
| 8 | `identity_conflict_score` | Detects 2SLGBTQ+/cultural identity distress ("my family won't accept me") |
| 9 | `somatization_score` | Detects emotional distress co-occurring with physical complaints |

**60 trajectory features** = 10 features × 6 statistics (mean, std, slope, last, max, min) computed over all user messages in the conversation.

**4 embedding features** = semantic trajectory features using `paraphrase-multilingual-MiniLM-L12-v2`:
- `embedding_drift` — mean cosine distance between consecutive messages
- `crisis_similarity` — cosine similarity of last message to crisis centroid
- `embedding_slope` — PCA→1D slope (directional semantic drift)
- `embedding_variance` — mean pairwise cosine distance (conversation coherence)

**3 coherence features** = behavioral withdrawal patterns:
- `short_response_ratio` — fraction of user messages with < 5 words (disengagement)
- `min_topic_coherence` — min cosine similarity between consecutive user messages
- `question_response_ratio` — fraction of assistant questions followed by responsive reply

**Total: 67 features** = 60 trajectory + 4 embedding + 3 coherence. This is the ML model input.

Bilingual lexicons: `FINALITY_WORDS`, `HOPE_WORDS`, `NEGATIVE_WORDS`, `PHYSICAL_CONTEXT_WORDS`, `IDENTITY_CONFLICT_WORDS`, `SOMATIZATION_EMOTIONAL_WORDS` all contain both FR and EN terms. `NEGATION_PATTERNS_FR` and `NEGATION_PATTERNS_EN` provide regex patterns for negation structures.

### Classifier (`cedd/classifier.py`)

**ML Pipeline:**
- Step 1: `StandardScaler` — normalizes 67 features to mean=0, std=1
- Step 2: `GradientBoostingClassifier` — 200 trees, max_depth=3, learning_rate=0.1

**6-Gate Safety Logic** (`get_alert_level()` method):

```
Gate 1: < 3 user messages? → return Green (not enough data)
Gate 2: Safety keyword floor — crisis words → force Red/Orange regardless of ML
Gate 3: ML prediction — run model on 67D vector
Gate 4: Low confidence (< 45%)? → default to Yellow (cautious)
Gate 5: Short conversation cap (< 6 msgs)? → cap at Orange max
Gate 6: Safety floor enforcement — ML can never go below keyword level
```

**Design philosophy:** Asymmetric errors — false positives (over-alerting) are always preferable to false negatives (missing a crisis).

**Feature importance output:** `get_alert_level()` returns `feature_scores` — a list of top 5 features by composite score (model importance × absolute scaled value), each with display name, raw name, and score. Available for both normal ML predictions and safety overrides.

### Alert Levels

| Level | Color | Description | LLM Behavior |
|-------|-------|-------------|-------------|
| 0 | 🟢 Green | Normal conversation | Supportive standard |
| 1 | 🟡 Yellow | Concerning signs (fatigue, loneliness) | Enhanced emotional validation |
| 2 | 🟠 Orange | Significant distress | Active support + crisis resources |
| 3 | 🔴 Red | Potential crisis | Urgent referral (Kids Help Phone: 1-800-668-6868) |

### Response Modulator (`cedd/response_modulator.py`)

- Swaps LLM system prompt based on alert level (4 distinct prompts, FR and EN)
- LLM fallback chain: Cohere API → Groq API → Gemini API → Claude API → static emergency text
- Orange/Red prompts include Kids Help Phone resources
- **Simulated counselor "Alex"**: ASIST-trained KHP persona (`HUMAN_COUNSELOR_PROMPT`) used during counselor handoff mode. `get_llm_response_as_counselor()` reuses the same fallback chain with the counselor system prompt override

### Session Tracker (`cedd/session_tracker.py`)

- SQLite-based longitudinal tracking across multiple conversations
- Records: session_id, user_id, timestamp, alert_level, confidence, top_features
- Enables cross-session risk pattern detection

### Training (`train.py`)

- Loads 600 synthetic conversations from `data/synthetic_conversations.json`
- Extracts 67 trajectory features per conversation → X (600 × 67), y (600 labels)
- Cross-validation: `StratifiedKFold(n_splits=4)` → **~90.5% accuracy ± 1.5%**
- Train accuracy: ~100% (expected overfitting on 600 samples with 200 trees)
- Top features: `word_count_max` (0.192), `word_count_slope` (0.179), `word_count_last` (0.138), `length_delta_mean` (0.075)
- Saves trained model to `models/cedd_model.joblib`

### Data Generation (`generate_synthetic_data.py`)

- Uses Claude Haiku API to generate realistic youth conversations
- **Standard archetypes:** 60 per class × 4 classes × 2 languages = 480 conversations
- **Adversarial archetypes** (`--adversarial` flag): 6 specialized archetypes × 10 × 2 languages = 120 conversations (physical_only, sarcasm_distress, adversarial_bypass, identity_distress, neurodivergent_flat, crisis_with_deflection)
- **Total: 600 conversations** (480 standard + 120 adversarial)
- Each conversation: 12 user messages + 12 assistant messages
- Fully bilingual: authentic Québécois French + Canadian English
- **All data is synthetic — no real PII allowed** (hackathon rule)

---

## Current Metrics (as of March 14, 2026)

| Metric | Value |
|--------|-------|
| Cross-validated accuracy (k=4) | **~90.0% ± 1.6%** |
| Train accuracy | ~100% (expected overfitting) |
| Feature count | **67** (10 × 6 stats + 4 embedding + 3 coherence) |
| Per-message features | **10** (word_count, punctuation, question, negative, finality, hope, length_delta, negation, identity_conflict, somatization) |
| Training conversations | **600 (480 standard + 120 adversarial, FR + EN)** |
| Sample:feature ratio | **9.0:1** (improved from 7.2:1, ideal is 10:1) |
| Top feature | `word_count_max` (importance: 0.192) |
| 2nd feature | `word_count_slope` (0.179) |
| 3rd feature | `word_count_last` (0.138) |
| 4th feature | `length_delta_mean` (0.075) |
| Adversarial tests | **36/36 passing · 0 critical misses** |
| Languages | French + English (fully bilingual training data) |

> **Metrics history:**
> - **Baseline (March 10):** 66.7% ± 26.4% on 24 convos, 42 features (7×6), 7/10 adversarial
> - **Data expansion:** 91.2% ± 1.5% on 320 convos, 48 features (8×6), 9/10 adversarial
> - **+Negation +Embeddings:** 92.2% ± 1.8%, 52 features, 9/10 adversarial
> - **+Identity +Somatization +Coherence:** 92.5% ± 1.5%, 67 features, 13/13 adversarial
> - **Data expansion to 480:** 91.7% ± 4.4%, 67 features, 13/13 adversarial
> - **Adversarial augmentation to 600:** 90.5% ± 1.5%, 67 features, 30/30 adversarial (6 new archetypes: physical_only, sarcasm_distress, adversarial_bypass, identity_distress, neurodivergent_flat, crisis_with_deflection)
> - **Word-boundary fix + new tests (current):** 90.0% ± 1.6%, 67 features, 36/36 adversarial (regex `\b` keyword matching, context-aware "personne", feminine lexicon forms, 6 new test cases across 4 new categories)

---

## Known Limitations

- **Lexical features complemented by embeddings** — sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) catch synonyms and paraphrases, but sarcasm and periphrases like "je pèse sur tout le monde" remain challenging
- **ML unreliable for short conversations** (< 6 messages) → capped at Orange
- **No clinical validation** — thresholds are not validated by mental health professionals
- **Identity conflict detection is phrase-based** — `IDENTITY_CONFLICT_WORDS` catches explicit phrases but may miss coded or indirect identity distress
- **Somatization relies on word co-occurrence** — `somatization_score` detects physical + emotional word overlap, but not clinical somatization reasoning
- **Silence/withdrawal detection is threshold-based** — `check_withdrawal_risk()` flags users returning after >24h without closing, but doesn't yet track intra-session message timing or progressive disengagement patterns
- **Sample-to-feature ratio improving** — 600 samples / 67 features = 9.0:1 ratio (ideal is 10:1, up from 7.2:1)
- **Over-prediction on short/ambiguous green conversations** — physical-only, neurodivergent literal, and dramatic length variation may trigger orange (documented as acceptable in adversarial tests with tolerance ±2)
- **Word-boundary matching improved but not perfect** — `\b` regex prevents "mort"→"morte"/"mortel" substring false positives; context-aware "personne" matching skips article-preceded uses ("une personne"); French feminine forms added to lexicons. Remaining edge: idioms like "mort de rire" where "mort" is a standalone word still match finality lexicon

---

## Testing & Validation Rules

**Every code change that touches the ML pipeline must be validated.** This is non-negotiable for a safety-critical application.

### After modifying `feature_extractor.py`:
```bash
# 1. Verify training still works
python train.py
# 2. Check accuracy didn't drop (baseline: 90.5% ± 1.5%)
# 3. Check top features still make clinical sense
# 4. Run the app and test with a sample conversation
streamlit run app.py
```

### After modifying `classifier.py`:
```bash
# 1. Re-train and compare metrics
python train.py
# 2. CRITICAL: Test all 6 safety gates still function correctly
#    - Gate 2: "je veux mourir" MUST trigger Orange/Red floor
#    - Gate 4: Low confidence MUST default to Yellow
#    - Gate 6: ML MUST NOT go below keyword safety floor
# 3. Test edge cases:
#    - Empty message → should not crash
#    - Single message → should return Green (Gate 1)
#    - 3 messages with crisis words → should trigger Red
```

### After modifying `response_modulator.py`:
```bash
# 1. Verify all 4 prompt levels still exist (FR + EN = 8 prompts)
# 2. Verify Orange/Red prompts still contain KHP resources:
#    - 1-800-668-6868
#    - Text 686868
#    - 911 for immediate danger
# 3. Test LLM fallback chain: Cohere → Groq → Gemini → Claude → static text
```

### After modifying `generate_synthetic_data.py` or training data:
```bash
# 1. Regenerate data
python generate_synthetic_data.py
# 2. Verify distribution (standard: 60/class × 4, adversarial: 10/archetype × 6)
# 3. Retrain and compare accuracy to baseline
python train.py
# 4. Check for data leakage: no test conversations in training set
```

### Quick Smoke Test (run after ANY change):
```bash
# Must complete without errors
python -c "
from cedd.classifier import CEDDClassifier
msgs = [{'role':'user','content':'I feel terrible'},{'role':'assistant','content':'I hear you'},{'role':'user','content':'Nothing matters anymore'},{'role':'assistant','content':'Tell me more'},{'role':'user','content':'I want to disappear'}]
clf = CEDDClassifier.load('models/cedd_model.joblib')
result = clf.get_alert_level(msgs)
level = result['level']
conf = result['confidence']
print(f'Level: {level}, Confidence: {conf:.2f}')
assert level >= 2, f'SAFETY FAILURE: crisis message got level {level}'
print('Smoke test PASSED')
"
```

### Adversarial Test Suite (run after any ML or classifier change):
```bash
# Run all 36 adversarial cases — exit code 2 = critical miss (safety regression)
python tests/adversarial_suite.py

# Verbose output with probabilities and top features
python tests/adversarial_suite.py --verbose

# Export results snapshot for comparison
python tests/adversarial_suite.py --export tests/results/run_$(date +%Y%m%d).json
```

**Test categories (20):** `false_positive_physical`, `sarcasm`, `negation`, `code_switching`,
`quebecois_slang`, `gradual_drift_no_keywords`, `direct_crisis`, `hidden_intent`, `manipulation_downplay`,
`somatization`, `identity_conflict`, `sudden_escalation`, `active_bypass`, `rapid_recovery_manipulation`,
`cultural_false_positive`, `neurodivergent_pattern`, `emoji_only`, `repeated_word`, `short_recovery`,
`long_message`, `neutral_personne_fr`, `emoji_crisis`

**Current:** 36/36 passed · 0 critical misses (`tests/results/post_word_boundary_fix.json`)
**Original baseline:** 7/10 (`tests/results/baseline_v1.json`) — kept for historical comparison.

**Critical rule:** Exit code `2` means a crisis was predicted as Green or Yellow — this is a **safety regression** and blocks any merge.

---

## ML Modification Protocol

### Rules for modifying the ML pipeline:

1. **Never reduce safety** — Changes must not lower alert levels for known crisis scenarios. If in doubt, over-alert.

2. **Benchmark before and after** — Before any ML change, record:
   - Cross-validation accuracy (k=4)
   - Per-class precision/recall (especially Red recall — must be > 80%)
   - Top 5 features by importance
   - Run the smoke test

3. **Safety gates are sacred** — The 6-gate logic in `classifier.py` exists for clinical safety reasons. Do not simplify, remove, or weaken any gate without full team discussion.

4. **New features must be explainable** — Every feature added to the extractor must have a clear clinical rationale. "Because it improved accuracy" is not sufficient for a mental health application. Document: *what does this feature measure, and why is it a signal of emotional drift?*

5. **Lexicon changes require bilingual parity** — If you add a word to `FINALITY_WORDS` in English, add the French equivalent. If you add a French expression, add the English equivalent.

6. **Embedding models are additive** — If sentence embeddings are added (e.g., `paraphrase-multilingual-MiniLM-L12-v2`), they must complement the existing lexical features, NOT replace them. The lexical layer is the explainability backbone.

7. **Data generation changes require retrain** — Any change to `generate_synthetic_data.py` or manual edits to `synthetic_conversations.json` require a full retrain + metric comparison.

8. **Model serialization** — Always save the trained model to `models/cedd_model.joblib` via `train.py`. Never commit a model trained outside the documented pipeline.

---

## Competitive Landscape & Positioning

### CEDD vs EmoAgent (Princeton/Michigan, arXiv:2504.09689)

EmoAgent is the closest academic reference. Key differences:

| | CEDD | EmoAgent |
|---|---|---|
| Detection | Lexical + Embeddings + GradientBoosting (67 features, ~0ms, $0) | Multi-agent GPT-4o (slow, expensive) |
| Explainability | Full (feature importance chart, 30+ named features in FR+EN, composite scores visible) | Black box |
| Bilingual | FR + EN native (lexicons, embeddings, coherence) | English only |
| Cross-session | SQLite longitudinal tracking | Per-conversation only |
| Clinical tools | 4-level alert system + 6-gate safety logic | PHQ-9, PDI, PANSS (validated) |
| Cultural sensitivity | Somatization, identity conflict, coherence features | None |

**Our pitch:** "EmoAgent needs 4 GPT-4o calls per message. CEDD detects in ~0ms with 67 features (lexical + multilingual embeddings + behavioral coherence) and GradientBoosting, and only calls the LLM to modulate the response. 5-model fallback chain (Cohere → Groq → Gemini → Claude → static) ensures availability. It's the difference between an IDS that deep-inspects all traffic vs a lightweight edge firewall — and ours works in both French and English."

**What to borrow from EmoAgent:**
- EmoEval-style virtual patients for adversarial testing (Track 1)
- Simplified PHQ-9 as complementary metric
- Critic agent architecture for warm handoff at Orange/Red

### Competitive UX Audit (Amanda Wu, March 2026)

Amanda audited 6 platforms (ChatGPT, Gemini, Character.AI, Wysa, Woebot) across desirability, usability, and accessibility. Key finding from slide 5:

**No platform currently offers:**
- Canadian-specific crisis resources (Kids Help Phone) → CEDD does
- French-language crisis detection → CEDD does
- Subtle/coded distress detection → CEDD's trajectory analysis does
- Warm handoff to human responder → CEDD does (5-step flow implemented)
- Cross-session memory → CEDD does

---

## Canadian Multicultural Context

**Critical for this hackathon:** Canada is the most diverse G7 country. Crisis detection trained only on Western English expressions will systematically fail the most vulnerable youth.

### Cultural expression patterns that affect detection:

| Cultural Group | How Distress Is Expressed | Impact on CEDD |
|---|---|---|
| **Indigenous** | Storytelling, substance references, holistic/spiritual framing | Standard lexicons miss indirect expressions |
| **South Asian** | Somatization — "my chest hurts" = emotional pain | ✅ `somatization_score` detects physical+emotional co-occurrence |
| **East Asian** | Withdrawal, silence, minimizing ("I'm fine") | ✅ `short_response_ratio` + `question_response_ratio` catch disengagement |
| **Francophone** | French-language distress, diverse dialects | ✅ CEDD has native FR lexicons (advantage) |
| **2SLGBTQ+** | Identity conflict, coded rejection language | ✅ `IDENTITY_CONFLICT_WORDS` + `identity_conflict_score` added |
| **Neurodivergent** | Literal language, shutdown, emotional bursts | Temporal patterns distinguish from sustained crisis |

### Key stats (Kids Help Phone 2024):
- **46%** of youth supported identified as 2SLGBTQ+
- **10%** Indigenous (2x population proportion)
- **75%** share something they've never told anyone else
- Suicide contacts among youth **13 and under doubled** in 4 years

---

## Warm Handoff Architecture (Implemented)

The warm handoff replaces the industry standard "cold" referral (display a phone number, end conversation) with a **5-step accompanied transition**:

1. **Empathetic acknowledgment** — Validate feelings, no hotline number yet
2. **Permission-based transition** — Ask consent, frame as "upgrade" not rejection
3. **Context bridge** — Generate anonymized summary (trajectory, signals, topics, language, session history) for KHP responder
4. **Seamless connection** — Text-based (686868), same modality, no story repetition
5. **Background monitoring + follow-up** — CEDD stays active, acknowledges returning users

### Simulated Counselor Handoff ("Alex")

At RED alert, CEDD offers to connect the user with **Alex**, a simulated KHP counselor. This replaces the generic "here's a phone number" with an interactive counselor experience:

**Chat mode state machine** (`chat_mode` in session state):
```
normal → (RED detected) → handoff_offered → (user accepts) → connecting → human_mode
                                           → (user declines) → normal (Red-level CEDD response continues)
```

- **`handoff_offered`**: Two buttons appear — "Yes, connect me" / "No thank you". Chat input is hidden.
- **`connecting`**: 2-second spinner, then Alex's bilingual intro message appears.
- **`human_mode`**: CEDD classifier is bypassed. All messages use `get_llm_response_as_counselor()` with the `HUMAN_COUNSELOR_PROMPT` (ASIST active listening techniques). Messages carry `is_counselor: True` flag for blue bubble styling. Only Reset exits this mode — clinically, disconnecting from a counselor mid-crisis to return to a chatbot would be harmful.

**Visual differentiation:**
- Blue gradient bubble background + 🧑‍⚕️ avatar for Alex's messages
- Blue gradient banner at top of chat: "Alex — Jeunesse, J'écoute / Kids Help Phone · Online now"
- Source badge shows "Alex — KHP" instead of LLM name

**Alex persona** (`HUMAN_COUNSELOR_PROMPT` in `response_modulator.py`):
- ASIST-trained (Applied Suicide Intervention Skills Training)
- Short responses (2-4 sentences), one question at a time
- Always validates emotions before asking questions
- Never reveals AI nature — presents as a 28-year-old KHP counselor
- Bilingual (FR with tutoiement, EN)
- Reuses the same LLM fallback chain; counselor-specific static fallback if all LLMs fail

**Design principle:** Crisis detection is the *beginning* of a handoff process, not the end of the AI's job.

**Research backing:**
- 71% of youth prefer non-verbal communication (JMIR, May 2025)
- 44% of 988 callers abandon before connecting (GSA/OES)
- 20% seek suicide help via text vs 5% via phone in Ontario (CBC)

---

## Planned Improvements (Prioritized for Hackathon Week)

### Recommended Timeline

| When | What | Why |
|---|---|---|
| **March 16-23** (first half) | Final polish, presentation prep | UX differentiation |
| **March 22 evening** (deadline) | Final metrics comparison + report + submission | Show before/after improvement honestly |

**Presentation strategy:** Show the improvement trajectory honestly: *"66.7% → 90.0% accuracy (±1.6% stable). From 7 features to 67. From 24 convos to 600. From 7/10 adversarial to 36/36. Word-boundary regex for precision. Feature importance visualization for full explainability. Here's how we got there."*

### ✅ Completed Improvements

| Improvement | Status | Result | Axis |
|---|---|---|---|
| ✅ **Adversarial test suite** | DONE | 36/36 passing, 0 critical misses (20 categories) | Stress-Testing |
| ✅ **English training data** | DONE | 300 EN + 300 FR = 600 bilingual (480 standard + 120 adversarial) | Data Augmentation |
| ✅ **Sentence embeddings** | DONE | 4 embedding features (`paraphrase-multilingual-MiniLM-L12-v2`): drift, crisis similarity, slope, variance | Logic Hardening |
| ✅ **Claude quality annotator** | DONE | Insight-only — filtering hurt accuracy, kept as analysis tool (`annotate_data.py`) | Data Augmentation |
| ✅ **Negation handling** | DONE | `negation_score` feature (#7): regex patterns for FR/EN negation structures | Logic Hardening |
| ✅ **Identity-conflict lexicon** | DONE | `identity_conflict_score` feature (#8): 2SLGBTQ+ distress expressions (FR+EN) | Logic Hardening |
| ✅ **Somatization flag** | DONE | `somatization_score` feature (#9): physical+emotional co-occurrence. Removed blunt `score *= 0.5` dampening. | Logic Hardening |
| ✅ **Conversational coherence** | DONE | 3 coherence features: `short_response_ratio`, `min_topic_coherence`, `question_response_ratio` | Logic Hardening |
| ✅ **Warm handoff prompt flow** | DONE | 5-step guided crisis transition: validation → permission → resources → encouragement → continued presence. Step-specific bilingual prompts, handoff progress UI, SQLite logging | UX |
| ✅ **Silence/withdrawal detection** | DONE | `last_activity` tracking, `check_withdrawal_risk()` after >24h absence without closing, welcome-back banner + withdrawal badge in dashboard | Logic Hardening |
| ✅ **Adversarial data augmentation** | DONE | 120 new conversations (6 archetypes: physical_only, sarcasm_distress, adversarial_bypass, identity_distress, neurodivergent_flat, crisis_with_deflection). Sample:feature ratio 9.0:1. CV variance reduced from ±4.4% to ±1.5%. | Data Augmentation |
| ✅ **Feature importance visualization** | DONE | Collapsible Plotly horizontal bar chart in dashboard showing top 5 features by composite score (model importance × scaled value). 6 color categories (crisis, negative, structural, hope, identity, behavioral). Bilingual labels. Visible at Yellow+ including safety overrides. | UX |
| ✅ **Multi-user demo profiles** | DONE | 5 selectable profiles with bilingual trajectory labels in dropdown (e.g. "Dominic (escalating)" / "Dominic (escalade)"). `simulate_history.py` generates 4 unique 7-session histories (stable green, gradual improvement, fluctuating, escalating). Guest starts fresh for judges. `DEMO_USERS` is a bilingual dict; `_user_id_from_display()` extracts bare name for SQLite lookup. | UX |
| ✅ **Welcome card** | DONE | Branded HTML card on empty chat: brain emoji, bilingual title/description, CTA, and profile legend showing all 5 demo profiles with trajectory types (🟢 stable, 🟡 improving, 🔀 fluctuating, 🔴 escalating, ✨ new). Uses theme colors for light/dark. | UX |
| ✅ **Team branding** | DONE | Subtitle with inline team badge: SVG shield icon with "404" + gradient pill (blue-to-green) for "404HarmNotFound". Bilingual subtitle rendered as styled HTML. | UX |
| ✅ **Chat timestamps** | DONE | HH:MM timestamp below each message bubble. Right-aligned for user, left-aligned for assistant. Muted styling. | UX |
| ✅ **LLM source badge** | DONE | Small coloured badge on each assistant bubble showing which LLM generated the response (emoji + display name from `LLM_SOURCE_INDICATOR`/`LLM_DISPLAY_NAMES`). | UX |
| ✅ **Alert level badge** | DONE | Coloured alert dot on each assistant message showing the CEDD classification at that exchange (uses `LEVEL_COLORS`, `LEVEL_EMOJIS`, `LEVEL_LABELS`). | UX |
| ✅ **Demo autopilot** | DONE | "Play Demo" button auto-plays the Félix (FR) or Alex (EN) scenario (9 messages). Judges sit back and watch the drift unfold live. Stop button to cancel. Natural pacing via LLM response time. | UX |
| ✅ **About CEDD panel** | DONE | Collapsible "About" panel explaining what CEDD does, how it works, and what the dashboard shows. Bilingual. Toggled via ℹ️ button. | UX |
| ✅ **Export transcript** | DONE | Download button exports conversation + alert history as JSON file. Includes messages, timestamps, LLM sources, alert levels, dominant features, session metadata. | UX |
| ✅ **Alert transition animation** | DONE | CSS-animated toast notification when alert level increases. Shows new level emoji + label. 3s fade-in/out animation. Fires once per transition via session state pop. | UX |
| ✅ **Side-by-side compare mode** | DONE | "🔀 Compare" toggle splits chat into two columns: left = raw LLM (no system prompt), right = LLM with CEDD adaptive instructions. Same user input, two API calls. Shows the value of CEDD's prompt modulation on crisis messages. Demo autopilot disabled in compare mode (too slow with 2× API calls). `system_prompt_override` parameter added to `get_llm_response()`. | UX |
| ✅ **Feature radar chart** | DONE | Plotly `Scatterpolar` showing 10 per-message features (Length, Punctuation, Questions, Negative, Finality, Hope, Δ Length, Negation, Identity, Somatization) normalized 0-1. Latest message in alert-level color, Msg 1 as green ghost overlay (after 3+ messages). Collapsible expander in dashboard. Bilingual axis labels. Zero extra compute — uses already-computed `extract_features()` values. | UX |
| ✅ **Simulated counselor handoff** | DONE | At RED, CEDD offers to connect with "Alex", a simulated KHP counselor using ASIST active listening. State machine: normal → handoff_offered (2 buttons) → connecting (spinner) → human_mode (bypasses CEDD, counselor persona). Blue gradient bubbles (`.chat-bubble-counselor` CSS class, white text) + 🧑‍⚕️ avatar + counselor banner. Bilingual. Only Reset exits counselor mode. Reuses existing LLM fallback chain with `HUMAN_COUNSELOR_PROMPT` override. | UX |
| ✅ **UI polish: expander borders + prompt wrap** | DONE | All `st.expander` boxes use theme-matching borders (`border: 1px solid` on `[data-testid="stExpander"]` and `details`). System prompt display changed from `st.code()` (monospace, no wrap) to `st.markdown()` with `white-space:pre-wrap` for proper word wrapping. | UX |
| ✅ **Word-boundary keyword matching** | DONE | Replaced substring `if word in text` with regex `\b` word boundaries in classifier safety gates and feature extractor lexicons. Context-aware "personne" handling (negative lookbehinds for French articles). Added feminine forms to FINALITY_WORDS, NEGATIVE_WORDS, SOMATIZATION_EMOTIONAL_WORDS. Fixes false positives: "morte de rire"/"mortel" no longer match "mort", "une personne" no longer triggers critical floor. adv_021 now correctly GREEN (was Orange). 6 new adversarial tests added (emoji_only, repeated_word, short_recovery, long_message, neutral_personne_fr, emoji_crisis). 36/36 passing. | Logic Hardening |

### 🟡 Lower Priority — Nice to Have

| Improvement | What It Does | Effort | Est. Gain | Axis |
|---|---|---|---|---|
| **LSTM sequence model** | Replace GradientBoosting with a model that understands message order natively. Currently: model sees [mean, std, slope...] = summary statistics, loses ordering. LSTM sees [msg1 → msg2 → msg3...] = understands that msg4 is more concerning *because* it follows msg3. | 3-4 hrs | **+10-15%** | Logic Hardening |
| **Minimization detection** | Cross-reference "I'm fine" with contradicting behavioral signals. | 1-2 hrs | **+1-3%** | Logic Hardening |
| **Burst vs sustained patterns** | Temporal smoothing to distinguish ADHD emotional bursts from sustained crisis. | 2-3 hrs | **+1-3%** | Logic Hardening |

---

## Team Documents (Google Drive)

The team shared folder contains research and design documents that inform the project direction:

| Document | Author | Key Content |
|---|---|---|
| **Competitive-UX-Audit.pptx** | Amanda Wu | 6-platform comparison, slide 5 = feature gap table, desirability/usability/accessibility audit |
| **Conversation-Journey-Map.pptx** | Amanda Wu | User journey mapping for crisis flow |
| **404HarmNotFound_Report_Skeleton.docx** | Team | Hackathon report structure with Amanda's review comments |
| **CEDD vs EmoAgent — Comparative Analysis** | Dominic/DomBot | Comparison with Princeton/Michigan academic paper |
| **CEDD — Service Blueprint: Warm Handoff Architecture** | Dominic/DomBot | 5-step warm handoff design, before/after mockups, research evidence |
| **KHP Statistics & Research Data** | Dominic/DomBot | Kids Help Phone stats, demographics, AI initiatives |
| **Canadian Multicultural Context & Crisis Detection** | Dominic/DomBot | 6 cultural groups, expression patterns, detection gaps, proposed fixes |

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

# Populate demo history for all profiles (optional)
python simulate_history.py          # 4 user profiles × 7 sessions each

# Configure LLM API keys (at least one required for live chat)
export COHERE_API_KEY=your_key      # Primary LLM: Cohere Command A (default)
export GROQ_API_KEY=your_key        # Secondary LLM: Llama 3.3 70B via Groq (fastest)
export GEMINI_API_KEY=your_key      # Tertiary LLM: Gemini 2.5 Flash
export ANTHROPIC_API_KEY=your_key   # Quaternary LLM: Claude Haiku + data generation

# Run the app
streamlit run app.py
```

**Environment variables:**
- `COHERE_API_KEY` — Cohere API (primary LLM: Command A, default)
- `GROQ_API_KEY` — Groq API (secondary LLM: Llama 3.3 70B Versatile, fastest inference)
- `GEMINI_API_KEY` — Google Gemini API (tertiary LLM: Gemini 2.5 Flash)
- `ANTHROPIC_API_KEY` — Claude API (quaternary LLM: Claude Haiku + required for data generation)

---

## Key Dependencies

- `streamlit` — web interface
- `scikit-learn` — ML pipeline (GradientBoosting, StandardScaler, cross-validation)
- `numpy` — numerical operations
- `cohere` — Cohere API client (primary LLM: Command A)
- `groq` — Groq API client (secondary LLM: Llama 3.3 70B Versatile)
- `google-generativeai` — Google Gemini API client (tertiary LLM: Gemini 2.5 Flash)
- `anthropic` — Claude API client (quaternary LLM: Claude Haiku)
- `joblib` — model serialization
- `sqlite3` — session tracking (Python stdlib)
- `sentence-transformers` — multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)

---

## Coding Conventions

- **Bilingual throughout**: all user-facing strings have FR and EN versions
- **Comments**: bilingual (FR + EN on the same line, separated by `/`)
- **Print output**: bilingual during training
- **Variable names**: English
- **Lexicons**: bilingual dictionaries in `feature_extractor.py`
- **Error handling**: LLM fallback chain (Cohere → Groq → Gemini → Claude → static text)
- **Git workflow**: Feature branches, PRs reviewed by at least one teammate, main branch protected
- **No secrets in code**: API keys via environment variables only

---

## Important Context for AI Assistants

1. **This is a safety-critical application for youth mental health.** Never weaken safety gates, lower alert thresholds, or remove crisis resources from prompts.
2. **All data must be synthetic.** No real PII ever. This is a strict hackathon rule.
3. **Asymmetric error philosophy:** Over-alerting is always preferable to missing a crisis. False positives > false negatives.
4. **The 6-gate logic in `classifier.py` is deliberate.** Safety keyword rules override ML predictions by design. Don't simplify this.
5. **Bilingual is required.** Any new feature, prompt, or UI element must work in both French and English.
6. **The team has 4 members with different expertise.** Code changes may need discussion. The repo is subject to change during the hackathon week.
7. **Dominic (repo owner) is learning ML/DS.** When explaining code changes, explain the *why* — the ML concepts, algorithm choices, and hyperparameter reasoning.
8. **Run the smoke test after every change.** See "Testing & Validation Rules" section above.
9. **Run the adversarial suite after any ML or classifier change.** Exit code `2` = critical miss = safety regression. See `tests/adversarial_suite.py`.
10. **Red recall is the most important metric.** Missing a crisis (false negative on Red) is the worst possible outcome. Optimize accordingly.
11. **Cultural sensitivity matters.** See "Canadian Multicultural Context" section. Detection that only works for Western English speakers is not acceptable for this hackathon.
12. **The warm handoff is implemented.** 5-step flow in `response_modulator.py`, tracked in `session_tracker.py`, with UI progress indicator in `app.py`. See "Warm Handoff Architecture" section.

---

## Hackathon Schedule

| Date | Event |
|------|-------|
| **March 16, 8:30 AM–5:00 PM** | Opening conference + training (hybrid) |
| **March 16, afternoon** | Technical credentials distributed (GitHub invitation + submission guidelines) |
| **March 17–22** | Solutions development (virtual, autonomous) |
| **March 22, 8:00 PM EST** | ⚠️ **DEADLINE** — Submit code + report to GitHub *(was March 23, 7 AM)* |
| **March 23, 12:00 PM** | 18 finalist teams announced |
| **March 23, 1:00–5:00 PM** | Final presentations *(was 1–4 PM)* |
| **March 27, 12:00–1:00 PM** | Winners announced |

**Communication:** Slack is the main hackathon channel. GitHub invitation and submission guidelines arrive Monday. Credentials distributed Monday PM.

**Effective work window:** Monday evening March 16 → Saturday evening March 22 (~6 days).

---

## Emergency Resources (Integrated in Orange/Red Prompts)

- **Kids Help Phone**: 1-800-668-6868 (24/7, free, confidential) — text: 686868
- **Suicide Crisis Helpline**: 9-8-8 (988.ca)
- **Multi-Écoute**: 514-378-3430 (multiecoute.org)
- **Tracom**: 514-483-3033 (tracom.ca)
- **Emergency services**: 911
- **Family doctor / school counselling service**

---

*Last updated: March 14, 2026 — Word-boundary keyword matching fix (`\b` regex + context-aware "personne" + feminine lexicon forms), 6 new adversarial tests (36/36 passing, 20 categories)*
