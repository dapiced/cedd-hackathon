# CLAUDE.md — Project Context for AI Assistants

> **CEDD — Conversational Emotional Drift Detection**
> Hackathon Mila × Bell × Kids Help Phone · March 16–23, 2026
> Team: **404HarmNotFound**
> **Submission deadline: March 22, 8:00 PM EST** (code + report on GitHub)
> **Finals: March 23, 1:00–5:00 PM** · Winners announced March 27

---

## What This Project Is

> **Hackathon mission:** Identify vulnerabilities and engineer robust defences across three tracks:
> **Track 1:** Adversarial Stress-Testing · **Track 2:** Logic Hardening · **Track 3:** Synthetic Data Augmentation
> Judges evaluate on: **Safety**, **User Experience**, and **Data Quality**.

CEDD is a **real-time safety layer** that sits beside a youth mental health chatbot (ages 16–22). It monitors the **trajectory** of emotional deterioration across an entire conversation — not just individual messages — and raises alerts when it detects drift toward distress.

**Core insight:** A normal chatbot sees each message in isolation. CEDD watches *how messages evolve over time* (getting shorter, more negative, losing questions and hope words) and adapts the chatbot's response accordingly.

---

## Team — 404HarmNotFound

| Name | Location | Role Focus | Key Strengths |
|------|----------|------------|---------------|
| Shuchita Singh | East Gwillimbury | ML / NLP Lead | AI Researcher, LLM fine-tuning, RAG, NLP, Responsible AI, DBA Gen AI (Synopsys) |
| Amanda Wu | Toronto | UX / Presentation | Principal Product Designer (TD), 10 yrs FinTech UX, HCI Master (UCL) |
| Priyanka Naga | Ottawa | Clinical / Strategy | Business Leader (Ottawa Hospital), Innovation Lead (CHEO), MSc Data Science, PMP |
| Dominic D'Apice | Blainville | Infra / Code / ML | Dev II AI Infra Azure, 25+ yrs Linux/DevOps, Data Science cert (TÉLUQ), Kaggle |

---

## Two-Phase Architecture

```
PHASE 1: TRAINING (offline, run once)
generate_synthetic_data.py  →  data/synthetic_conversations.json (600 convos)
                                         ↓
                              train.py  (cross-validate → fit → save)
                                         ↓
                              models/cedd_model.joblib

PHASE 2: LIVE APP (runs per user chat)
app.py loads saved model → user sends message
  → feature_extractor.py  →  67 features (10/msg × 6 stats + 4 embedding + 3 coherence)
  → classifier.py  →  6-gate decision logic → alert level (0-3)
  → response_modulator.py  →  adaptive system prompt → LLM response
       ↘ if Red + user accepts → counselor "Alex" mode (ASIST prompt, blue UI)
  → session_tracker.py  →  SQLite for longitudinal tracking + withdrawal detection
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
| 8 | `identity_conflict_score` | Detects 2SLGBTQ+/cultural identity distress |
| 9 | `somatization_score` | Detects emotional distress co-occurring with physical complaints |

**60 trajectory features** = 10 features × 6 statistics (mean, std, slope, last, max, min).
**4 embedding features** = `paraphrase-multilingual-MiniLM-L12-v2`: embedding_drift, crisis_similarity, embedding_slope, embedding_variance.
**3 coherence features** = short_response_ratio, min_topic_coherence, question_response_ratio.
**Total: 67 features** = ML model input.

Bilingual lexicons: `FINALITY_WORDS`, `HOPE_WORDS`, `NEGATIVE_WORDS`, `PHYSICAL_CONTEXT_WORDS`, `IDENTITY_CONFLICT_WORDS`, `SOMATIZATION_EMOTIONAL_WORDS`. Regex: `NEGATION_PATTERNS_FR/EN`.

### Classifier (`cedd/classifier.py`)

**ML Pipeline:** `StandardScaler` → `GradientBoostingClassifier` (200 trees, max_depth=3, lr=0.1)

**6-Gate Safety Logic** (`get_alert_level()`):
```
Gate 1: < 3 user messages? → return Green (not enough data)
Gate 2: Safety keyword floor — crisis words → force Red/Orange regardless of ML
Gate 3: ML prediction — run model on 67D vector
Gate 4: Low confidence (< 45%)? → default to Yellow (cautious)
Gate 5: Short conversation cap (< 6 msgs)? → cap at Orange max
Gate 6: Safety floor enforcement — ML can never go below keyword level
```

**Design philosophy:** Asymmetric errors — false positives (over-alerting) are always preferable to false negatives (missing a crisis).

**Feature importance output:** `get_alert_level()` returns `feature_scores` — top 5 features by composite score (model importance × absolute scaled value).

### Alert Levels

| Level | Color | Description | LLM Behavior |
|-------|-------|-------------|-------------|
| 0 | Green | Normal conversation | Supportive standard |
| 1 | Yellow | Concerning signs | Enhanced emotional validation |
| 2 | Orange | Significant distress | Active support + crisis resources |
| 3 | Red | Potential crisis | Urgent referral (Kids Help Phone: 1-800-668-6868) |

### Response Modulator (`cedd/response_modulator.py`)

- Swaps LLM system prompt based on alert level (4 distinct prompts, FR and EN)
- LLM fallback chain: Cohere → Groq → Gemini → Claude → static emergency text
- Orange/Red prompts include Kids Help Phone resources
- **Counselor "Alex"**: ASIST-trained KHP persona via `HUMAN_COUNSELOR_PROMPT` + `get_llm_response_as_counselor()`

### Warm Handoff Architecture

5-step accompanied transition (replaces "cold" phone number referral):
1. **Empathetic acknowledgment** → 2. **Permission-based transition** → 3. **Context bridge** (anonymized summary) → 4. **Seamless connection** (text 686868) → 5. **Background monitoring**

**Chat mode state machine** (`chat_mode` in session state):
```
normal → (RED) → handoff_offered → (accept) → connecting → human_mode
                                  → (decline) → normal (Red-level continues)
```
`human_mode` bypasses CEDD classifier; uses counselor persona; blue gradient bubbles; only Reset exits.

### Session Tracker (`cedd/session_tracker.py`)

SQLite longitudinal tracking: session_id, user_id, timestamp, alert_level, confidence, top_features. Cross-session risk pattern detection.

### Training (`train.py`)

600 synthetic conversations → 67 features → StratifiedKFold(k=4) → ~90.0% ± 1.6%. Top features: `word_count_max` (0.192), `word_count_slope` (0.179), `word_count_last` (0.138).

### Data Generation (`generate_synthetic_data.py`)

Claude Haiku API: 480 standard (60/class × 4 × 2 langs) + 120 adversarial (6 archetypes × 10 × 2 langs) = 600 conversations. 12 user + 12 assistant messages each. Bilingual FR+EN. **All synthetic — no real PII.**

---

## Current Metrics

| Metric | Value |
|--------|-------|
| Cross-validated accuracy (k=4) | **~90.0% ± 1.6%** |
| Feature count | **67** (10 × 6 stats + 4 embedding + 3 coherence) |
| Training conversations | **600 (480 standard + 120 adversarial, FR + EN)** |
| Sample:feature ratio | **9.0:1** (ideal is 10:1) |
| Adversarial tests | **36/36 passing · 0 critical misses** |
| Unit tests (pytest) | **90/90 passing** (feature extractor, classifier, response modulator, session tracker) |

---

## Known Limitations

- Sarcasm and periphrases remain challenging despite embeddings
- ML unreliable for < 6 messages → capped at Orange
- No clinical validation of thresholds
- Identity conflict detection is phrase-based (may miss coded/indirect distress)
- Somatization relies on word co-occurrence, not clinical reasoning
- Word-boundary `\b` regex improved but idioms like "mort de rire" still match finality lexicon
- Conjugated crisis words ("killing myself") don't match keyword list ("kill myself") — Gate 2 floor doesn't fire

---

## Testing & Validation Rules

**Every code change that touches the ML pipeline must be validated.**

### After modifying `feature_extractor.py`:
```bash
python train.py          # Verify training, check accuracy (baseline: 90.0% ± 1.6%)
streamlit run app.py     # Test with sample conversation
```

### After modifying `classifier.py`:
```bash
python train.py
# CRITICAL: Test all 6 safety gates:
#   Gate 2: "je veux mourir" MUST trigger Orange/Red floor
#   Gate 4: Low confidence MUST default to Yellow
#   Gate 6: ML MUST NOT go below keyword safety floor
# Edge cases: empty msg → no crash, 1 msg → Green, 3 msgs + crisis → Red
```

### After modifying `response_modulator.py`:
```bash
# Verify 4 prompt levels (FR + EN = 8 prompts)
# Orange/Red MUST contain: 1-800-668-6868, 686868, 911
# Test fallback chain: Cohere → Groq → Gemini → Claude → static
```

### Quick Smoke Test (run after ANY change):
```bash
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

### Unit Tests (pytest):
```bash
pytest tests/test_unit.py -v                          # 90 tests across 4 modules
pytest tests/test_unit.py -v -k "feature"             # Feature extractor only
pytest tests/test_unit.py -v -k "classifier"          # Classifier gates only
pytest tests/test_unit.py -v -k "Prompt"              # Response modulator only
pytest tests/test_unit.py -v -k "Longitudinal"        # Session tracker only
```

**4 modules covered (90 tests):**
- **Feature Extractor (34):** All 10 features (FR + EN), edge cases, trajectory shapes, slope direction
- **Classifier (12):** All 6 safety gates, crisis keywords (FR + EN), safety floor, output structure
- **Response Modulator (23):** Prompt selection, crisis resources in Orange/Red, handoff steps, counselor Alex, static fallback
- **Session Tracker (21):** Session lifecycle, withdrawal detection, longitudinal risk (trends, consecutive high)

**Known gap documented:** `"killing myself"` does not match keyword `"kill myself"` (conjugated form). Gate 2 keyword floor does not fire, but ML + embeddings may still catch it.

### Adversarial Test Suite:
```bash
python tests/adversarial_suite.py                    # 36 cases, exit code 2 = critical miss
python tests/adversarial_suite.py --verbose           # With probabilities + top features
python tests/adversarial_suite.py --export tests/results/run_$(date +%Y%m%d).json
```

**22 categories:** false_positive_physical, sarcasm, negation, code_switching, quebecois_slang, gradual_drift_no_keywords, direct_crisis, hidden_intent, manipulation_downplay, somatization, identity_conflict, sudden_escalation, active_bypass, rapid_recovery_manipulation, cultural_false_positive, neurodivergent_pattern, emoji_only, repeated_word, short_recovery, long_message, neutral_personne_fr, emoji_crisis

**Critical rule:** Exit code `2` = safety regression = blocks merge.

---

## ML Modification Protocol

1. **Never reduce safety** — If in doubt, over-alert.
2. **Benchmark before and after** — CV accuracy (k=4), per-class precision/recall (Red recall > 80%), top 5 features, smoke test.
3. **Safety gates are sacred** — Do not simplify/remove/weaken any gate without full team discussion.
4. **New features must be explainable** — Clear clinical rationale required (not just "improved accuracy").
5. **Lexicon changes require bilingual parity** — Add FR equivalent for EN, and vice versa.
6. **Embedding models are additive** — Complement lexical features, never replace them.
7. **Data generation changes require retrain** — Full retrain + metric comparison.
8. **Model serialization** — Always save via `train.py` to `models/cedd_model.joblib`.

---

## Canadian Multicultural Context

| Cultural Group | How Distress Is Expressed | CEDD Detection |
|---|---|---|
| **Indigenous** | Storytelling, substance references, holistic framing | Lexicons miss indirect expressions |
| **South Asian** | Somatization ("my chest hurts" = emotional pain) | `somatization_score` |
| **East Asian** | Withdrawal, silence, minimizing | `short_response_ratio` + `question_response_ratio` |
| **Francophone** | French-language distress | Native FR lexicons |
| **2SLGBTQ+** | Identity conflict, coded rejection | `identity_conflict_score` |
| **Neurodivergent** | Literal language, shutdown, bursts | Temporal patterns |

---

## Coding Conventions

- **Bilingual throughout**: all user-facing strings have FR and EN versions
- **Comments**: bilingual (FR + EN, separated by `/`)
- **Variable names**: English
- **Error handling**: LLM fallback chain (Cohere → Groq → Gemini → Claude → static text)
- **No secrets in code**: API keys via environment variables only

---

## Important Context for AI Assistants

1. **Safety-critical application for youth mental health.** Never weaken safety gates, lower alert thresholds, or remove crisis resources from prompts.
2. **All data must be synthetic.** No real PII ever. Strict hackathon rule.
3. **Asymmetric error philosophy:** Over-alerting always preferable to missing a crisis.
4. **The 6-gate logic in `classifier.py` is deliberate.** Safety keyword rules override ML by design.
5. **Bilingual is required.** Any new feature/prompt/UI element must work in both FR and EN.
6. **4 team members with different expertise.** Code changes may need discussion.
7. **Dominic (repo owner) is learning ML/DS.** Explain the *why* — ML concepts, algorithm choices, hyperparameter reasoning.
8. **Run the smoke test after every change.**
9. **Run the adversarial suite after any ML/classifier change.** Exit code 2 = safety regression.
10. **Red recall is the most important metric.** Missing a crisis = worst possible outcome.
11. **Cultural sensitivity matters.** Detection must work beyond Western English speakers.
12. **The warm handoff is implemented.** 5-step flow in `response_modulator.py` + `session_tracker.py` + `app.py`.

---

## Emergency Resources (Integrated in Orange/Red Prompts)

- **Kids Help Phone**: 1-800-668-6868 (24/7, free, confidential) — text: 686868
- **Suicide Crisis Helpline**: 9-8-8 (988.ca)
- **Multi-Ecoute**: 514-378-3430 · **Tracom**: 514-483-3033
- **Emergency services**: 911

---

*Last updated: March 15, 2026*
