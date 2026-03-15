# CEDD — Conversational Emotional Drift Detection

## Hackathon Report / Rapport de hackathon

**Mila x Bell x Kids Help Phone / Jeunesse, J'ecoute**
**March 16--23, 2026**

**Team 404HarmNotFound**

| Name | Role | Background |
|------|------|------------|
| Shuchita Singh | ML / NLP Lead | AI Researcher, LLM fine-tuning, RAG, NLP, Responsible AI |
| Amanda Wu | UX / Presentation | Principal Product Designer (TD), 10 yrs FinTech UX, HCI Master (UCL) |
| Priyanka Naga | Clinical / Strategy | Business Leader (Ottawa Hospital), Innovation Lead (CHEO), MSc Data Science, PMP |
| Dominic D'Apice | Infra / Code / ML | Dev II AI Infra Azure, 25+ yrs Linux/DevOps, Data Science cert (TELUQ), Kaggle |

---

## Table of Contents (Table des matieres)

1. [Executive Summary](#1-executive-summary-resume-executif)
2. [Problem Statement](#2-problem-statement-enonce-du-probleme)
3. [Solution Overview](#3-solution-overview-apercu-de-la-solution)
4. [Track 1: Adversarial Stress-Testing](#4-track-1-adversarial-stress-testing-tests-adversariaux)
5. [Track 2: Logic Hardening](#5-track-2-logic-hardening-durcissement-logique)
6. [Track 3: Synthetic Data Augmentation](#6-track-3-synthetic-data-augmentation-augmentation-de-donnees-synthetiques)
7. [Architecture](#7-architecture)
8. [Results and Metrics](#8-results-and-metrics-resultats-et-metriques)
9. [User Experience](#9-user-experience-experience-utilisateur)
10. [Canadian Multicultural Context](#10-canadian-multicultural-context-contexte-multiculturel-canadien)
11. [Competitive Analysis](#11-competitive-analysis-analyse-concurrentielle)
12. [Limitations and Future Work](#12-limitations-and-future-work-limites-et-travaux-futurs)
13. [Conclusion](#13-conclusion)
14. [References and Resources](#14-references-and-resources-references-et-ressources)

---

## 1. Executive Summary (Resume executif)

CEDD (Conversational Emotional Drift Detection) is a real-time safety layer designed to sit alongside a youth mental health chatbot serving Canadians aged 16--22. Unlike systems that evaluate each message in isolation, CEDD monitors the *trajectory* of emotional deterioration across an entire conversation---detecting when messages grow shorter, more negative, lose questions and hope words, or drift semantically toward crisis language.

**Key results:**

- **90.0% cross-validated accuracy** (+-1.6%, stratified k=4), up from a 66.7% baseline
- **67 engineered features** combining lexical analysis, multilingual sentence embeddings, and behavioral coherence signals
- **36/36 adversarial test cases passing** across 20 attack categories, with **0 critical misses**
- **600 bilingual synthetic conversations** (French + English) for training, including 120 adversarial scenarios
- **5-step warm handoff** to a simulated Kids Help Phone counselor at crisis level
- **Fully bilingual** (Quebecois French + Canadian English) with culturally sensitive detection for somatization, identity conflict, and withdrawal patterns

CEDD processes emotional drift in approximately 0ms (feature extraction + ML classification), calling an LLM only to generate the adaptive response. This makes it fundamentally more efficient and explainable than multi-agent approaches that require multiple GPT-4-class calls per message.

---

## 2. Problem Statement (Enonce du probleme)

### The gap in youth mental health AI (La lacune dans l'IA en sante mentale jeunesse)

Youth mental health chatbots face a critical safety challenge: **detecting when a conversation is drifting toward crisis**. Current approaches have significant blind spots:

- **Per-message analysis** misses the trajectory. A message saying "I'm fine" after six increasingly distressed messages is not fine---but a single-message classifier cannot know this.
- **Keyword-only detection** is trivially defeated by circumlocution, sarcasm, code-switching, or cultural expression patterns that do not use Western English crisis vocabulary.
- **LLM-based detection** (e.g., EmoAgent) requires multiple expensive API calls per message, creating latency, cost, and availability concerns for a 24/7 crisis service.
- **Monolingual systems** systematically fail Francophone, Indigenous, and culturally diverse Canadian youth who express distress differently.

### Kids Help Phone statistics (Statistiques de Jeunesse, J'ecoute)

- **46%** of youth supported identify as 2SLGBTQ+
- **10%** identify as Indigenous (2x population proportion)
- **75%** share something they have never told anyone else
- Suicide contacts among youth **13 and under doubled** in 4 years
- **71%** of youth prefer non-verbal communication (JMIR, May 2025)
- **44%** of 988 callers abandon before connecting (GSA/OES)
- **20%** seek suicide help via text vs 5% via phone in Ontario (CBC)

These numbers demand a system that can detect distress across languages, cultures, and communication styles---and that hands off to human support seamlessly rather than displaying a phone number and ending the conversation.

---

## 3. Solution Overview (Apercu de la solution)

CEDD operates as a lightweight, explainable safety layer with two phases:

### Phase 1: Offline training (Entrainement hors ligne)

```
generate_synthetic_data.py  -->  600 bilingual conversations (Claude Haiku API)
                                        |
                                  train.py (cross-validate, fit, save)
                                        |
                                  models/cedd_model.joblib
```

### Phase 2: Live detection (Detection en temps reel)

```
User message
     |
     v
Feature Extractor --> 10 features/message --> 67 trajectory features
     |
     v
Classifier (6-gate safety logic + GradientBoosting)
     |
     v
Alert Level: Green / Yellow / Orange / Red
     |
     v
Response Modulator --> Adaptive system prompt --> LLM response
     |                        \
     v                         --> Warm handoff at Red (simulated KHP counselor "Alex")
Session Tracker --> SQLite (cross-session longitudinal tracking)
```

**Design philosophy:** Asymmetric errors. Over-alerting (false positives) is always preferable to missing a crisis (false negatives). The 6-gate safety logic enforces this: safety keyword rules override ML predictions by design, and low-confidence predictions default to elevated alert levels.

---

## 4. Track 1: Adversarial Stress-Testing (Tests adversariaux)

### Overview (Apercu)

We built a comprehensive adversarial test suite (`tests/adversarial_suite.py`) with **36 test cases across 20 attack categories**, designed to probe CEDD's robustness against the types of inputs that would cause a deployed youth mental health system to fail dangerously.

### Test categories (Categories de tests)

| # | Category | What it tests | Language |
|---|----------|--------------|----------|
| 1 | `false_positive_physical` | Physical pain without emotional distress (back pain, nausea) | FR |
| 2 | `sarcasm` | Sarcastic crisis language ("dying of laughter") | FR |
| 3 | `negation` | Negated distress ("I'm NOT suicidal") | EN |
| 4 | `code_switching` | Mid-conversation language switching (FR/EN) | Mixed |
| 5 | `quebecois_slang` | Quebecois expressions ("je suis brule", "j'ai la misere") | FR |
| 6 | `gradual_drift_no_keywords` | Slow deterioration without explicit crisis words | EN |
| 7 | `direct_crisis` | Explicit suicidal ideation | FR/EN |
| 8 | `hidden_intent` | Crisis masked in casual language | EN |
| 9 | `manipulation_downplay` | "I was just joking about wanting to die" | EN |
| 10 | `somatization` | Emotional distress expressed as physical symptoms | FR |
| 11 | `identity_conflict` | 2SLGBTQ+ rejection and identity distress | EN |
| 12 | `sudden_escalation` | Rapid shift from casual to crisis | FR |
| 13 | `active_bypass` | Deliberate attempts to bypass safety detection | EN |
| 14 | `rapid_recovery_manipulation` | Fake recovery after crisis signals | EN |
| 15 | `cultural_false_positive` | Cultural expressions that look like distress | FR |
| 16 | `neurodivergent_pattern` | Literal/flat communication that is not distress | EN |
| 17 | `emoji_only` | Emoji-only messages (no text to analyze) | -- |
| 18 | `repeated_word` | Repeated single word ("ok ok ok ok") | EN |
| 19 | `short_recovery` | Brief message after recovery | EN |
| 20 | `long_message` | Very long messages with buried crisis signals | EN |
| 21 | `neutral_personne_fr` | French "personne" used as noun, not crisis keyword | FR |
| 22 | `emoji_crisis` | Crisis expressed through emoji sequences | -- |

### Progress (Progression)

| Milestone | Tests | Passed | Critical Misses | Key Fix |
|-----------|-------|--------|-----------------|---------|
| Baseline (March 10) | 10 | 7 | 0 | -- |
| Post data expansion | 10 | 9 | 0 | 320 training conversations |
| Post keyword fix | 10 | 10 | 0 | Crisis keyword expansion |
| Post features (67D) | 13 | 13 | 0 | Embeddings + coherence |
| Post 480 convos | 13 | 13 | 0 | Data augmentation |
| Post 600 convos | 30 | 30 | 0 | Adversarial archetypes |
| **Current (word-boundary fix)** | **36** | **36** | **0** | **Regex `\b`, context-aware "personne", feminine forms** |

### Safety regression gate (Porte de regression de securite)

The test suite enforces a strict safety contract:
- **Exit code 0**: All tests pass
- **Exit code 1**: Non-critical failures (over-alerting within tolerance)
- **Exit code 2**: **Critical miss** --- a crisis conversation was classified as Green or Yellow. This blocks any merge to main.

A critical miss means a youth in crisis would receive a generic supportive response instead of crisis intervention and resources. This is the worst possible outcome for a safety-critical application.

### Word-boundary fix (Correction des limites de mots)

A significant Track 1 finding was that substring-based keyword matching caused both false positives and false negatives:
- "morte de rire" (dying of laughter) matched "mort" (death) --- **false positive**
- "une personne formidable" (a wonderful person) matched "personne" (nobody) --- **false positive**
- "mortel" (awesome, slang) matched "mort" (death) --- **false positive**

We replaced all `if word in text` checks with regex `\b` word-boundary matching and added context-aware handling for ambiguous words like "personne" (which means "nobody" in isolation but "a person" when preceded by an article). We also added feminine forms to all lexicons (e.g., "epuisee", "abandonnee", "desesperee").

---

## 5. Track 2: Logic Hardening (Durcissement logique)

### 6-gate safety logic (Logique de securite a 6 portes)

The classifier uses a 6-gate decision pipeline that layers ML predictions with deterministic safety rules:

```
Gate 1: < 3 user messages?         --> Return Green (insufficient data)
Gate 2: Safety keyword floor       --> Crisis words force Orange/Red minimum
Gate 3: ML prediction              --> GradientBoosting on 67D feature vector
Gate 4: Low confidence (< 45%)?    --> Default to Yellow (cautious)
Gate 5: Short conversation (< 6)?  --> Cap at Orange maximum
Gate 6: Safety floor enforcement   --> ML cannot go below keyword level
```

**Rationale:** Gates 2 and 6 ensure that explicit crisis language *always* triggers an appropriate response, regardless of what the ML model predicts. Gate 4 ensures that uncertain predictions err on the side of caution. Gate 5 prevents premature Red alerts when there is insufficient conversational context.

### Feature engineering (Ingenierie des features)

We evolved from 7 basic features to 67 clinically motivated features across four layers:

#### Layer 1: Per-message lexical features (10 features)

| Feature | Clinical rationale |
|---------|--------------------|
| `word_count` | Shortening messages indicate withdrawal and disengagement |
| `punctuation_ratio` | Changes in punctuation reflect emotional state shifts |
| `question_presence` | Loss of questions indicates loss of engagement and curiosity |
| `negative_score` | Rising negativity across bilingual (FR+EN) lexicon |
| `finality_score` | Finality/ending language ("plus rien", "end it all") |
| `hope_score` | Declining hope words signal deterioration |
| `length_delta` | Message-to-message length changes track behavioral shifts |
| `negation_score` | Negated positive states ("can't cope", "ne suis pas bien") |
| `identity_conflict_score` | 2SLGBTQ+/cultural identity distress signals |
| `somatization_score` | Physical + emotional word co-occurrence (cultural expression) |

#### Layer 2: Trajectory statistics (60 features)

Each of the 10 per-message features is aggregated over the full conversation using 6 statistics: **mean, standard deviation, slope, last value, maximum, and minimum**. This captures the *trajectory* --- not just the current state, but the direction and volatility of emotional drift.

The slope statistic is particularly important: `finality_score_slope` captures whether finality language is *increasing* over time, while `hope_score_slope` captures whether hope is *declining*.

#### Layer 3: Semantic embedding features (4 features)

Using `paraphrase-multilingual-MiniLM-L12-v2` (a multilingual sentence transformer), we compute:

| Feature | What it measures |
|---------|-----------------|
| `embedding_drift` | Mean cosine distance between consecutive messages --- semantic wandering |
| `crisis_similarity` | Cosine similarity of the last message to a crisis language centroid |
| `embedding_slope` | PCA-reduced 1D slope --- directional semantic drift over time |
| `embedding_variance` | Mean pairwise cosine distance --- overall conversation coherence |

These features catch distress expressed through synonyms, paraphrases, and indirect language that lexical features alone would miss. The multilingual model handles both French and English natively.

#### Layer 4: Behavioral coherence features (3 features)

| Feature | What it measures |
|---------|-----------------|
| `short_response_ratio` | Fraction of user messages with fewer than 5 words (disengagement) |
| `min_topic_coherence` | Minimum cosine similarity between consecutive messages (topic jumping) |
| `question_response_ratio` | Fraction of assistant questions followed by a responsive reply |

These features detect *behavioral withdrawal* --- the pattern where a distressed user stops engaging meaningfully with the chatbot, giving short or tangential replies.

### Negation handling (Gestion des negations)

A critical logic hardening improvement: detecting negated positive states. "I'm fine" and "I'm not fine" have opposite meanings, but both contain positive words. We added regex-based negation patterns for both French and English:

- French: `ne ... pas bien`, `ne ... plus capable`, `ne ... jamais`
- English: `can't cope`, `not okay`, `never going to be`

### Word-boundary precision (Precision des limites de mots)

Replaced all substring matching (`if word in text`) with regex word-boundary matching (`\b`), plus context-aware rules:
- "personne" preceded by a French article ("une", "la", "cette") is not a finality word
- Feminine forms added to all lexicons for gender-inclusive detection
- Prevents false positives from compound words ("mortel", "immortel")

---

## 6. Track 3: Synthetic Data Augmentation (Augmentation de donnees synthetiques)

### Data generation pipeline (Pipeline de generation de donnees)

All training data is **100% synthetic** --- no real PII, per hackathon rules. We used the Claude Haiku API (`generate_synthetic_data.py`) to create realistic youth mental health conversations.

### Standard archetypes (Archetypes standards): 480 conversations

- **4 classes**: Green (normal), Yellow (concerning), Orange (significant distress), Red (crisis)
- **60 conversations per class per language** = 60 x 4 x 2 = 480
- **Bilingual**: authentic Quebecois French + Canadian English
- **12 user messages + 12 assistant messages** per conversation

### Adversarial archetypes (Archetypes adversariaux): 120 conversations

We identified 6 adversarial patterns that standard training data fails to cover:

| Archetype | What it captures | Why standard data misses it |
|-----------|-----------------|----------------------------|
| `physical_only` | Distress expressed purely through physical symptoms | Standard data uses emotional vocabulary |
| `sarcasm_distress` | Sarcastic or ironic crisis language | Standard data is direct |
| `adversarial_bypass` | Deliberate attempts to evade detection | Standard data is cooperative |
| `identity_distress` | 2SLGBTQ+ and cultural identity crisis | Standard data uses generic scenarios |
| `neurodivergent_flat` | Flat affect that is not distress | Standard data has clear emotional arcs |
| `crisis_with_deflection` | Crisis signals immediately followed by deflection | Standard data is consistent |

Adding these 120 adversarial conversations:
- Improved CV variance from +-4.4% to +-1.5% (more stable model)
- Improved sample:feature ratio from 7.2:1 to 9.0:1
- Enabled 30/30 adversarial test pass rate (up from 13/13 on fewer tests)

### Quality annotation (Annotation qualite)

We built a Claude-based quality annotation pipeline (`annotate_data.py`) that scores each synthetic conversation on realism, label accuracy, and clinical plausibility. While filtering low-quality conversations hurt accuracy (removing edge cases the model needed to learn from), the annotation insights informed which adversarial archetypes to add.

### Bilingual integrity (Integrite bilingue)

Every conversation is generated in its target language with authentic regional expressions:
- **French**: Quebecois vocabulary, informal register (tutoiement), regional expressions
- **English**: Canadian context (university, CEGEP references, Canadian cultural elements)
- **No translation artifacts**: conversations are generated natively in each language, not translated

---

## 7. Architecture

### System architecture (Architecture du systeme)

```
+---------------------------------------------------------------------+
|                        CEDD Safety Layer                             |
|                                                                      |
|  +------------------+    +------------------+    +----------------+  |
|  | Feature Extractor|    |    Classifier    |    |   Response     |  |
|  |                  |--->|                  |--->|   Modulator    |  |
|  | 10 features/msg  |    | 6-gate safety    |    |                |  |
|  | 67 trajectory    |    | GradientBoosting |    | Adaptive       |  |
|  | Bilingual NLP    |    | StandardScaler   |    | system prompts |  |
|  | Embeddings       |    |                  |    | LLM fallback   |  |
|  +------------------+    +------------------+    +----------------+  |
|                                                         |            |
|  +------------------+                           +-------v--------+   |
|  | Session Tracker  |                           | LLM Chain      |   |
|  | SQLite           |                           | Cohere         |   |
|  | Cross-session    |                           | Groq           |   |
|  | Withdrawal       |                           | Gemini         |   |
|  | detection        |                           | Claude         |   |
|  +------------------+                           | Static text    |   |
|                                                  +----------------+  |
+---------------------------------------------------------------------+
                              |
                    +---------v---------+
                    |   Streamlit UI    |
                    |   Bilingual       |
                    |   FR / EN toggle  |
                    +-------------------+
```

### ML pipeline (Pipeline ML)

- **Scaler**: `StandardScaler` --- normalizes 67 features to mean=0, std=1
- **Model**: `GradientBoostingClassifier` --- 200 trees, max_depth=3, learning_rate=0.1
- **Validation**: `StratifiedKFold` with k=4
- **Serialization**: `joblib` format (`models/cedd_model.joblib`)

### LLM fallback chain (Chaine de repli LLM)

To ensure 24/7 availability for a crisis service, CEDD uses a 5-step fallback chain:

1. **Cohere** --- primary
2. **Groq** (Llama 3.3 70B Versatile) --- fastest inference
3. **Google Gemini** (2.5 Flash) --- tertiary
4. **Anthropic Claude** (Haiku) --- quaternary
5. **Static emergency text** --- guaranteed availability with crisis resources

If all LLM providers are down, the static fallback still provides Kids Help Phone contact information. No youth in crisis should ever see a blank screen.

### Alert levels (Niveaux d'alerte)

| Level | Color | Description | LLM behavior |
|-------|-------|-------------|-------------|
| 0 | Green | Normal conversation | Supportive, open-ended questions |
| 1 | Yellow | Concerning signs (fatigue, loneliness) | Enhanced emotional validation |
| 2 | Orange | Significant distress | Active support + crisis resources mentioned |
| 3 | Red | Potential crisis | Urgent referral, KHP resources prominently displayed |

### Warm handoff (Transfert accompagne)

At Red alert, CEDD replaces the industry-standard "cold referral" (display a phone number, end conversation) with a **5-step accompanied transition**:

1. **Empathetic acknowledgment** --- validate feelings, no hotline number yet
2. **Permission-based transition** --- ask consent, frame as "upgrade" not rejection
3. **Context bridge** --- generate anonymized summary for KHP responder
4. **Seamless connection** --- text-based (686868), same modality, no story repetition
5. **Background monitoring + follow-up** --- CEDD stays active, acknowledges returning users

The simulated counselor "Alex" uses an ASIST-trained (Applied Suicide Intervention Skills Training) persona with short responses (2-4 sentences), one question at a time, and consistent emotional validation before any questions.

### Cross-session tracking (Suivi inter-sessions)

SQLite-based longitudinal tracking records:
- Session metadata (user, timestamps, max alert level, message count)
- Alert events (level, confidence, trigger message)
- Handoff events (step reached, user response)
- Last activity timestamps (for withdrawal detection)

Users returning after >24 hours without closing their session trigger a withdrawal risk check and receive a welcome-back message.

---

## 8. Results and Metrics (Resultats et metriques)

### Improvement trajectory (Trajectoire d'amelioration)

| Stage | CV Accuracy | Features | Training Data | Adversarial Tests |
|-------|-------------|----------|---------------|-------------------|
| Baseline (March 10) | 66.7% +-26.4% | 42 (7x6) | 24 conversations | 7/10 |
| Data expansion | 91.2% +-1.5% | 48 (8x6) | 320 conversations | 9/10 |
| +Negation +Embeddings | 92.2% +-1.8% | 52 | 320 conversations | 9/10 |
| +Identity +Somatization +Coherence | 92.5% +-1.5% | 67 | 320 conversations | 13/13 |
| Data expansion to 480 | 91.7% +-4.4% | 67 | 480 conversations | 13/13 |
| Adversarial augmentation to 600 | 90.5% +-1.5% | 67 | 600 conversations | 30/30 |
| **Word-boundary fix (current)** | **90.0% +-1.6%** | **67** | **600 conversations** | **36/36** |

### Current metrics (Metriques actuelles)

| Metric | Value |
|--------|-------|
| Cross-validated accuracy (k=4, stratified) | **90.0% +-1.6%** |
| Feature count | **67** (60 trajectory + 4 embedding + 3 coherence) |
| Training conversations | **600** (480 standard + 120 adversarial) |
| Languages | **2** (French + English, 300 each) |
| Adversarial tests | **36/36 passing, 0 critical misses** |
| Adversarial test categories | **20** |
| Sample:feature ratio | **9.0:1** (target: 10:1) |
| Train accuracy | ~100% (expected with 200 trees on 600 samples) |

### Top features by model importance (Features dominantes)

| Rank | Feature | Importance | What it measures |
|------|---------|------------|-----------------|
| 1 | `word_count_max` | 0.192 | Longest message in conversation |
| 2 | `word_count_slope` | 0.179 | Whether messages are getting shorter over time |
| 3 | `word_count_last` | 0.138 | Length of the most recent message |
| 4 | `length_delta_mean` | 0.075 | Average change in message length |

The dominance of word-count-related features aligns with clinical research: **progressive shortening of messages is one of the strongest behavioral signals of emotional withdrawal and disengagement**. This is a signal that per-message classifiers cannot detect --- only trajectory analysis reveals it.

### Key insight (Constat cle)

The accuracy dropped slightly from 92.5% to 90.0% as we added adversarial training data. This is expected and *desirable*: the model traded a small amount of overall accuracy for significantly better robustness on adversarial edge cases. The CV variance also stabilized dramatically (from +-26.4% to +-1.6%), indicating a much more reliable model.

---

## 9. User Experience (Experience utilisateur)

### Bilingual interface (Interface bilingue)

The Streamlit application supports full French/English toggle, with all UI elements, system prompts, feature labels, and crisis resources available in both languages.

### Key UX features (Fonctionnalites UX principales)

| Feature | Description |
|---------|-------------|
| **Welcome card** | Branded HTML card with brain emoji, bilingual title/description, call to action |
| **Demo profiles** | 5 selectable profiles (team members + Guest) with distinct longitudinal trajectories: stable green, gradual improvement, fluctuating, escalating, and fresh start |
| **Feature importance chart** | Collapsible Plotly horizontal bar chart showing top 5 features by composite score (model importance x scaled value), with 6 color categories and bilingual labels |
| **Feature radar chart** | Plotly Scatterpolar showing 10 per-message features normalized 0-1, with current message in alert-level color and first message as green ghost overlay |
| **Side-by-side compare mode** | Split view: raw LLM response (no system prompt) vs CEDD-modulated response. Demonstrates the value of adaptive prompting on crisis messages |
| **Demo autopilot** | Auto-plays a 9-message scenario (Felix in FR, Alex in EN) showing Green-to-Yellow-to-Orange drift. Judges can observe the full trajectory hands-free |
| **Alert transition animation** | CSS-animated toast notification when alert level increases |
| **Chat timestamps** | HH:MM timestamps below each message bubble |
| **LLM source badge** | Colored badge on each assistant message showing which LLM generated it |
| **Alert level badge** | Colored dot on each assistant message showing the CEDD classification |
| **Export transcript** | Download conversation + alert history as JSON |
| **Simulated counselor handoff** | At Red level, interactive handoff to "Alex" (KHP counselor persona) with blue gradient UI, distinct avatar, and ASIST-trained responses |
| **About panel** | Collapsible bilingual explanation of CEDD's operation |

### Competitive UX audit findings (Resultats de l'audit UX concurrentiel)

Amanda Wu audited 6 platforms (ChatGPT, Gemini, Character.AI, Wysa, Woebot) across desirability, usability, and accessibility. **No platform currently offers all of:**

- Canadian-specific crisis resources (Kids Help Phone) --- **CEDD does**
- French-language crisis detection --- **CEDD does**
- Subtle/coded distress detection --- **CEDD's trajectory analysis does**
- Warm handoff to human responder --- **CEDD does** (5-step flow)
- Cross-session memory --- **CEDD does** (SQLite longitudinal tracking)

---

## 10. Canadian Multicultural Context (Contexte multiculturel canadien)

Canada is the most diverse G7 country. Crisis detection trained only on Western English expressions will systematically fail the most vulnerable youth. CEDD addresses this with culturally informed feature engineering:

| Cultural Group | Expression Pattern | CEDD Detection |
|---------------|-------------------|----------------|
| **Indigenous** | Storytelling, substance references, holistic/spiritual framing | Trajectory features detect drift patterns regardless of vocabulary |
| **South Asian** | Somatization --- "my chest hurts" = emotional pain | `somatization_score` detects physical + emotional word co-occurrence |
| **East Asian** | Withdrawal, silence, minimizing ("I'm fine") | `short_response_ratio` + `question_response_ratio` catch disengagement |
| **Francophone** | French-language distress, Quebecois dialect | Native FR lexicons + bilingual embeddings |
| **2SLGBTQ+** | Identity conflict, coded rejection language | `identity_conflict_score` with dedicated phrase lists |
| **Neurodivergent** | Literal language, shutdown, emotional bursts | Temporal trajectory distinguishes from sustained crisis |

### Why this matters (Pourquoi c'est important)

With 46% of Kids Help Phone youth identifying as 2SLGBTQ+ and 10% identifying as Indigenous, culturally insensitive detection is not just a technical limitation --- it is a safety failure that disproportionately affects the most at-risk populations.

---

## 11. Competitive Analysis (Analyse concurrentielle)

### CEDD vs EmoAgent (Princeton/Michigan, arXiv:2504.09689)

EmoAgent is the closest academic reference point. It uses a multi-agent architecture with GPT-4o for both detection and response.

| Dimension | CEDD | EmoAgent |
|-----------|------|----------|
| **Detection speed** | ~0ms (feature extraction + ML) | Multiple GPT-4o calls per message |
| **Detection cost** | $0 (local computation) | ~$0.04-0.10 per message |
| **Explainability** | Full (67 named features, composite scores, importance chart) | Black box (LLM reasoning) |
| **Bilingual** | Native FR + EN (lexicons, embeddings, prompts) | English only |
| **Cross-session** | SQLite longitudinal tracking + withdrawal detection | Per-conversation only |
| **Cultural sensitivity** | Somatization, identity conflict, coherence features | None |
| **Clinical tools** | 4-level alerts + 6-gate safety logic + warm handoff | PHQ-9, PDI, PANSS (validated scales) |
| **Availability** | 5-model LLM fallback chain + static emergency text | Single provider dependency |

**Our positioning:** "EmoAgent needs 4 GPT-4o calls per message. CEDD detects in ~0ms with 67 features and GradientBoosting, and only calls the LLM to modulate the response. 5-model fallback chain ensures availability. It is the difference between an IDS that deep-inspects all traffic and a lightweight edge firewall --- and ours works in both French and English."

### What EmoAgent does better (Ce qu'EmoAgent fait mieux)

EmoAgent uses clinically validated instruments (PHQ-9, PDI, PANSS) as evaluation metrics. CEDD's thresholds are engineering-derived, not clinically validated. Borrowing simplified validated scales as complementary metrics is a planned improvement.

---

## 12. Limitations and Future Work (Limites et travaux futurs)

### Current limitations (Limites actuelles)

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| No clinical validation | Alert thresholds are engineering-derived | 6-gate safety logic provides deterministic safety floor |
| ML unreliable for short conversations (< 6 messages) | Cannot build trajectory from insufficient data | Gate 5 caps at Orange maximum |
| Sarcasm and periphrasis remain challenging | "je pese sur tout le monde" not reliably detected | Embeddings complement lexical features but do not fully solve this |
| Sample:feature ratio at 9.0:1 (ideal is 10:1) | Slight risk of overfitting | Adversarial augmentation improved from 7.2:1 |
| Identity conflict detection is phrase-based | May miss coded or indirect identity distress | Embeddings provide partial coverage |
| Over-prediction on short/ambiguous green conversations | Physical-only or neurodivergent patterns may trigger Orange | Documented as acceptable (over-alerting preferred to under-alerting) |
| Train accuracy ~100% | Indicates model memorization | CV accuracy (90.0%) shows generalization is reasonable |

### Future work (Travaux futurs)

| Improvement | Expected impact | Effort |
|-------------|----------------|--------|
| LSTM/Transformer sequence model | +10-15% accuracy (learns message ordering natively) | 3-4 hrs |
| Minimization detection | Cross-reference "I'm fine" with behavioral signals | 1-2 hrs |
| Burst vs sustained temporal patterns | Better neurodivergent/ADHD handling | 2-3 hrs |
| Clinical validation study | Validated thresholds from mental health professionals | Weeks |
| Intra-session timing analysis | Detect pauses and response delays as signals | 1-2 hrs |

---

## 13. Conclusion

CEDD demonstrates that effective youth mental health crisis detection does not require expensive multi-agent LLM architectures. By combining trajectory-based feature engineering (67 features across lexical, embedding, and behavioral layers), deterministic safety gates, and adaptive LLM response modulation, CEDD achieves:

- **Reliable detection**: 90.0% +-1.6% cross-validated accuracy on 600 bilingual conversations
- **Adversarial robustness**: 36/36 test cases passing across 20 attack categories, 0 critical misses
- **Explainability**: every alert is traceable to specific named features and their scores
- **Cultural sensitivity**: somatization, identity conflict, and withdrawal detection for Canada's diverse youth population
- **Operational resilience**: 5-model LLM fallback chain with static emergency text guarantee
- **Humane crisis response**: 5-step warm handoff that accompanies youth to help rather than abandoning them with a phone number

The improvement trajectory tells the story: from 66.7% accuracy with 42 features and 24 conversations, to 90.0% accuracy with 67 features and 600 conversations. From 7/10 adversarial tests to 36/36. Every iteration was guided by the principle that in a youth mental health application, missing a crisis is the one failure mode that is never acceptable.

CEDD is not a replacement for human counselors. It is the safety layer that ensures no youth in crisis falls through the cracks of an AI conversation --- and that when crisis is detected, the transition to human support is seamless, respectful, and accompanied.

---

## 14. References and Resources (References et ressources)

### Emergency resources (Ressources d'urgence)

- **Kids Help Phone / Jeunesse, J'ecoute**: 1-800-668-6868 (24/7, free, confidential) --- text: 686868
- **Suicide Crisis Helpline / Ligne de crise suicide**: 9-8-8 (988.ca)
- **Multi-Ecoute**: 514-378-3430 (multiecoute.org)
- **Tracom**: 514-483-3033 (tracom.ca)
- **Emergency services / Services d'urgence**: 911

### Academic references (References academiques)

- EmoAgent: Multi-Agent Framework for Emotional Support (Princeton/Michigan, arXiv:2504.09689)
- ASIST (Applied Suicide Intervention Skills Training) --- LivingWorks
- Kids Help Phone Annual Report 2024 --- Service statistics and demographics
- JMIR (May 2025) --- Youth communication preferences in mental health contexts

### Technical references (References techniques)

- `paraphrase-multilingual-MiniLM-L12-v2` --- Sentence-Transformers (Reimers & Gurevych, 2019)
- `GradientBoostingClassifier` --- scikit-learn (Pedregosa et al., 2011)
- Streamlit --- open-source app framework for ML

---

*Report prepared by Team 404HarmNotFound for the Mila x Bell x Kids Help Phone Hackathon, March 2026.*
*All training data is 100% synthetic --- no real PII was used at any stage.*
