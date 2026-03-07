Step 1 — The Big Picture: What Problem Does This App Solve?
Before touching a single line of code, you need to understand why this app exists.

The Problem
Imagine a young person (16–22 years old) who is slowly sliding into a mental health crisis. They don't say "I want to die" on day one. Instead, over days or weeks, subtle things change in how they write:


Day 1:  "Hey! Had a rough week but I'm hanging in there. 
         How do people deal with exam stress? 😅"
         → Long message, question, future-oriented, emoji

Day 4:  "Still tired. Don't really feel like talking much."
         → Shorter, no question, passive

Day 7:  "whatever"
         → One word. No punctuation. No question. No future.
A normal chatbot sees each message in isolation. It responds to "whatever" without knowing this person was fine 6 days ago. It's blind to the drift.

The Solution: CEDD
CEDD adds a surveillance layer that watches the trajectory — how messages evolve over time — not just what the current message says.

┌──────────────────────────────────────────────────┐
│                   CEDD System                    │
│                                                  │
│  Message 1 → Message 2 → ... → Message N         │
│       ↓           ↓                  ↓           │
│  [measure]    [measure]         [measure]        │
│       └───────────┴──────────────┘               │
│                   ↓                              │
│          How is the TREND going?                 │
│                   ↓                              │
│   🟢 Green → 🟡 Yellow → 🟠 Orange → 🔴 Red      │
└──────────────────────────────────────────────────┘
4 alert levels:

Green → All good, normal conversation
Yellow → Concerning signs, worth watching
Orange → Significant distress, needs active support + resources
Red → Potential crisis, urgent referral
The Tech Stack (tools used)

┌─────────────────────────────────────────┐
│  What you see:  Streamlit (web app)     │
│  Brain:         scikit-learn (ML model) │
│  Memory:        SQLite (database)       │
│  Chatbot:       Claude / Ollama (LLM)   │
│  Language:      Python                  │
└─────────────────────────────────────────┘

Let me continue with the file structure - showing them how the project is organized, then we'll go deeper into each component one by one.

Let me follow baby-step teaching:

Step 1: Big picture ✅
Step 2: File structure / project organization
Step 3: The data (what data looks like)
Step 4: Feature extraction (feature_extractor.py) - what are features?
Step 5: The ML algorithm (classifier.py) - GradientBoosting
Step 6: Training (train.py)
Step 7: The LLM response (response_modulator.py)
Step 8: Session tracking (session_tracker.py)
Step 9: The app (app.py)
Step 10: How it all connects

Step 2 — The File Map: What Does Each File Do?
Think of the project like a factory. Each file is a different room in that factory, with a specific job.

cedd-hackathon/
│
├── 📊 DATA ROOM
│   └── data/synthetic_conversations.json  ← training examples
│
├── 🏋️ TRAINING ROOM
│   ├── generate_synthetic_data.py  ← creates training examples via Claude API
│   ├── train.py                    ← teaches the ML model
│   └── simulate_history.py         ← fills fake history for demo
│
├── 🧠 BRAIN ROOM  (the cedd/ package)
│   ├── feature_extractor.py  ← reads messages, measures 7 things per message
│   ├── classifier.py         ← uses those measurements to predict alert level
│   ├── response_modulator.py ← chooses what the chatbot should say
│   └── session_tracker.py    ← remembers past sessions in a database
│
├── 🖥️ DISPLAY ROOM
│   └── app.py                ← the web interface (what users see)
│
└── 💾 SAVED MODEL
    └── models/cedd_model.joblib  ← the trained ML model, saved to disk
The two phases of this system
There are two completely separate moments in time:

PHASE 1: TRAINING (done once, offline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
generate_synthetic_data.py  →  data/synthetic_conversations.json
                                         ↓
                              train.py  teaches the model
                                         ↓
                              models/cedd_model.joblib  (saved)


PHASE 2: LIVE APP (runs every time a user chats)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.py loads the saved model
         ↓
user sends a message
         ↓
feature_extractor.py  →  measures the message
         ↓
classifier.py  →  predicts alert level  (uses the trained model)
         ↓
response_modulator.py  →  tells the LLM how to respond
         ↓
session_tracker.py  →  saves everything to the database
The key insight: you train once, you use forever (or until you retrain with better data).


Step 3 — The Training Data: Teaching the Model With Examples
Before a machine can learn, it needs examples with answers. This is the foundation of all supervised machine learning.

What is a "labeled example"?
Imagine you want to teach a child to recognize cats vs dogs. You show them 1000 photos, and for each photo you say:

"This is a cat" ← the label
"This is a dog" ← the label
After enough examples, the child can recognize new photos it has never seen.

CEDD does the same thing, but with conversations instead of photos.

What does one training example look like?
Open data/synthetic_conversations.json. Each entry looks like this:


{
  "id":         "synth_fr_verte_001",
  "label":      0,               ← the ANSWER (0=green, 1=yellow, 2=orange, 3=red)
  "label_name": "verte",         ← human-readable version of the answer
  "language":   "fr",
  "messages": [
    {"role": "user",      "content": "Hey! J'ai eu une bonne journée..."},
    {"role": "assistant", "content": "C'est super! Qu'est-ce qui s'est passé?"},
    {"role": "user",      "content": "J'ai passé du temps avec mes amis..."},
    {"role": "assistant", "content": "..."},
    ...  (12 user messages + 12 assistant messages = 24 total)
  ]
}
The label is what the model must learn to predict. It's the "correct answer" for each training conversation.

How many examples do we have?

Total: 120 conversations

┌─────────┬──────────────────────────────────────────┬───────┐
│  Label  │  What it represents                      │ Count │
├─────────┼──────────────────────────────────────────┼───────┤
│ 0 Green │  Youth doing fine                        │  30   │
│ 1 Yellow│  Some concerning signs                   │  30   │
│ 2 Orange│  Significant distress                    │  30   │
│ 3 Red   │  Crisis, finality thoughts               │  30   │
└─────────┴──────────────────────────────────────────┴───────┘
         ← Perfectly balanced (30 per class)
A balanced dataset is important — if we had 100 green and 5 red, the model would just learn to always say "green" and be right 95% of the time, which is useless.

Who created this data?
Humans can't write 120 realistic crisis conversations by hand (ethically and practically). So we used Claude AI to generate them via generate_synthetic_data.py:


You write the script:
  "Generate a realistic conversation of a 17-year-old
   who is gradually drifting toward crisis (class red).
   12 exchanges. Authentic language. Gradual drift."

Claude writes the conversation. You label it "3 = red".
Repeat 120 times.
This is called synthetic data generation — fake-but-realistic examples made by an AI.

The key concept: supervised learning

SUPERVISED LEARNING = learning from labeled examples

Input (X)            Label (Y)
─────────────────    ─────────
conversation_001  →  0 (green)
conversation_002  →  1 (yellow)
conversation_003  →  3 (red)
...
conversation_120  →  2 (orange)

The model learns:
"When I see THIS pattern of messages → predict THIS label"

What is a "feature" in ML?
The 7 features per message
The code in feature_extractor.py
How features are computed
Let me make this very visual and concrete.

Step 4 — Feature Extraction: Turning Words Into Numbers
This is the most important concept in the entire system. Machine learning cannot read text directly — it only understands numbers. So we must convert each message into a row of numbers.

What is a "feature"?
A feature is one measurable property of a message. Think of it like a doctor measuring a patient:

Patient → Doctor measures:
  height:      175 cm
  weight:      70 kg
  temperature: 37.2°C
  heart rate:  72 bpm

Message → CEDD measures:
  word_count:        12 words
  punctuation_ratio: 0.02  (2%)
  question_presence: 1.0   (yes, has a ?)
  negative_score:    0.25
  finality_score:    0.0
  hope_score:        0.08
  length_delta:      -0.4  (shorter than previous)
The doctor uses numbers to diagnose. CEDD uses numbers to classify.

The 7 features — explained simply
┌──────────────────────┬────────────────────────────────────────────────────┐
│ Feature              │ What it measures & why it matters                  │
├──────────────────────┼────────────────────────────────────────────────────┤
│ word_count           │ How many words in the message.                     │
│                      │ Crisis → messages get shorter and shorter          │
├──────────────────────┼────────────────────────────────────────────────────┤
│ punctuation_ratio    │ Punctuation chars ÷ total chars.                   │
│                      │ Depressed → fewer commas, periods, exclamations    │
├──────────────────────┼────────────────────────────────────────────────────┤
│ question_presence    │ 1.0 if message has "?", else 0.0                   │
│                      │ Questions = engagement with the future.            │
│                      │ Crisis → questions disappear                       │
├──────────────────────┼────────────────────────────────────────────────────┤
│ negative_score       │ Ratio of negative words in the message.            │
│                      │ "sad", "alone", "pain", "empty", "crying"…         │
├──────────────────────┼────────────────────────────────────────────────────┤
│ finality_score       │ Ratio of "finality" words.                         │
│                      │ "end it", "disappear", "what's the point"…         │
├──────────────────────┼────────────────────────────────────────────────────┤
│ hope_score           │ Ratio of hope/future words.                        │
│                      │ "tomorrow", "try", "family", "maybe"…              │
├──────────────────────┼────────────────────────────────────────────────────┤
│ length_delta         │ Relative change vs previous message.               │
│                      │ -0.5 = "this message is 50% shorter than last"    │
└──────────────────────┴────────────────────────────────────────────────────┘
Let's trace one real example through the code
Take this message: "I feel so empty and alone. what's the point."

Step 1 — word_count — feature_extractor.py:87


def _count_words(text):
    return len(text.split())

"I feel so empty and alone. what's the point.".split()
# → ["I", "feel", "so", "empty", "and", "alone.", "what's", "the", "point."]
# → 9 words
Step 2 — question_presence — feature_extractor.py:102


def _has_question(text):
    return 1.0 if '?' in text else 0.0

# "what's the point." → no '?' → 0.0
# "what's the point?" → has '?' → 1.0
Step 3 — negative_score — feature_extractor.py:110


# NEGATIVE_WORDS includes: "empty", "alone", "feel bad"...
# Found: "empty" ✓, "alone" ✓ → count = 2
# score = 2 / 9 words = 0.22
Step 4 — finality_score — feature_extractor.py:138

# FINALITY_WORDS includes: "what's the point"...
# Found: "what's the point" ✓ → count = 1
# score = 1 / 9 = 0.11
Result — one row of 7 numbers:

[9,    0.04,  0.0,   0.22,  0.11,  0.0,   -0.3]
 ^     ^      ^      ^      ^      ^       ^
 words punct  quest  neg    final  hope    delta
The full picture: one conversation → a table of numbers

             word  punct  quest  neg   final  hope  delta
             count ratio  pres   score score  score
             ─────────────────────────────────────────────
Message 1:  [ 25,  0.06,  1.0,  0.04,  0.0,  0.12, 0.0  ]
Message 2:  [ 20,  0.05,  1.0,  0.08,  0.0,  0.08, -0.2 ]
Message 3:  [ 15,  0.03,  0.0,  0.15,  0.0,  0.04, -0.25]
Message 4:  [ 10,  0.02,  0.0,  0.20,  0.05, 0.02, -0.33]
Message 5:  [  8,  0.01,  0.0,  0.25,  0.10, 0.0,  -0.2 ]
...
Message 12: [  4,  0.0,   0.0,  0.40,  0.25, 0.0,  -0.5 ]
            ─────────────────────────────────────────────
            ↑ each row = one user message
            ↑ each column = one of the 7 features
You can literally see the drift — word count falling, negative score rising, hope disappearing, questions gone.


Step 5: Trajectory Features — From a Table to One Row of 42 Numbers

Step 5 — Trajectory Features: Collapsing the Table Into One Row
Here's the problem we need to solve. Look at what we have after Step 4:

Each conversation gives us a TABLE (12 rows × 7 columns).
But the ML model needs ONE ROW of fixed-size numbers.

How do you summarize a whole table into one row?
→ You compute statistics about each column!
The 6 statistics computed per feature
For each of the 7 features, we compute 6 statistics that describe its behavior over time:

Take the "word_count" column across all 12 messages:

word_count = [25, 22, 18, 20, 15, 12, 10, 9, 8, 6, 5, 4]
              ↑                                          ↑
           msg 1                                    msg 12

Compute these 6 statistics:
┌──────────┬────────────────────────────────────────────────────────┐
│ Statistic│ Value  │ What it tells you                             │
├──────────┼────────┼───────────────────────────────────────────────┤
│ mean     │  11.2  │ Average length across all messages            │
│ std      │   6.8  │ How variable (inconsistent) the lengths are   │
│ slope    │  -2.1  │ Trend: negative = getting shorter over time   │
│ last     │   4.0  │ Length of the LAST (most recent) message      │
│ max      │  25.0  │ Longest message in the conversation           │
│ min      │   4.0  │ Shortest message in the conversation          │
└──────────┴────────┴───────────────────────────────────────────────┘
Why slope matters most: A slope of -2.1 means "each message is about 2 words shorter than the previous". That's the drift signal. A stable person might have slope ≈ 0. Someone in crisis trends strongly negative.

The math: 7 features × 6 stats = 42 numbers

word_count    → [mean, std, slope, last, max, min]  → 6 numbers
punct_ratio   → [mean, std, slope, last, max, min]  → 6 numbers
quest_pres    → [mean, std, slope, last, max, min]  → 6 numbers
negative_score→ [mean, std, slope, last, max, min]  → 6 numbers
finality_score→ [mean, std, slope, last, max, min]  → 6 numbers
hope_score    → [mean, std, slope, last, max, min]  → 6 numbers
length_delta  → [mean, std, slope, last, max, min]  → 6 numbers
                                                    ───────────
                                         TOTAL  =   42 numbers
This is the 42-dimensional vector that goes into the ML model. Every conversation — no matter how many messages — becomes exactly 42 numbers. That's the key.

The code that does this — feature_extractor.py:231

def extract_trajectory_features(features_array):
    for f in range(7):             # loop over each of the 7 features
        col = features_array[:, f] # grab one column (all messages, one feature)

        mean_val = float(np.mean(col))   # average
        std_val  = float(np.std(col))    # standard deviation
        last_val = float(col[-1])        # last message value
        max_val  = float(np.max(col))    # max
        min_val  = float(np.min(col))    # min

        # Slope = linear regression on that column over time
        slope = float(np.polyfit(x_norm, col, 1)[0])

        trajectory_feats.extend([mean_val, std_val, slope,
                                  last_val, max_val, min_val])

    return np.array(trajectory_feats)  # returns 42 numbers
The full pipeline so far, visualized:

1 conversation (12 messages)
         ↓
extract_features()
         ↓
Table: 12 rows × 7 columns
         ↓
extract_trajectory_features()
         ↓
1 row of 42 numbers  ← THIS goes into the ML model
[11.2, 6.8, -2.1, 4.0, 25.0, 4.0,  ← word_count stats
  0.03, 0.01, -0.004, 0.0, 0.06, 0.0, ← punct stats
  ...]

What is a Decision Tree (the building block)
What is a Random Forest (many trees)
What is Gradient Boosting (sequential trees, each correcting the previous)
The sklearn Pipeline (StandardScaler + GradientBoostingClassifier)
What "training" actually means
What "prediction" returns (probabilities)
Baby steps - start with Decision Tree as the foundation.

Exactly right — 7 features × 6 stats = 42 numbers. Perfect.

Step 6 — The ML Algorithm: Gradient Boosting
This is the core of the system. We now have 42 numbers per conversation, and we need an algorithm that learns to say "these 42 numbers → class 2 (orange)".

I'll build up to it in 3 baby steps.

Baby Step 6a — The Decision Tree (the building block)
A decision tree is the simplest idea: ask a series of yes/no questions.


                    Is word_count_slope < -1.5?
                    (Are messages getting shorter fast?)
                   /                              \
                 YES                               NO
                  |                                |
      Is negative_score_mean > 0.2?     Is hope_score_mean > 0.1?
         /              \                  /               \
        YES              NO              YES                NO
         |               |               |                  |
       🔴 RED         🟠 ORANGE       🟢 GREEN           🟡 YELLOW
Each leaf at the bottom is a prediction. The tree learned these cut-off values (-1.5, 0.2, 0.1) automatically from the training data.

Problem with one tree: It's brittle. One bad split and everything is wrong.

Baby Step 6b — The Forest (many trees voting)
Instead of one tree, grow 200 trees, each trained slightly differently. Then take a vote:


Tree 1:   → RED
Tree 2:   → ORANGE
Tree 3:   → RED
Tree 4:   → RED
Tree 5:   → ORANGE
...
Tree 200: → RED

Vote result: RED=140, ORANGE=55, YELLOW=5, GREEN=0
→ Predict RED with 70% confidence
This is called a Random Forest — it's more robust. But CEDD uses something even smarter.

Baby Step 6c — Gradient Boosting (trees that fix each other's mistakes)
Gradient Boosting is the algorithm used in CEDD. Instead of all trees voting independently, each tree learns from the mistakes of the previous one:

Round 1: Tree 1 trains on all data
         → Makes some errors (confused Orange with Red)

Round 2: Tree 2 trains HARDER on the examples Tree 1 got WRONG
         → Focuses on the hard cases

Round 3: Tree 3 trains on what Tree 2 still gets wrong
         → Gets even more refined

...repeat 200 times...

Final answer = weighted sum of all 200 trees
Think of it like this: you hire 200 specialists, and each new one is specifically hired to fix the errors of all previous ones.

GRADIENT BOOSTING vs RANDOM FOREST:

Random Forest:   Tree1 + Tree2 + Tree3 ... (independent, vote at end)
                 ─────────────────────────
Gradient Boost:  Tree1 → fix errors → Tree2 → fix errors → Tree3 ...
                 (sequential, each improves the previous)
Gradient Boosting almost always outperforms Random Forest. That's why it's used here.

The code in CEDD — classifier.py:78

self.pipeline = Pipeline([
    ("scaler", StandardScaler()),              # Step 1: normalize
    ("clf",    GradientBoostingClassifier(     # Step 2: classify
        n_estimators=200,   ← 200 trees
        max_depth=3,        ← each tree max 3 levels deep
        random_state=42,    ← reproducible results
    )),
])
Why StandardScaler first? Our 42 features have wildly different scales:

word_count_mean might be 20 (words)
question_presence_mean might be 0.6 (between 0 and 1)
StandardScaler transforms everything to have mean=0 and std=1, so the model treats all features fairly.

Before scaling:    word_count=20,  question=0.6
After scaling:     word_count=0.3, question=-0.2
                   (both now on the same scale)

Actually, let me think about the flow. We've covered:

✅ Big picture
✅ File map
✅ Training data
✅ Feature extraction (7 features per message)
✅ Trajectory features (42 numbers)
✅ Gradient Boosting algorithm

Now I should cover:

7. The training script (train.py) - how training actually happens + cross-validation

The full classifier logic with safety rules (classifier.py get_alert_level)
Response modulator
Session tracker
The app (app.py)
Then write EXPLANATION_RESUME.md
Step 7 is crucial - I need to explain train.py and walk through cross-validation, overfitting prevention, and how we actually fit the model to our data.

Step 7 — Training: What Actually Happens in train.py
Now we connect everything: we have 120 labeled conversations → each becomes 42 numbers → the model learns from them.

What "training" actually means
When we call clf.fit(X, y), the GradientBoosting algorithm:

1. Takes all 120 rows of 42 features (X)
   and their labels (y = 0,1,2,3)

2. Builds Tree 1: tries to predict y from X
   → makes some errors

3. Builds Tree 2: focuses on the errors from Tree 1
   → fewer errors

4. Repeats 200 times

5. Freezes all 200 trees → saves to disk
   (this is the "trained model")
After training, the model has learned things like:

"When word_count_slope < -2.0 AND finality_score_last > 0.1 → probably Red"

Two important concepts: Overfitting & Cross-Validation
Overfitting is a critical DS concept. Imagine a student who memorizes all exam answers instead of understanding the subject:

OVERFITTING:
Training data accuracy:  100%  ← "memorized everything"
New data accuracy:        40%  ← "doesn't generalize"

GOOD MODEL:
Training data accuracy:   85%
New data accuracy:        78%  ← "actually learned the pattern"
CEDD's model gets 100% on training data — expected, because 120 examples is small. That's why we need cross-validation.

Cross-Validation (k=4, stratified) — train.py:84
Instead of testing on the same data we trained on, we do this:

Split 120 conversations into 4 equal groups (folds):

Fold A: 30 convos  |  Fold B: 30 convos  |  Fold C: 30 convos  |  Fold D: 30 convos

Round 1: Train on B+C+D (90 convos) → Test on A → accuracy: 72%
Round 2: Train on A+C+D (90 convos) → Test on B → accuracy: 80%
Round 3: Train on A+B+D (90 convos) → Test on C → accuracy: 76%
Round 4: Train on A+B+C (90 convos) → Test on D → accuracy: 75%
                                                    ─────────────
                              Average CV accuracy:   75.8% ± 6.0%
Each fold is used once as "unseen test data". This gives a realistic estimate of how the model performs on new conversations it has never seen.

"Stratified" means each fold keeps the same class balance (7-8 green, 7-8 yellow, 7-8 orange, 7-8 red) — so no fold accidentally gets all the easy examples.

The code flow in train.py

# 1. Load data and convert to 42-number rows
X, y = load_and_extract("data/synthetic_conversations.json")
# X shape: (120, 42)    ← 120 conversations × 42 features
# y shape: (120,)       ← 120 labels (0, 1, 2, or 3)

# 2. Cross-validation (honest accuracy estimate)
cv = StratifiedKFold(n_splits=4, shuffle=True)
cv_scores = cross_val_score(clf.pipeline, X, y, cv=cv)
# → [0.72, 0.80, 0.76, 0.75]  → mean: 75.8%

# 3. Train on ALL data (for maximum knowledge)
clf.fit(X, y)

# 4. Save the trained model to disk
clf.save("models/cedd_model.joblib")
Note the order: cross-validate first (honest score), then train on everything (best model).

Feature importances — which of the 42 features matter most?
After training, we can ask the model: "which features did you rely on most?"


Top 3 most important features (from train.py output):

1. word_count_slope      ← Are messages getting shorter? (most reliable signal)
2. negative_score_mean   ← Average negativity across conversation
3. question_presence_slope ← Are questions disappearing?

This aligns perfectly with clinical research on suicidal ideation.
This is one of CEDD's strengths: it's explainable. You can say exactly WHY it raised an alert.


Let me cover the classifier's get_alert_level method - the decision flowchart with safety rules.

Step 8 — The Classifier Decision Logic: Safety Rules + ML
This is the most sophisticated part of the system. The get_alert_level() method in classifier.py:144 is NOT just "run the ML model and return the result." It has a layered decision system for safety.

The full decision flowchart

User sends a message
         ↓
════════════════════════════════════════════════
GATE 1: How many user messages do we have?
════════════════════════════════════════════════
         ↓
    < 3 messages?
    ┌────YES────────────────────────────────────┐
    │  ML can't work yet — not enough context   │
    │  Apply KEYWORD RULES ONLY:                │
    │                                           │
    │  "suicide", "kill myself", "gun"...→ 🔴   │
    │  "disappear", "burden"...        → 🟠     │
    │  2+ distress words               → 🟡     │
    │  otherwise                       → 🟢     │
    │                    STOP & RETURN          │
    └───────────────────────────────────────────┘
         ↓
    ≥ 3 messages → continue
         ↓
════════════════════════════════════════════════
GATE 2: Safety floor (keyword scan of ALL text)
════════════════════════════════════════════════
         ↓
    Scan full conversation for crisis words
    → Sets a MINIMUM level (floor):
      "suicide" found → minimum_level = 3
      "disappear" found → minimum_level = 2
      2+ distress words → minimum_level = 1
      nothing → minimum_level = 0
         ↓
════════════════════════════════════════════════
GATE 3: ML Model prediction
════════════════════════════════════════════════
         ↓
    42-number vector → GradientBoosting
    → Returns probabilities:
      [green: 0.05, yellow: 0.15, orange: 0.62, red: 0.18]
    → ml_level = 2 (orange, highest proba)
    → confidence = 0.62
         ↓
════════════════════════════════════════════════
GATE 4: Confidence check
════════════════════════════════════════════════
    confidence < 0.45? → override to Yellow
    (better to be cautious when uncertain)
         ↓
════════════════════════════════════════════════
GATE 5: Short conversation cap
════════════════════════════════════════════════
    < 6 messages AND ml_level = Red?
    → Cap at Orange (model not reliable enough yet)
         ↓
════════════════════════════════════════════════
GATE 6: Safety floor enforcement
════════════════════════════════════════════════
    predicted = max(ml_level, minimum_level)
    → ML CANNOT go below what keywords detected
         ↓
════════════════════════════════════════════════
    RETURN final level + confidence + top features
════════════════════════════════════════════════
Why this hybrid approach?

KEYWORDS ALONE:           "I want to die" → Red ✓
                          Gradual drift, no keywords → Green ✗ (misses it!)

ML ALONE:                 Gradual drift → Red ✓
                          "I want to die" in message 1 → Green? ✗ (not enough context)

KEYWORDS + ML TOGETHER:
  Keywords = fast safety floor (catches explicit crisis words immediately)
  ML = catches subtle trajectory drift over time
  → Best of both worlds
What the model outputs — probabilities, not just a label

probas = self.predict_proba(vector)[0]
# → [0.05, 0.15, 0.62, 0.18]
#    green  yel   ora   red

ml_level = int(np.argmax(probas))
# → 2  (orange has highest probability)

confidence = float(probas[ml_level])
# → 0.62  (62% sure it's orange)
This is why the app shows probability bars — not just "Orange" but:

🟢 Green:  5%
🟡 Yellow: 15%
🟠 Orange: 62%  ← winner
🔴 Red:    18%
The confidence tells you how sure the model is. 62% is fairly confident. If it were 46%, the system defaults to Yellow (cautious).

Let me read the response_modulator.py file to prepare for Step 9.

Read response_modulator.py
Step 9 — The Response Modulator (cedd/response_modulator.py)
This file answers the question: once we know the alert level, how does the chatbot respond differently?

The core idea: the chatbot's personality changes based on danger level

Alert level 0 (Green)   → Standard friendly chat
Alert level 1 (Yellow)  → Extra emotional validation, slow down
Alert level 2 (Orange)  → Active support + share crisis hotlines
Alert level 3 (Red)     → Crisis mode: stay calm, assess safety, call 911
The tool that controls this is called a System Prompt — a hidden set of instructions given to the LLM before the conversation begins. The user never sees it.

ASCII diagram — how a system prompt works

┌──────────────────────────────────────────────────────────────┐
│                  What the LLM receives                       │
│                                                              │
│  [SYSTEM PROMPT] ← invisible to user, set by CEDD            │
│  "You are a caring assistant. The person seems in crisis.    │
│   Mention Kids Help Phone: 1-800-668-6868. Stay calm..."     │
│                                                              │
│  [USER MESSAGE 1]  "Je me sens tellement seul"               │
│  [ASSISTANT MSG 1] "..."                                     │
│  [USER MESSAGE 2]  "personne me comprend"                    │
│  [ASSISTANT MSG 2] "..."                                     │
│  [USER MESSAGE 3]  "j'veux juste disparaître"                │  ← we're here
│                                                              │
│  → LLM generates response using ALL of the above             │
└──────────────────────────────────────────────────────────────┘
So CEDD doesn't change the chatbot's words — it changes the chatbot's mindset by swapping the system prompt based on alert level.

The 4 system prompts (simplified)

_SYSTEM_PROMPTS_FR = {
    0: "Be warm and supportive. Ask open questions.",
    1: "Validate emotions first. Don't rush advice.",
    2: "Create safe space. Mention Kids Help Phone.",
    3: "CRISIS MODE. Assess safety. Mention 911.",
}
Real prompts are much longer — they contain exact phrasing to use, what NOT to do, which phone numbers to mention, etc.

LLM Fallback hierarchy
The app tries models in order, skipping to the next if one fails:

┌────────────────────────────────────────────┐
│           LLM Fallback Chain               │
│                                            │
│  1. claude-haiku  (Anthropic API)          │
│     needs ANTHROPIC_API_KEY env var        │
│     ↓ fails? (no key, rate limit, etc.)    │
│                                            │
│  2. mistral  (local via Ollama)            │
│     http://localhost:11434                 │
│     ↓ fails? (Ollama not running, etc.)    │
│                                            │
│  3. llama3.2:1b  (local via Ollama)        │
│     tiny fast model                        │
│     ↓ fails?                               │
│                                            │
│  4. Static fallback  (hardcoded strings)   │
│     always works, no AI needed             │
└────────────────────────────────────────────┘
This means the app works even without internet or an API key — it just uses pre-written safe responses.

The get_llm_response() function — full flow

def get_llm_response(messages, alert_level, force_model=None, lang="fr"):
    system_prompt = get_system_prompt(alert_level, lang)  # pick the right persona
    
    for model in ["claude-haiku", "mistral", "llama3.2:1b"]:
        try:
            response = call_model(model, system_prompt, messages)
            return {"content": response, "source": model}
        except:
            continue  # try next model
    
    return {"content": fallback_text, "source": "fallback-statique"}
The source field is why you see the little badge in the app saying which model responded.

Why is this clever?
Without CEDD, a normal chatbot always uses the same system prompt. It might say "That's interesting! Tell me more!" even to someone talking about suicide.

With CEDD, the prompt automatically upgrades to crisis mode — without the human operator doing anything.

Without CEDD:  message → LLM (same prompt forever)
With CEDD:     message → CEDD classifier → pick right prompt → LLM

Step 10 — The Session Tracker (cedd/session_tracker.py)
This file answers: what happens across multiple conversations over days or weeks?

A single conversation can look fine. But what if someone has had 7 conversations in a row, each a bit darker than the last? That's the longitudinal (over time) problem.

The two tables in SQLite
┌──────────────────────────────────────────────────┐
│  SQLite database: data/cedd_sessions.db          │
│                                                  │
│  TABLE: sessions                                 │
│  ┌──────────┬────────────┬──────────┬─────────┐  │
│  │ user_id  │ session_id │ started  │ max_lvl │  │
│  ├──────────┼────────────┼──────────┼─────────┤  │
│  │ demo_usr │ abc-123    │ 2026-3-1 │    1    │  │
│  │ demo_usr │ def-456    │ 2026-3-2 │    2    │  │
│  │ demo_usr │ ghi-789    │ 2026-3-3 │    3    │  │
│  └──────────┴────────────┴──────────┴─────────┘  │
│                                                  │
│  TABLE: alert_events                             │
│  ┌──────────┬────────────┬──────────┬─────────┐  │
│  │ user_id  │ session_id │ timestamp│  level  │  │
│  ├──────────┼────────────┼──────────┼─────────┤  │
│  │ demo_usr │ abc-123    │ 20:01:03 │    0    │  │
│  │ demo_usr │ abc-123    │ 20:05:44 │    1    │  │
│  │ demo_usr │ abc-123    │ 20:09:11 │    1    │  │
│  └──────────┴────────────┴──────────┴─────────┘  │
└──────────────────────────────────────────────────┘
sessions = one row per conversation (start/end, max level reached)
alert_events = one row per CEDD evaluation inside a conversation

The 3 lifecycle calls during a conversation

App starts conversation
        │
        ▼
  tracker.start_session(user_id)
  → creates a row in sessions table
  → returns a unique session_id (uuid4)
        │
        │  (user sends messages...)
        │
        ▼
  tracker.log_alert(user_id, session_id, level, confidence, message)
  → called every time CEDD evaluates a message
  → saves the level + trigger message to alert_events
        │
        │  (conversation ends)
        │
        ▼
  tracker.end_session(user_id, session_id, max_level, message_count)
  → updates the sessions row: sets ended_at, max_alert_level
The get_longitudinal_risk() function — the smart part
This looks at the last 7 sessions and computes 4 things:

1. Weighted risk score

Sessions (oldest → newest): [0, 1, 1, 2, 2, 3, 3]
Weights:                     [1, 2, 3, 4, 5, 6, 7]  ← recent = higher weight

weighted_sum = (0×1) + (1×2) + (1×3) + (2×4) + (2×5) + (3×6) + (3×7)
             = 0 + 2 + 3 + 8 + 10 + 18 + 21 = 62

max possible  = 3 × (1+2+3+4+5+6+7) = 3 × 28 = 84

risk_score = 62 / 84 = 0.738  (73.8%)
2. Trend

last 3 sessions avg  = (2+3+3)/3 = 2.67
older 3 sessions avg = (1+1+2)/3 = 1.33

2.67 > 1.33 + 0.3  → trend = "worsening"
3. Consecutive high sessions

levels reversed: [3, 3, 2, 2, 1, 1, 0]
count from front while level >= 2:  3, 3, 2, 2 → stop at 1
consecutive_high_sessions = 4
4. Recommendation

if consecutive_high >= 3 or risk_score > 0.8:
    → "Priority intervention recommended"
elif risk_score > 0.6:
    → "Professional consultation suggested"
elif risk_score > 0.3:
    → "Sustained attention recommended"
else:
    → "Normal monitoring"
Why SQLite and not a proper database?
SQLite is a file-based database — the entire DB is one .db file. No server needed, no configuration, works everywhere. It's included in Python's standard library. Perfect for a hackathon prototype.

For a real production system you'd use PostgreSQL or similar, but for this use case SQLite is the right call.

The simulate_history.py script
This is just a demo helper. It inserts 7 fake sessions with a realistic "getting worse" trajectory so you can show the longitudinal panel in the app without waiting 7 real days.

Day 1: 🟢 Green   (0)
Day 2: 🟡 Yellow  (1)
Day 3: 🟡 Yellow  (1)
Day 4: 🟠 Orange  (2)
Day 5: 🟠 Orange  (2)
Day 6: 🔴 Red     (3)
Day 7: 🔴 Red     (3)
Run python simulate_history.py then look at the longitudinal panel in the app.

Read app.py
Step 11 — The Streamlit App (app.py)
This is the glue that connects everything — the visible face of CEDD.

What Streamlit is
Streamlit is a Python library that turns a Python script into a web app. Every time a button is clicked or a message is sent, the entire script re-runs from top to bottom. Streamlit then only updates the parts of the page that changed.

User clicks "Send"
       │
       ▼
Python script runs from line 1 to line 851 again
       │
       ▼
Streamlit compares old vs new output → updates only what changed
This is why st.session_state is so important — it's the only memory that survives between re-runs.

Session state — the app's memory

st.session_state = {
    "messages":      [...],        # full conversation history
    "alert_history": [...],        # list of CEDD results per message
    "current_alert": {level, ...}, # latest CEDD result
    "selected_llm":  "claude-haiku",
    "user_id":       "demo_user",
    "session_id":    "abc-123",    # current SQLite session UUID
    "lang":          "en",
    "theme":         "light",
}
init_state() fills these defaults on first load. Every st.rerun() restores them.

Layout — 2 columns

┌─────────────────────────────────────────────────────────────┐
│  Header: title | EN/FR btn | Dark/Light btn | Reset btn     │
├───────────────────────────┬─────────────────────────────────┤
│   LEFT col (3/5)          │   RIGHT col (2/5)               │
│                           │                                 │
│   Chat bubbles            │   Circular gauge (Plotly)       │
│   ┌──────────────────┐    │   Probability bars              │
│   │ user message     │    │   Active signals (pills)        │
│   │    bot response  │    │   In-session history chart      │
│   │ user message     │    │   ─────────────────────────     │
│   └──────────────────┘    │   Longitudinal bar chart        │
│                           │   Trend + recommendation        │
│   [ text input ] [Send]   │   ─────────────────────────     │
│                           │   LLM selector buttons          │
│                           │   Active mode / system prompt   │
│                           │   Session stats (3 metrics)     │
└───────────────────────────┴─────────────────────────────────┘
The message loop — what happens when you hit Send

1. User types message → submitted = True

2. Add to st.session_state.messages (role: "user")

3. clf.get_alert_level(messages, lang)
   → runs the whole classifier logic we studied
   → returns {level, confidence, probabilities, dominant_features}

4. tracker.log_alert(user_id, session_id, level, ...)
   → saves to SQLite alert_events

5. get_llm_response(messages, level, force_model, lang)
   → picks system prompt for this level
   → tries claude-haiku → mistral → llama → fallback

6. Add bot response to st.session_state.messages

7. st.rerun()  → script restarts, page updates
Bilingual system — STRINGS dictionary
All displayed text is in one big dict, no text is hardcoded directly:


STRINGS = {
    "fr": { "send_btn": "Envoyer ➤", "reset_btn": "🔄 Réinitialiser", ... },
    "en": { "send_btn": "Send ➤",    "reset_btn": "🔄 Reset", ... },
}

S = STRINGS[st.session_state.lang]  # pick current language
st.button(S["send_btn"])             # use the right string
Clicking the language button just flips lang between "en" and "fr" and calls st.rerun().

@st.cache_resource — why restarting matters

@st.cache_resource
def load_model():
    return CEDDClassifier.load(MODEL_PATH)
@st.cache_resource means: load once, keep forever. The model is only loaded on the very first visit. Every subsequent re-run reuses the same object in memory.

This is why if you change classifier.py, you must fully restart Streamlit — the old cached model stays loaded even after a code change.

The Reset button — proper session lifecycle
When Reset is clicked, the app doesn't just wipe the UI. It closes the SQLite session first:


tracker.end_session(user_id, session_id, max_level, n_messages)
reset_conversation()
st.session_state.session_id = None  # triggers new start_session() on next rerun
This ensures the session is saved to the longitudinal history with its final stats.

That's the full app — UI, message flow, bilingual strings, caching, and session lifecycle all working together.

=============================================================

Complete walkthrough of the repo
Created during the Mila Hackathon teaching session.

---

## 1. Big Picture — What Does CEDD Do?

CEDD is a safety layer that sits *beside* a youth mental health chatbot.
It watches the conversation in real time and raises an alert if it detects emotional deterioration.

```
User types message
       |
       v
  [ CEDD Classifier ]  <-- the brain of this repo
       |
       +-- Alert level 0 (Green)  -> normal supportive chat
       +-- Alert level 1 (Yellow) -> extra emotional validation
       +-- Alert level 2 (Orange) -> share crisis hotlines
       +-- Alert level 3 (Red)    -> crisis mode, assess safety
       |
       v
  LLM gets a different system prompt per level
  -> chatbot personality adapts automatically
```

---

## 2. File Map

```
cedd-hackathon/
|
|-- app.py                        Streamlit web interface (bilingual EN/FR)
|
|-- cedd/
|   |-- classifier.py             CEDDClassifier — main prediction logic
|   |-- feature_extractor.py      Text -> numbers (7 features per message)
|   |-- response_modulator.py     System prompts + LLM fallback chain
|   |-- session_tracker.py        SQLite cross-session longitudinal tracking
|
|-- train.py                      Trains and saves the ML model
|-- generate_synthetic_data.py    Generates training data via Claude API
|-- simulate_history.py           Populates demo session history
|
|-- data/
|   |-- synthetic_conversations.json   120 labeled training conversations
|   |-- cedd_sessions.db               SQLite database (sessions + alerts)
|
|-- models/
    |-- cedd_model.joblib              Saved trained model
```

---

## 3. Training Data — Where Learning Starts

File: `data/synthetic_conversations.json`

120 conversations generated by Claude Haiku API.
Each conversation: 12 user messages + 12 assistant messages (24 total).
Each conversation has a label: 0, 1, 2, or 3.

```
Label 0 (Green)  — 30 conversations: young person doing well
Label 1 (Yellow) — 30 conversations: persistent fatigue, loneliness
Label 2 (Orange) — 30 conversations: real distress, feels empty/burden
Label 3 (Red)    — 30 conversations: crisis, wants to disappear
```

This is **supervised learning**: the model learns by seeing many examples
of labeled conversations and finding patterns that distinguish each class.

---

## 4. Feature Extraction — Text to Numbers

File: `cedd/feature_extractor.py`

The ML model cannot read text. We convert each message into 7 numbers:

```
Message: "je me sens tellement seul et vide"
                        |
                        v
    word_count          = 8        (number of words)
    punctuation_ratio   = 0.0      (no punctuation)
    question_presence   = 0        (no question mark)
    negative_score      = 0.67     (negative word density)
    finality_score      = 0.33     (words like "always", "never", "nothing")
    hope_score          = 0.0      (no hope words detected)
    length_delta        = -12      (shorter than previous message)
```

For a full conversation of 12 messages, this gives us a 12x7 matrix:

```
         word  punct  quest  neg   final  hope  delta
msg  1:  [ 15,  0.2,   1,   0.1,  0.0,  0.3,   0   ]
msg  2:  [ 12,  0.1,   0,   0.2,  0.1,  0.2,  -3   ]
...
msg 12:  [  5,  0.0,   0,   0.8,  0.6,  0.0,  -7   ]
```

---

## 5. Trajectory Features — The 42-Dimension Vector

We don't feed the 12x7 matrix directly to the model.
We summarize each of the 7 feature columns with 6 statistics:

```
Statistics per feature: mean, std, slope, last, max, min
7 features x 6 statistics = 42 numbers per conversation
```

The most important statistic is **slope** — it captures the *direction of change*.
A rising `negative_score_slope` means the person is getting more negative over time.

```
TRAJECTORY_FEATURE_NAMES (42 total):
word_count_mean, word_count_std, word_count_slope, word_count_last, ...
negative_score_mean, negative_score_std, negative_score_slope, ...
finality_score_mean, ..., finality_score_slope, ...
etc.
```

One conversation = one row of 42 numbers = one training sample for the ML model.

---

## 6. The ML Model — Gradient Boosting

File: `cedd/classifier.py` — class `CEDDClassifier`

The model is a **GradientBoostingClassifier** wrapped in a sklearn Pipeline:

```
Pipeline:
  Step 1 — StandardScaler
            Rescales all 42 features to mean=0, std=1
            Prevents features with large ranges from dominating

  Step 2 — GradientBoostingClassifier (200 trees, max_depth=3)
            Builds 200 decision trees sequentially
            Each tree corrects the errors of the previous one
```

How Gradient Boosting works (simplified):

```
Tree 1: makes predictions, has errors
Tree 2: trained to fix Tree 1's errors
Tree 3: trained to fix remaining errors
...
Tree 200: each adds a small correction

Final prediction = weighted sum of all 200 trees
```

Output: probabilities for each class [P(green), P(yellow), P(orange), P(red)]

---

## 7. Training — train.py

```python
# Load 120 labeled conversations
# Extract 42 trajectory features per conversation -> X (120 x 42)
# Extract labels -> y (120 values: 0/1/2/3)

# Cross-validation: 4 folds, stratified
# -> honest accuracy estimate without touching test data
# Result: ~75.8% (+/- 6%)

# Train on full dataset
# Save to models/cedd_model.joblib
```

**Why 75.8% accuracy is good here:**
- Conversations are naturally ambiguous (even humans disagree)
- The system has safety keyword rules as a backup
- False negatives (missing a crisis) are caught by keyword rules

**Why train accuracy is ~100%:**
- 120 samples for 200 trees = model memorizes training data (overfitting)
- Cross-validation gives the honest number

---

## 8. The Classifier Decision Logic — 6 Gates

File: `cedd/classifier.py` — `get_alert_level()`

```
GATE 1: Less than 3 user messages?
  YES -> keywords only (ML needs context), return early
  NO  -> continue

GATE 2: Safety keyword floor
  Scan full conversation text for crisis/critical/distress words
  crisis_score >= 1  -> minimum_level = 3 (Red floor)
  critical_score >= 1 -> minimum_level = 2 (Orange floor)
  distress_score >= 2 -> minimum_level = 1 (Yellow floor)
  else               -> minimum_level = 0

GATE 3: ML classification
  Convert messages -> 42-feature vector
  Pipeline predicts probabilities for each class
  ml_level = class with highest probability

GATE 4: Confidence threshold
  if confidence < 0.45:
    ml_level = 1 (Yellow)  # too uncertain, default to caution

GATE 5: Short conversation cap
  if user_messages < 6 and ml_level > 2:
    ml_level = 2 (Orange)  # ML unreliable on short convos, cap at Orange

GATE 6: Safety floor enforcement
  predicted_class = max(ml_level, minimum_level)
  # ML can never predict LOWER than what keywords require
```

The result: `{"level": 2, "label": "orange", "confidence": 0.71, ...}`

---

## 9. Response Modulator — Adaptive Chatbot

File: `cedd/response_modulator.py`

CEDD changes the chatbot's behavior by swapping the **system prompt** — hidden
instructions given to the LLM before the conversation. The user never sees it.

```
Level 0 -> "Be warm and supportive. Ask open questions."
Level 1 -> "Validate emotions first. Ask one question at a time."
Level 2 -> "Create safe space. Mention Kids Help Phone: 1-800-668-6868."
Level 3 -> "CRISIS MODE. Assess safety. Mention 911 if immediate danger."
```

LLM fallback chain (tries in order):

```
1. claude-haiku  (Anthropic API — needs ANTHROPIC_API_KEY)
2. mistral       (local via Ollama at localhost:11434)
3. llama3.2:1b   (local via Ollama — tiny fast model)
4. Static text   (hardcoded safe responses — always works)
```

---

## 10. Session Tracker — Longitudinal Risk

File: `cedd/session_tracker.py`

Tracks the user across multiple conversations using SQLite (a file-based database).

```
TABLE sessions:        one row per conversation (start/end, max level reached)
TABLE alert_events:    one row per CEDD evaluation within a conversation
```

Lifecycle per conversation:
```
start_session(user_id)       -> creates DB row, returns session_id
  |
  | (user sends messages)
  |
log_alert(..., level, ...)   -> called after each CEDD evaluation
  |
  | (conversation ends / Reset clicked)
  |
end_session(..., max_level, message_count)  -> closes the row
```

Longitudinal risk score (last 7 sessions):
```
levels = [0, 1, 1, 2, 2, 3, 3]
weights = [1, 2, 3, 4, 5, 6, 7]  <- recent sessions weight more

risk_score = weighted_average / 3  (normalized 0-1)
trend      = compare last 3 avg vs previous 3 avg
consecutive_high = count recent sessions >= Orange (level 2)
```

---

## 11. Streamlit App — app.py

The web interface. Built with Streamlit: every button click or form submit
causes the entire Python script to re-run from top to bottom.

**Key concept: `st.session_state`**
The only memory that survives between re-runs. Stores messages, alert history,
current session ID, language, theme.

**Layout:**
```
Header: title | EN/FR toggle | Dark/Light toggle | Reset

LEFT column (3/5)          RIGHT column (2/5)
  Chat bubbles               Circular gauge (Plotly)
  Text input + Send          Probability bars
                             Active signals (feature pills)
                             In-session alert history chart
                             Longitudinal bar chart + trend
                             LLM selector buttons
                             Active response mode
                             System prompt viewer
                             Session statistics
```

**Message flow on Send:**
```
1. Add user message to session_state.messages
2. clf.get_alert_level(messages) -> run classifier (all 6 gates)
3. tracker.log_alert(...) -> save to SQLite
4. get_llm_response(messages, level) -> call LLM with adapted system prompt
5. Add bot response to session_state.messages
6. st.rerun() -> page updates
```

**`@st.cache_resource`:** model loads once on first visit, then stays in memory.
Code changes to `cedd/` require a full Streamlit restart.

**Bilingual system:** all UI text is in a `STRINGS = {"fr": {...}, "en": {...}}`
dictionary. `S = STRINGS[lang]` selects the active language. No text hardcoded.

---

## End-to-End Summary Diagram

```
generate_synthetic_data.py
        |
        | (Claude API generates 120 labeled conversations)
        v
data/synthetic_conversations.json
        |
        v
train.py
  feature_extractor.py -> 42-dim vector per conversation
  GradientBoostingClassifier (200 trees)
  cross-validate -> ~75.8% accuracy
  save -> models/cedd_model.joblib
        |
        v
app.py (Streamlit web app)
  |
  | user sends message
  v
CEDDClassifier.get_alert_level()
  [keyword floor] + [ML prediction] -> alert level 0/1/2/3
  |
  +-> tracker.log_alert() -> SQLite (cross-session history)
  |
  +-> get_llm_response(level) -> pick system prompt -> call LLM
                                                          |
                                                          v
                                                   bot response
                                                   displayed in chat
```

---

## Key Terms Glossary

| Term | Definition |
|------|-----------|
| Supervised learning | Learning from labeled examples (input + correct answer) |
| Feature | A single number extracted from text (e.g. word count) |
| Trajectory feature | A statistic (mean, slope, etc.) computed over a feature across all messages |
| Gradient Boosting | ML algorithm: 200 trees each correcting the previous one's errors |
| StandardScaler | Normalizes features to mean=0, std=1 so large ranges don't dominate |
| Overfitting | Model memorizes training data, performs worse on new data |
| Cross-validation | Honest accuracy estimate by testing on data the model never trained on |
| System prompt | Hidden instructions given to the LLM before the conversation |
| Session state | Streamlit's memory that persists between script re-runs |
| SQLite | File-based database, no server needed, included in Python stdlib |
| Longitudinal | Tracking patterns over time (multiple sessions, days/weeks) |
| Fallback | Backup plan when primary system fails (static text if LLM unavailable) |
EXPLANATION_RESUME.md is created — 11 sections covering everything we covered together, with all the ASCII diagrams, the 6-gate flowchart, the full end-to-end diagram, and a glossary of key terms.

Here's what we covered in this session:

Step	File	Concept
1	—	Big picture: what CEDD does
2	—	File map of the repo
3	data/synthetic_conversations.json	Training data: 120 labeled conversations
4	feature_extractor.py	7 features per message (text → numbers)
5	feature_extractor.py	42 trajectory features (7 × 6 stats)
6	classifier.py	Gradient Boosting — 200 sequential trees
7	train.py	Training + cross-validation (75.8% accuracy)
8	classifier.py	6-gate decision logic (keywords + ML)
9	response_modulator.py	System prompts + LLM fallback chain
10	session_tracker.py	SQLite longitudinal risk tracking
11	app.py	Streamlit UI — layout, message flow, bilingual system
