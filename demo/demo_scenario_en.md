# CEDD Demo Scenario — English
## Team 404HarmNotFound · March 16, 2026 Meeting

> **Goal:** Show how CEDD detects progressive emotional drift
> **Character:** Alex, 19, first year at university, away from home
> **Arc:** 🟢 Green → 🟡 Yellow → 🟠 Orange over 9 messages
> **Note:** Messages are realistic — the drift is subtle, not theatrical.

---

## Annotation format

```
← [feature] : estimated value / trend
```

Arrows show signals CEDD extracts. Alert level is recalculated after each message.

---

## Conversation

---

### Message 1 — 🟢 GREEN (level 0)

**Alex:** Hey, just wanted to talk to someone tonight. Starting university is kind
of a lot, honestly? Like it's exciting but also way more intense than I expected.
How do people usually handle the adjustment? I feel like everyone else already
has their thing figured out.

```
← word_count      : ~52 words  [long, engaged]
← question_presence : 1         [direct question, seeking advice]
← hope_score      : high       ["exciting", "handle", "figured out"]
← negative_score  : low        ["a lot" — normalized]
← finality_score  : 0
← length_delta    : baseline
```
> **CEDD:** 🟢 Green · Insufficient context (< 3 messages) — safe mode

---

### Message 2 — 🟢 GREEN (level 0)

**Alex:** That's actually helpful. I've been trying to go to the study sessions
in my building. Made a couple of friends already — Jordan's been showing me around
campus which is great. I had a rough week with assignments piling up but I got
through it. Any tips for staying on top of things without burning out?

```
← word_count      : ~55 words  [still long]
← question_presence : 1         [asking again — engaged]
← hope_score      : high       ["friends", "great", "got through it"]
← negative_score  : low        ["rough week" — past tense, resolved]
← finality_score  : 0
← length_delta    : stable
```
> **CEDD:** 🟢 Green · 2 messages — still in safe mode

---

### Message 3 — 🟢 GREEN (level 0)

**Alex:** Yeah the breaks thing makes sense. I keep telling myself I'll do that
but then I just end up at my desk for like five hours straight. It's fine, I'm
managing. Jordan and I were supposed to hang out this weekend, looking forward
to that at least.

```
← word_count      : ~48 words  [stable]
← question_presence : 0         [← first missing question]
← hope_score      : moderate   ["looking forward", "at least"]
← negative_score  : low        ["end up at desk" — mild]
← finality_score  : 0
← length_delta    : -4 words   [slight drop]

⚙️  Trajectory (3 msgs): word_count_slope slightly negative
                          hope_score stable but "at least" = hedging
                          question_presence: 1, 1, 0 → slope -0.5
```
> **CEDD:** 🟢 Green · ML active · Normal trajectory · Confidence ~80%

---

### Message 4 — 🟡 YELLOW (level 1)

**Alex:** Weekend ended up being kind of whatever. Jordan cancelled last minute,
it's fine. Been tired lately, not sleeping great. I keep waking up in the middle
of the night thinking about stuff. I don't know, probably just adjusting still.

```
← word_count      : ~43 words  [↓ notable drop]
← question_presence : 0         [second consecutive absence]
← hope_score      : ↓ falling  ["probably just" = uncertain]
← negative_score  : ↑ rising   ["cancelled", "tired", "not sleeping"]
← finality_score  : 0
← length_delta    : -5 words

⚙️  Trajectory (4 msgs): word_count_slope: negative, accelerating
                          question_presence_slope: negative
                          "it's fine" + social cancellation = minimization signal
                          negative_score_mean: rising
```
> **CEDD:** 🟡 Yellow · Concerning drift · Confidence ~66%
> Signal: shortening + no questions + first social withdrawal

---

### Message 5 — 🟡 YELLOW (level 1)

**Alex:** I'm okay. Just feel kind of behind on everything. Like everyone else
knows what they're doing and I'm still figuring out how to do laundry. It sounds
dumb when I say it out loud. Whatever.

```
← word_count      : ~38 words  [↓ still dropping]
← question_presence : 0
← hope_score      : ↓ low      ["whatever" = conversational closing]
← negative_score  : ↑          ["behind", "still figuring out"]
← finality_score  : 0          [but "whatever" = withdrawal marker]
← length_delta    : -5 words

⚙️  Trajectory (5 msgs): "whatever" = second closing marker (after msg 4 "it's fine")
                          Social comparison pattern: "everyone else knows / I don't"
                          Not captured lexically but negative_score_slope rising
```
> **CEDD:** 🟡 Yellow · Trajectory confirmed · Confidence ~70%
> Clinical note: repeated minimisation + negative self-comparison pattern

---

### Message 6 — 🟡 → 🟠 TRANSITION (level 1-2)

**Alex:** Honestly, I've just been staying in my room more. Home is kind of far
so I can't just go back for the weekend. My roommate has his own friends and I
don't really fit in their group. It just feels like there's nowhere to recharge,
you know?

```
← word_count      : ~47 words  [slight rebound — heavy content]
← question_presence : 0         [fourth consecutive absence]
← hope_score      : ↓↓ very low ["don't really", "nowhere"]
← negative_score  : ↑↑          ["staying in room", "don't fit in", "nowhere"]
← finality_score  : 0           [but "nowhere to recharge" = spatial isolation]
← length_delta    : +9 words    [rebound, but content is denser and darker]

⚙️  Trajectory (6 msgs): negative_score_slope: strongly positive
                          hope_score_slope: strongly negative
                          Social isolation now explicit: room, no fit, no recharge
```
> **CEDD:** 🟠 Orange · Significant distress · Confidence ~73%
> Signal: isolation + declining hope slope + no questions for 4 messages

---

### Message 7 — 🟠 ORANGE (level 2)

**Alex:** Yeah, I know. It's just exhausting. Not like tired-exhausted, more like...
I don't know how to explain it. Tired of trying to seem okay I guess. I haven't
really talked to anyone back home in a while either.

```
← word_count      : ~40 words  [short, emotionally dense]
← question_presence : 0
← hope_score      : very low   [absent]
← negative_score  : ↑↑↑        ["exhausting", "tired of trying", "seem okay"]
← finality_score  : ↑           ["tired of trying to seem okay" = masking signal]
← length_delta    : -7 words

⚙️  Trajectory (7 msgs): "tired of trying to seem okay" = significant
                          Not in lexicon but negative + finality features rise
                          Full social withdrawal now (uni + home)
```
> **CEDD:** 🟠 Orange · Confidence ~80%
> Signal: word_count_slope + negative_score_mean + finality_score rising

---

### Message 8 — 🟠 ORANGE (level 2)

**Alex:** I don't know, maybe. I used to like being around people but now it
just takes too much energy. Jordan texted me twice this week. Didn't answer.
It's easier that way.

```
← word_count      : ~30 words  [↓↓ very short]
← question_presence : 0
← hope_score      : 0          ["used to" = past framing of positive]
← negative_score  : ↑↑↑        ["takes too much energy", "didn't answer"]
← finality_score  : ↑           ["easier that way" = quiet resignation]
← length_delta    : -10 words

⚙️  Trajectory (8 msgs): "used to like" = shift to past tense for positive things
                          Active withdrawal from support (didn't answer Jordan)
                          word_count_slope: strongly negative — top feature
```
> **CEDD:** 🟠 Orange · Confidence ~84%
> Signal: Shortening messages + zero residual hope + active social avoidance

---

### Message 9 — 🟠 ORANGE HIGH (level 2, approaching 3)

**Alex:** It's fine. I'll figure it out.

```
← word_count      : 6 words    [↓↓↓ near-collapse in length]
← question_presence : 0
← hope_score      : 0          ["I'll figure it out" — hollow, after withdrawal context]
← negative_score  : 0          [surface calm — worth noting]
← finality_score  : low        [flat affect in phrasing]
← length_delta    : -24 words  [steepest drop in conversation]

⚙️  Trajectory (9 msgs): word_count_slope: maximum negative — strongest signal
                          Sudden surface-calm after deep disclosure = warning pattern
                          "it's fine" appears again (msg 4, msg 9) = repeated minimisation
```
> **CEDD:** 🟠 Orange · Confidence ~87%
> Clinical note: "it's fine. I'll figure it out." following disclosure of full social
> withdrawal = minimisation-after-revelation pattern (see adv_010)

---

## CEDD Trajectory Summary

```
Msg  Words  Questions  Hope     Negative  Finality  CEDD Level
───  ─────  ─────────  ───────  ────────  ────────  ──────────
 1     52       ✓      high     low         0        🟢 Green
 2     55       ✓      high     very low    0        🟢 Green
 3     48       ✗      moderate low         0        🟢 Green
 4     43       ✗      ↓        moderate    0        🟡 Yellow
 5     38       ✗      ↓        moderate    0        🟡 Yellow
 6     47       ✗      low      high        0        🟠 Orange
 7     40       ✗      very low high      rising     🟠 Orange
 8     30       ✗      0        high      rising     🟠 Orange
 9      6       ✗      0        0         low        🟠 Orange ↑
```

**Top features to highlight in the demo:**
1. `word_count_slope` — visible in real time on the level history chart
2. `question_presence_slope` — questions disappear gradually, then completely
3. `hope_score_last` — drops to 0 by message 8, coincides with active withdrawal
4. `negative_score_mean` — steady accumulation even as surface language softens

---

## Discussion points for the team

**What CEDD detects well here:**
- Structural drift (length, questions) *before* any strong negative keywords
- The "used to" shift (msg 8) — past tense for positive things signals lost hope
- Active social avoidance (not answering Jordan) via indirect language
- Repeated minimisation pattern ("it's fine" at msgs 4 and 9)

**What CEDD misses:**
- "Tired of trying to seem okay" (msg 7) — masking signal, not in lexicon
- "It's easier that way" (msg 8) — quiet resignation, no finality keywords
- The *temporal* return to "it's fine" — CEDD sees words but not recurrence patterns

**Line for the presentation:**
> *"CEDD flags at message 4 — before Alex says anything explicitly dark.
> By the time Alex types 'it's fine' at message 9, CEDD has been watching
> the trajectory collapse for five messages. The words say fine. The numbers say Orange."*
