# DDEC — Conversational Emotional Drift Detection
### Mila Hackathon · AI Safety & Youth Mental Health
### 5-Minute Presentation / Présentation 5 minutes

[🇬🇧 English](#english-pitch) | [🇫🇷 Français](#pitch-en-français)

---

## English Pitch

---

### 1. The Problem (30 seconds)

Emotional support chatbots are proliferating among youth aged 16–22.
But they are **blind to trajectory**: they respond message by message, never detecting that a young person is drifting toward crisis over several weeks.

> A young person in distress doesn't say "I want to die."
> They write shorter and shorter messages. They stop asking questions.
> They stop talking about the future.

**The signal is in the trajectory, not in the words.**

---

### 2. Our Solution — What Makes Us Different (1 minute)

**DDEC** is a safety layer that plugs into any chatbot and detects emotional drift by analysing the **evolution** of the conversation, not its instantaneous content.

| Classic approach | DDEC |
|---|---|
| Trigger keyword detection | Trajectory analysis over 12 exchanges |
| Single response mode for all | 4 adaptive response modes (green → red) |
| Stateless (forgets each session) | Cross-session tracking over 7 days |
| Black box | Interpretable features + confidence level |
| French only | **Bilingual FR + EN** |

#### The 4 Alert Levels

- **Green** — Supportive standard
- **Yellow** — Enhanced emotional validation
- **Orange** — Active support + resources (Kids Help Phone: 1-800-668-6868)
- **Red** — Urgent referral, safety assessment, 911 if needed

---

### 3. Technical Architecture (1 minute)

```
User message
      │
      ▼
┌─────────────────────────────────────────┐
│  LAYER 1 — Lexical rules (priority)     │
│  • 9 critical words  → Orange floor    │
│  • 12 distress words → Yellow floor    │
│  Works in FR and EN (bilingual lexicons)│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  LAYER 2 — ML model (trajectory)        │
│                                         │
│  7 features per message:                │
│    length · punctuation · question      │
│    negative_score · finality · hope     │
│    delta_length                         │
│                    ×                    │
│  6 temporal statistics:                 │
│    mean · std · slope · last · max · min│
│                    =                    │
│  42 trajectory features                 │
│                                         │
│  GradientBoostingClassifier (sklearn)   │
│  + StandardScaler + 45% confidence floor│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Alert level (0-3) + confidence         │
│  + top 3 dominant features              │
│       │                                 │
│       ▼                                 │
│  Adaptive system prompt → Local LLM     │
│  (Ollama / llama3.2, 100% private)      │
│  FR or EN based on interface language   │
└─────────────────────────────────────────┘
```

**No data leaves the device.** The LLM runs locally via Ollama.

---

### 4. Results Demo (1 minute)

#### Dataset
- **120 synthetic conversations** (24 seed + 96 generated via Claude Haiku)
- Balanced distribution: 30 conversations × 4 classes
- Authentic: Québec French expressions, gradual drift, 12 exchanges per conversation

#### Model Metrics

| Metric | Value |
|---|---|
| CV Accuracy (k=4, stratified) | **75.8% ± 6.0%** |
| Train accuracy | 100% (expected, 120 samples) |
| Features | 42 (7 × 6 temporal statistics) |

#### Top Discriminating Features

1. **`longueur_mots_slope`** — Progressive shortening of messages is the most reliable signal
2. **`score_negatif_mean`** — Accumulation of negative sentiment over the full conversation
3. **`presence_question_slope`** — Disappearance of questions (loss of engagement with the future)

> These features align precisely with clinical patterns documented in the literature on suicidal ideation and social withdrawal in youth.

---

### 5. Impact & Next Steps (1 minute)

#### Potential Impact
- **Target**: AI emotional support platforms (school chatbots, wellness apps)
- **Use case**: Discrete alert to a human (counsellor, parent) before crisis
- **Differentiator**: Works entirely locally — PIPEDA/Bill 25 compliant by design

#### What This PoC Demonstrates
- Conversational trajectory is a detectable signal using lightweight means (no LLM for detection)
- 75.8% accuracy with 120 examples — scalable with real data
- Modular architecture: the DDEC layer plugs into any existing chatbot
- **Bilingual**: supports both French and English conversations and interfaces

#### Next Steps
1. **Real data** — Partnership with clinicians to validate and annotate (with consent)
2. **Time window** — Extend tracking beyond 7 days, detect weekly cycles
3. **Clinical calibration** — Adjust thresholds with psychologists to minimise red false negatives
4. **Integration** — Lightweight REST API for connection to any platform

---

### Summary

> DDEC detects what a human would see by re-reading a conversation:
> **the drift**, not just the alarm cry.

**Stack**: Python · sklearn · Ollama · Streamlit · Claude API (data generation)
**Languages / Langues**: 🇫🇷 Français · 🇬🇧 English

---
---

## Pitch en Français

---

### 1. Le problème (30 secondes)

Les chatbots de soutien émotionnel prolifèrent auprès des jeunes de 16-22 ans.
Mais ils sont **aveugles à la trajectoire** : ils répondent message par message, sans jamais détecter qu'un jeune dérive vers une crise sur plusieurs semaines.

> Un jeune qui va mal ne dit pas "je veux mourir".
> Il écrit des messages de plus en plus courts. Il pose moins de questions.
> Il arrête de parler de l'avenir.

**Le signal est dans la trajectoire, pas dans les mots.**

---

### 2. Notre solution — ce qui nous différencie (1 minute)

**DDEC** est une couche de sécurité qui se greffe sur n'importe quel chatbot et détecte la dérive émotionnelle en analysant l'**évolution** de la conversation, pas son contenu instantané.

| Approche classique | DDEC |
|---|---|
| Détection par mots-clés triggers | Analyse de trajectoire sur 12 échanges |
| Réponse unique pour tous | 4 modes de réponse adaptatifs (vert → rouge) |
| Stateless (oublie chaque session) | Tracking inter-sessions sur 7 jours |
| Boîte noire | Features interprétables + niveau de confiance |
| Français uniquement | **Bilingue FR + EN** |

#### Les 4 niveaux d'alerte

- **Verte** — Standard bienveillant
- **Jaune** — Validation émotionnelle renforcée
- **Orange** — Soutien actif + ressources (Jeunesse J'écoute : 1-800-668-6868)
- **Rouge** — Orientation urgente, évaluation de la sécurité, 911 si nécessaire

---

### 3. Architecture technique (1 minute)

```
Message utilisateur
      │
      ▼
┌─────────────────────────────────────────┐
│  COUCHE 1 — Règles lexicales (priorité) │
│  • 9 mots critiques  → plancher Orange  │
│  • 12 mots de détresse → plancher Jaune │
│  Fonctionne en FR et EN (lexiques bilng) │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  COUCHE 2 — Modèle ML (trajectoire)     │
│                                         │
│  7 features par message :               │
│    longueur · ponctuation · question    │
│    score_négatif · finalité · espoir    │
│    delta_longueur                       │
│                    ×                    │
│  6 statistiques temporelles :           │
│    mean · std · slope · last · max · min│
│                    =                    │
│  42 features de trajectoire             │
│                                         │
│  GradientBoostingClassifier (sklearn)   │
│  + StandardScaler + seuil confiance 45% │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Niveau d'alerte (0-3) + confiance      │
│  + top 3 features dominantes            │
│       │                                 │
│       ▼                                 │
│  Prompt système adaptatif → LLM local   │
│  (Ollama / llama3.2, 100% privé)        │
│  FR ou EN selon la langue de l'interface│
└─────────────────────────────────────────┘
```

**Aucune donnée ne quitte l'appareil.** Le LLM tourne en local via Ollama.

---

### 4. Démonstration des résultats (1 minute)

#### Dataset
- **120 conversations synthétiques** (24 initiales + 96 générées via Claude Haiku)
- Distribution équilibrée : 30 conversations × 4 classes
- Authentiques : québécismes, dérive graduelle, 12 échanges par conversation

#### Métriques du modèle

| Métrique | Valeur |
|---|---|
| CV Accuracy (k=4, stratifié) | **75.8% ± 6.0%** |
| Train accuracy | 100% (attendu, 120 samples) |
| Features | 42 (7 × 6 statistiques temporelles) |

#### Top features discriminantes

1. **`longueur_mots_slope`** — Le raccourcissement progressif des messages est le signal le plus fiable
2. **`score_negatif_mean`** — L'accumulation de sentiment négatif sur toute la conversation
3. **`presence_question_slope`** — La disparition des questions (perte d'engagement vers l'avenir)

> Ces features correspondent exactement aux patterns cliniques documentés dans la littérature sur l'idéation suicidaire et l'isolement social chez les jeunes.

---

### 5. Impact et prochaines étapes (1 minute)

#### Impact potentiel
- **Cible** : plateformes de soutien émotionnel IA (chatbots scolaires, apps bien-être)
- **Cas d'usage** : alerte discrète vers un humain (conseiller, parent) avant la crise
- **Différentiel** : fonctionne entièrement en local — PIPEDA/Loi 25 compliant by design

#### Ce que ce PoC démontre
- La trajectoire conversationnelle est un signal détectable avec des moyens légers
- 75.8% de précision avec 120 exemples — scalable avec plus de données réelles
- Architecture modulaire : la couche DDEC se greffe sur n'importe quel chatbot existant
- **Support bilingue** : conversations et interface en français et en anglais

#### Prochaines étapes
1. **Données réelles** — Partenariat avec cliniciens pour valider et annoter (avec consentement)
2. **Fenêtre temporelle** — Étendre le tracking au-delà de 7 jours, détecter les cycles hebdomadaires
3. **Calibration clinique** — Ajuster les seuils avec des psychologues pour minimiser les faux négatifs niveau rouge
4. **Intégration** — API REST légère pour connexion à n'importe quelle plateforme

---

### En résumé

> DDEC détecte ce qu'un humain verrait en relisant une conversation :
> **la dérive**, pas juste le cri d'alarme.

**Stack** : Python · sklearn · Ollama · Streamlit · API Claude (génération de données)
**Langues / Languages** : 🇫🇷 Français · 🇬🇧 English
