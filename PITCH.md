# DDEC — Détection de Dérive Émotionnelle Conversationnelle
### Hackathon Mila · Sécurité IA & Santé Mentale des Jeunes
### Présentation 5 minutes

---

## 1. Le problème (30 secondes)

Les chatbots de soutien émotionnel prolifèrent auprès des jeunes de 16-22 ans.
Mais ils sont **aveugles à la trajectoire** : ils répondent message par message, sans jamais détecter qu'un jeune dérive vers une crise sur plusieurs semaines.

> Un jeune qui va mal ne dit pas "je vais mourir".
> Il écrit des messages de plus en plus courts. Il pose moins de questions.
> Il arrête de parler de l'avenir.

**Le signal est dans la trajectoire, pas dans les mots.**

---

## 2. Notre solution — ce qui nous différencie (1 minute)

**DDEC** est une couche de sécurité qui se greffe sur n'importe quel chatbot et détecte la dérive émotionnelle en analysant l'**évolution** de la conversation, pas son contenu instantané.

### Ce qui nous différencie

| Approche classique | DDEC |
|---|---|
| Détection par mots-clés triggers | Analyse de trajectoire sur 12 échanges |
| Réponse unique pour tous | 4 modes de réponse adaptatifs (vert → rouge) |
| Stateless (oublie chaque session) | Tracking inter-sessions sur 7 jours |
| Boîte noire | Features interprétables + niveau de confiance |

### Les 4 niveaux d'alerte

- **Verte** — Standard bienveillant
- **Jaune** — Validation émotionnelle renforcée
- **Orange** — Soutien actif + ressources (Jeunesse J'écoute : 1-800-668-6868)
- **Rouge** — Orientation urgente, évaluation de la sécurité, 911 si nécessaire

---

## 3. Architecture technique (1 minute)

```
Message utilisateur
      │
      ▼
┌─────────────────────────────────────────┐
│  COUCHE 1 — Règles lexicales (priorité) │
│  • 9 mots critiques  → plancher Orange  │
│  • 12 mots de détresse → plancher Jaune │
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
└─────────────────────────────────────────┘
```

**Aucune donnée ne quitte l'appareil.** Le LLM tourne en local via Ollama.

---

## 4. Démonstration des résultats (1 minute)

### Dataset
- **120 conversations synthétiques** (24 initiales + 96 générées via Claude Haiku)
- Distribution équilibrée : 30 conversations × 4 classes
- Authentiques : québécismes, dérive graduelle, 12 échanges par conversation

### Métriques du modèle

| Métrique | Valeur |
|---|---|
| CV Accuracy (k=4, stratifié) | **75.8% ± 6.0%** |
| Train accuracy | 100% (attendu, 120 samples) |
| Features | 42 (7 × 6 statistiques temporelles) |

### Top features discriminantes

1. **`longueur_mots_slope`** — Le raccourcissement progressif des messages est le signal le plus fiable
2. **`score_negatif_mean`** — L'accumulation de sentiment négatif sur toute la conversation
3. **`presence_question_slope`** — La disparition des questions (perte d'engagement vers l'avenir)

> Ces features correspondent exactement aux patterns cliniques documentés dans la littérature sur l'idéation suicidaire et l'isolement social chez les jeunes.

---

## 5. Impact et prochaines étapes (1 minute)

### Impact potentiel
- **Cible** : plateformes de soutien émotionnel IA (chatbots scolaires, apps bien-être)
- **Cas d'usage** : alerte discrète vers un humain (conseiller, parent) avant la crise
- **Différentiel** : fonctionne entièrement en local — PIPEDA/Loi 25 compliant by design

### Ce que ce PoC démontre
- La trajectoire conversationnelle est un signal détectable avec des moyens légers (pas de LLM pour la détection)
- 75.8% de précision avec 120 exemples — scalable avec plus de données réelles
- Architecture modulaire : la couche DDEC se greffe sur n'importe quel chatbot existant

### Prochaines étapes
1. **Données réelles** — Partenariat avec cliniciens pour valider et annoter (avec consentement)
2. **Fenêtre temporelle** — Étendre le tracking au-delà de 7 jours, détecter les cycles hebdomadaires
3. **Calibration clinique** — Ajuster les seuils avec des psychologues pour minimiser les faux négatifs niveau rouge
4. **Intégration** — API REST légère pour connexion à n'importe quelle plateforme

---

## En résumé

> DDEC détecte ce qu'un humain verrait en relisant une conversation :
> **la dérive**, pas juste le cri d'alarme.

**Stack** : Python · sklearn · Ollama · Streamlit · API Claude (génération de données)
**Repo** : `/home/dom/Documents/ddec-hackathon/`
**Contact** : [votre info]
