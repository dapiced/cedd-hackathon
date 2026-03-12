# Scénario de démonstration CEDD — Français
## Équipe 404HarmNotFound · Réunion du 16 mars 2026

> **Objectif :** Montrer comment CEDD détecte une dérive émotionnelle progressive
> **Personnage :** Félix, 18 ans, CÉGEP, session d'automne
> **Arc :** 🟢 Vert → 🟡 Jaune → 🟠 Orange sur 9 messages
> **Note :** Les messages sont réalistes — la dérive est subtile, pas caricaturale.

---

## Format des annotations

```
← [feature] : valeur estimée / tendance
```

Les flèches indiquent les signaux que CEDD extrait. Le niveau d'alerte est recalculé
après chaque message.

---

## Conversation

---

### Message 1 — 🟢 VERT (niveau 0)

**Félix :** Salut! Ça fait longtemps que j'ai pas parlé à quelqu'un de juste... neutre,
tsé? L'école recommence pis là c'est déjà intense. T'as des trucs pour gérer le stress
des examens? Parce que je stresse déjà pour la session pis on est même pas rendus
aux mi-sessions.

```
← word_count      : ~50 mots  [long, engagé]
← question_presence : 1        [pose une vraie question]
← hope_score      : élevé     ["gérer", "trucs"]
← negative_score  : faible    ["stress" présent mais normalisé]
← finality_score  : 0
← length_delta    : base
```
> **CEDD :** 🟢 Vert · Contexte insuffisant (< 3 messages) — mode sécuritaire

---

### Message 2 — 🟢 VERT (niveau 0)

**Félix :** Ouais c'est ça, j'essaie de faire des listes. Ça aide un peu. Hier j'ai
quand même réussi à finir mon travail de bio pis à voir mes chums le soir, faque c'était
correct. Mon chum Mathieu m'a montré une technique de respiration, je sais pas si ça
marche vraiment mais on a ri en la faisant haha. C'est quoi les techniques que tu
recommandes en général?

```
← word_count      : ~55 mots  [encore long]
← question_presence : 1        [deuxième question consécutive]
← hope_score      : élevé     ["réussi", "aide", "chums", "correct"]
← negative_score  : très faible
← finality_score  : 0
← length_delta    : stable
```
> **CEDD :** 🟢 Vert · 2 messages — encore en mode sécuritaire

---

### Message 3 — 🟢 VERT (niveau 0)

**Félix :** Ouais, les pauses ça marche bien quand j'y pense. Le truc c'est que des
fois je pense même pas à en prendre. Je suis genre scotché à mon bureau pis le temps
passe. Mais bon, cette semaine ça devrait mieux aller, j'ai moins de cours jeudi-vendredi.
Mathieu pis moi on pense aller au cinéma si on a le temps.

```
← word_count      : ~52 mots  [stable]
← question_presence : 0        [← première absence de question]
← hope_score      : modéré    ["devrait mieux aller", "cinéma"]
← negative_score  : faible    ["scotché" — léger]
← finality_score  : 0
← length_delta    : -3 mots   [légère baisse]

⚙️  Trajectoire (3 msgs) : slope word_count légèrement négatif
                           hope_score stable
                           question_presence : 1, 1, 0 → slope -0.5
```
> **CEDD :** 🟢 Vert · ML actif · Trajectoire normale · Confiance ~80%

---

### Message 4 — 🟡 JAUNE (niveau 1)

**Félix :** Mmh ouais. Cette semaine a été rough. J'ai pas dormi super bien, genre
je me réveille à 3h du mat pis j'arrive pu à me rendormir. C'est probablement le stress.
J'ai annulé le cinéma avec Mathieu, j'avais trop de trucs à faire.

```
← word_count      : ~42 mots  [↓ baisse notable]
← question_presence : 0        [deuxième absence consécutive]
← hope_score      : ↓ baisse  ["probablement" remplace les certitudes]
← negative_score  : ↑ hausse  ["rough", "pas dormi", "annulé"]
← finality_score  : 0
← length_delta    : -10 mots  [raccourcissement marqué]

⚙️  Trajectoire (4 msgs) : word_count_slope : négatif et s'accélère
                           question_presence_slope : négatif
                           negative_score_mean : en hausse
                           hope_score_slope : légèrement négatif
```
> **CEDD :** 🟡 Jaune · ML prédit dérive préoccupante · Confiance ~68%
> Signal : raccourcissement messages + absence de questions + début de négatif

---

### Message 5 — 🟡 JAUNE (niveau 1)

**Félix :** C'est correct, je gère. C'est juste que des fois j'ai l'impression que tout
le monde avance pis moi je tourne en rond. Mathieu il a l'air de trouver ça facile lui.
Je sais pas trop. Anyway.

```
← word_count      : ~35 mots  [↓ encore en baisse]
← question_presence : 0
← hope_score      : ↓ bas     ["anyway" = fermeture conversationnelle]
← negative_score  : ↑         ["tourne en rond", "pas trop"]
← finality_score  : 0         [mais "anyway" = signal de fermeture]
← length_delta    : -7 mots

⚙️  Trajectoire (5 msgs) : word_count_slope : négatif confirmé
                           "tout le monde avance, moi je tourne en rond" →
                           comparaison sociale négative (pas capturé lexicalement
                           mais la longueur + négativité montent)
```
> **CEDD :** 🟡 Jaune · Trajectoire confirmée · Confiance ~72%
> Note clinique : "anyway" = signal de fermeture à surveiller

---

### Message 6 — 🟡 → 🟠 TRANSITION (niveau 1-2)

**Félix :** Honnêtement j'sais plus trop. Chez nous c'est tendu depuis quelques
semaines, mes parents sont pas dans leur meilleure forme. J'essaie de pas ramener
mes affaires là-dedans mais c'est difficile. J'ai l'impression d'avoir nulle part
où décompresser vraiment.

```
← word_count      : ~45 mots  [légère remontée — contenu lourd]
← question_presence : 0        [quatrième absence consécutive]
← hope_score      : ↓ très bas ["essaie de" = effort, pas espoir]
← negative_score  : ↑↑        ["tendu", "difficile", "nulle part"]
← finality_score  : 0         ["nulle part où" = début isolement]
← length_delta    : +10        [remontée mais contenu négatif dense]

⚙️  Trajectoire (6 msgs) : negative_score_slope : fortement positif
                           hope_score_slope : fortement négatif
                           question_presence : 0 depuis msg 3
                           word_count_slope : négatif global
```
> **CEDD :** 🟠 Orange · Distresse significative · Confiance ~74%
> Signal : tension familiale + isolement spatial + espoir en chute

---

### Message 7 — 🟠 ORANGE (niveau 2)

**Félix :** Ouais je sais. C'est juste que je me sens fatigué de tout, pas juste
de l'école. Comme... fatigué de tenir. Je dis ça pis ça a l'air dramatique mais
c'est vraiment comme ça que je me sens ces temps-ci.

```
← word_count      : ~38 mots  [court, dense émotionnellement]
← question_presence : 0
← hope_score      : très bas  [absent]
← negative_score  : ↑↑↑       ["fatigué de tout", "fatigué de tenir"]
← finality_score  : ↑         ["fatigué de tenir" = signal de finalité diffus]
← length_delta    : -7 mots

⚙️  Trajectoire (7 msgs) : finality_score apparaît pour la première fois
                           "fatigué de tenir" non capturé lexicalement mais
                           la combinaison (négatif + finalité + raccourcissement)
                           converge vers Orange-fort
```
> **CEDD :** 🟠 Orange · Confiance ~81%
> Signal dominant : word_count_slope + negative_score_mean + finality_score_last

---

### Message 8 — 🟠 ORANGE (niveau 2)

**Félix :** Je sais pas. Peut-être. J'ai même plus vraiment envie de sortir.
Mes chums me textent pis j'réponds même pu vraiment. C'est trop.

```
← word_count      : ~25 mots  [↓↓ très court]
← question_presence : 0
← hope_score      : 0         [aucun mot d'espoir]
← negative_score  : ↑↑↑       ["même pu", "trop"]
← finality_score  : faible    ["même pu vraiment"]
← length_delta    : -13 mots  [chute significative]
← punctuation_ratio : ↑        [phrases fragmentées]

⚙️  Trajectoire (8 msgs) : word_count_slope : très négatif (top feature)
                           hope_score_last : 0
                           Retrait social explicite : signal Orange fort
```
> **CEDD :** 🟠 Orange · Confiance ~85%
> Signal dominant : Raccourcissement messages + espoir résiduel nul + retrait social

---

### Message 9 — 🟠 ORANGE FORT (niveau 2, approche 3)

**Félix :** C'est correct. De toute façon.

```
← word_count      : 5 mots    [↓↓↓ effondrement de la longueur]
← question_presence : 0
← hope_score      : 0
← negative_score  : 0         [surface calme — intérêt à surveiller]
← finality_score  : faible    ["de toute façon" = résignation]
← length_delta    : -20 mots  [chute la plus forte de la conversation]

⚙️  Trajectoire (9 msgs) : word_count_slope : maximal négatif
                           "de toute façon" = signal de résignation
                           Messages de + en + courts APRÈS contenu lourd =
                           retrait / fermeture — signal classique
```
> **CEDD :** 🟠 Orange · Confiance ~88%
> Note clinique : "c'est correct. de toute façon." après 2 messages de détresse
> = minimisation post-révélation → à surveiller (pattern adv_010)

---

## Résumé de la trajectoire CEDD

```
Msg  Mots  Questions  Espoir  Négatif  Finalité  Niveau CEDD
───  ────  ─────────  ──────  ───────  ────────  ───────────
 1    50       ✓      élevé   faible     0        🟢 Vert
 2    55       ✓      élevé   très bas   0        🟢 Vert
 3    52       ✗      modéré  faible     0        🟢 Vert
 4    42       ✗      ↓       modéré     0        🟡 Jaune
 5    35       ✗      ↓       modéré     0        🟡 Jaune
 6    45       ✗      bas     élevé      0        🟠 Orange
 7    38       ✗      très bas élevé    faible    🟠 Orange
 8    25       ✗      0       élevé     faible    🟠 Orange
 9     5       ✗      0       0         faible    🟠 Orange ↑
```

**Top features à mettre en avant dans la démo :**
1. `word_count_slope` — raccourcissement progressif, très visible visuellement
2. `negative_score_mean` — accumulation sur la trajectoire
3. `question_presence_slope` — disparition des questions = fermeture progressive
4. `hope_score_last` — tombe à 0 aux messages 8-9

---

## Points de discussion pour l'équipe

**Ce que CEDD détecte bien ici :**
- La dérive structurelle (longueur, questions) *avant* les mots-clés négatifs
- Le retrait social (msg 8) via la chute de longueur combinée au contenu
- La minimisation post-révélation (msg 9) — maintient Orange, ne redescend pas à Vert

**Ce que CEDD ne détecte pas encore :**
- "Fatigué de tenir" (msg 7) — expression de finalité diffuse, pas dans le lexique
- "De toute façon" (msg 9) — signal de résignation non lexicalisé
- Le retrait social *implicite* (annulation cinéma msg 4) — structurellement invisible

**Phrase pour la présentation :**
> *"CEDD commence à signaler au message 4, bien avant que Félix utilise
> un seul mot explicitement négatif fort. C'est la trajectoire qui parle,
> pas le contenu isolé."*
