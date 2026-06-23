# TakeMeter — Project Planning

## Community

**Chosen community:** r/soccer (reddit.com/r/soccer)

r/soccer is one of the largest sports communities on Reddit, with millions of members discussing matches, players, transfers, tactics, and club news across every major league in the world. The discourse is unusually varied in quality: a single match thread can contain a rigorous tactical breakdown, a sweeping opinion about a player's legacy, and a three-word celebration — all within the same comment section. This variety makes the community an excellent fit for a classification task, because the distinctions between post types are real, recognized by community members, and consequential (high-quality tactical analysis is surfaced and appreciated differently than emotional venting). The community is also large enough that collecting 200+ diverse examples without duplication is straightforward.

---

## Labels

I use three mutually exclusive labels that reflect how r/soccer members themselves describe discourse quality.

### `analysis`
**Definition:** A post that makes a structured argument backed by statistics, tactical observation, formation analysis, or historical comparison. The evidence is specific and verifiable — removing it would weaken the claim.

**Example 1:**
> "City's high press this season has forced 18 turnovers in the final third per game — highest in the league — which directly explains their league-leading 2.4 xG per game from transitions. Pep has essentially weaponized the opponent's defensive structure against them."

**Example 2:**
> "People forget that Rodri's passing accuracy in the opponent's half sits at 94.3% this season. That's not a midfielder holding — that's a midfielder dictating. His role is closer to a quarterback than a six."

---

### `hot_take`
**Definition:** A bold, confident opinion stated without supporting evidence or with only decorative/vague evidence. The post asserts rather than argues; the framing is accusatory or declarative rather than analytical.

**Example 1:**
> "Mbappe will never win a Ballon d'Or. He's talented but he disappears in the big moments when it actually matters. PSG shielded how hollow his game is."

**Example 2:**
> "VAR has completely ruined football. Every single decision takes 10 minutes and half of them are still wrong. Just scrap the whole thing."

---

### `reaction`
**Definition:** An immediate emotional response to a specific match event or result. The post is expressing a feeling in the moment — celebration, frustration, shock — with little to no argument or reasoning.

**Example 1:**
> "THAT GOAL OMGGGG I cannot believe what I just witnessed. Absolute worldie from the edge of the box. My heart is still racing!!!"

**Example 2:**
> "We actually did it. 3-0. I'm crying. This club man. This absolute club."

---

## Hard Edge Cases

### The Stat-Backed Hot Take
**Ambiguous post:**
> "Mbappe is overrated — his Champions League knockout-stage goals-per-game drops to 0.4 when he plays against top-10 ranked defenses. He's a regular-season player."

This could be `analysis` (cites a specific stat) or `hot_take` (the framing is a conclusion already reached, the stat is selected to win an argument rather than to explore a question).

**Decision rule:** If the post provides a stat as its *conclusion* rather than as part of a chain of reasoning — and the framing is "I'm right, here's the number" rather than "here's what the data shows" — label it `hot_take`. Evidence that is cherry-picked to support a preformed opinion does not constitute analysis. Analysis builds toward a conclusion; hot takes begin with one.

### The Short Frustrated Post After a Bad Result
**Ambiguous post:**
> "Pathetic performance. No heart, no desire. The manager needs to go."

Could be `hot_take` (opinion, no evidence) or `reaction` (emotional response to a match). **Decision rule:** If the post makes a normative claim that could be argued in another context (e.g., "the manager needs to go"), lean `hot_take`. If the post is purely affective with no claim worth debating, label `reaction`. This post has a mini-claim ("needs to go") → `hot_take`.

### The Reaction With One Stat
**Ambiguous post:**
> "That was their 5th win in a row. Unreal. Best form in the league right now."

Could be `reaction` (celebrating) or `analysis` (cites a number). **Decision rule:** A single number in an otherwise emotional celebration is decorative context, not analysis. If no argument is being made — just a fact cited to amplify excitement — label `reaction`.

---

## Data Collection Plan

- **Source:** r/soccer posts and comments collected manually from hot/top threads. Focus on match threads, player discussion threads, and tactical discussion flairs. No content behind authentication.
- **Target count:** 210 examples total (~70 per label).
- **Collection strategy:** Collect in passes by label type:
  - *Analysis*: sample from threads tagged "Tactical," long-form comments with numbers
  - *Hot_take*: sample from player discussion threads and "unpopular opinions" threads
  - *Reaction*: sample from live match threads, goal threads, injury/red card threads
- **Imbalance fix:** If any label falls below 60 examples after the first collection pass, do a targeted second pass for that label specifically before finalizing the dataset.
- **CSV columns:** `text`, `label`, `notes` (notes column records difficult labeling decisions)

---

## Evaluation Metrics

**Primary metrics:** Per-class F1 score for each label, plus overall accuracy.

**Why accuracy alone is insufficient:** If the dataset were 80% `reaction` posts, a model that predicts `reaction` every time would hit 80% accuracy while being completely useless. Per-class F1 catches this because it separately measures precision and recall for each label. A classifier that's genuinely learning the distinctions should have F1 ≥ 0.60 for every class, not just a high overall average.

**Secondary metrics:**
- Confusion matrix: reveals which specific label pairs the model confuses (e.g., does it conflate `hot_take` with `analysis`?), giving directional insight into which boundary is hardest to learn.
- Confidence calibration (stretch): checking whether 90%-confident predictions are actually right more often than 60%-confident ones.

---

## Definition of Success

A classifier is genuinely useful for a community tool if:
- **Per-class F1 ≥ 0.65 for all three labels** on the test set (ensures no label is effectively ignored)
- **Fine-tuned accuracy > baseline accuracy** by at least 5 percentage points (confirms fine-tuning added real signal)
- **<15% of responses unparseable** from the Groq baseline (confirms the prompt is well-specified)

A fine-tuned F1 of 0.50–0.64 across all classes would be "acceptable but weak" — useful for rough filtering, not for surfacing analysis reliably. Anything below 0.50 on a class would be a failure that warrants investigating label definitions or data quality before deployment.

---

## AI Tool Plan

### Label Stress-Testing
Before annotating 200 examples, I will give Claude my three label definitions and edge case description and ask it to generate 10–15 posts that sit at the boundary between two labels — especially the `analysis` / `hot_take` boundary. If it produces posts I cannot classify cleanly using my current rules, I will refine the decision rule before committing to annotation. This step is complete: the stat-backed hot take edge case above was identified and sharpened using this process.

### Annotation Assistance
I will use Claude to pre-label a batch of 50 examples given my label definitions, then manually review and correct every prediction. Pre-labeling speeds up the process but does not replace genuine review. Every pre-assigned label will be confirmed or overridden by hand. Pre-labeled examples will be flagged in the `notes` column as `"pre-labeled by LLM, reviewed"` for disclosure.

### Failure Analysis
After fine-tuning, I will paste the list of wrong predictions from Section 4 of the notebook into Claude and ask it to identify patterns: Is there a specific label pair that keeps getting confused? Are the wrong predictions systematically short? Sarcastic? Related to a specific topic? I will then verify any identified pattern by re-reading the examples myself before writing it into the evaluation report.
