# TakeMeter — r/soccer Discourse Classifier

Fine-tuned text classifier that labels r/soccer posts as **analysis**, **hot_take**, or **reaction**, benchmarked against a Groq zero-shot baseline.

---

## Project Overview

This project builds a three-class text classifier for r/soccer discourse. The goal is to distinguish posts that make structured, evidence-backed arguments (`analysis`) from posts that state bold opinions without evidence (`hot_take`) from posts that express immediate emotional responses to match events (`reaction`). The classifier is trained by fine-tuning `distilbert-base-uncased` on a hand-labeled dataset of 200 r/soccer posts.

---

## Community and Label Taxonomy

**Community:** r/soccer (reddit.com/r/soccer) — one of Reddit's largest sports communities, with active discourse that spans rigorous tactical breakdowns, sweeping takes on players and managers, and real-time emotional reactions to match events. The variation in post quality and purpose makes it a strong candidate for classification.

### Labels

| Label | Definition |
|-------|-----------|
| `analysis` | A structured argument backed by statistics, tactical observation, historical comparison, or formation analysis. The evidence is specific and verifiable — removing it would weaken the claim. |
| `hot_take` | A bold, confident opinion stated without supporting evidence, or with only decorative/vague evidence. The post asserts rather than argues. |
| `reaction` | An immediate emotional response to a specific match event or result. The post expresses feeling in the moment with little to no argument or reasoning. |

### Why these distinctions matter in this community

r/soccer members routinely debate whether a comment is "actually an analysis" or "just a hot take." The community has its own informal norms around discourse quality, and these three labels capture the gradient that community members already recognize. A classifier over these labels could be used to surface high-quality tactical content, filter noise in large threads, or characterize how a community's discourse changes around specific events.

### Hard Edge Case Decision Rules

**Stat-backed hot take:** A post that cites one stat but uses it to assert a pre-formed conclusion → `hot_take`. Analysis *builds* toward a conclusion using evidence; a hot take *begins* with a conclusion and finds a number to support it. If the stat is cherry-picked or decorative rather than genuinely explanatory, it's a hot take.

**Short emotional post with a normative claim:** "Pathetic. The manager needs to go." The normative claim ("needs to go") makes this debatable, so → `hot_take`. A pure expression of feeling with no claim worth debating → `reaction`.

**Reaction with one number:** "That's their 5th win in a row. Unreal." The number adds context to a celebration but doesn't make an argument → `reaction`.

---

## Dataset

**Source:** 200 r/soccer-style posts collected and authored to represent realistic community discourse across all three label types.

**Labeling process:** Each post was read and assigned a label according to the definitions in `planning.md`. Borderline cases were resolved using explicit decision rules (documented in `planning.md` and the `notes` column of `dataset.csv`). A subset of posts was pre-labeled using an LLM before manual review and correction (see AI Usage section).

### Label Distribution

| Label | Count | % |
|-------|-------|---|
| `analysis` | 68 | 34% |
| `hot_take` | 60 | 30% |
| `reaction` | 72 | 36% |
| **Total** | **200** | **100%** |

### Hard Annotation Decisions

**1. Stat-backed hot take** — Post: *"Mbappe is overrated — his Champions League knockout-stage conversion rate drops significantly against top-10 ranked defenses. He's a regular-season player."* This cites a specific stat, which pulled toward `analysis`. Decision: `hot_take`. The post begins with a conclusion ("overrated") and the stat is selected to support that preformed view rather than to explore a question. The framing is accusatory, not investigative.

**2. Short rant with a normative claim** — Post: *"Pathetic performance. No heart, no desire. The manager needs to go."* The emotional tone pulled toward `reaction`. Decision: `hot_take`. The post makes a debatable normative claim ("needs to go") that could be argued in another context. A pure reaction expresses a feeling; this post makes a claim.

**3. Celebration with one stat** — Post: *"That was their 5th win in a row. Unreal. Best form in the league right now."* The number pulled toward `analysis`. Decision: `reaction`. The stat is decorative context for a celebration, not evidence for an argument. No claim is being made that the number genuinely supports — it's amplifying excitement, not reasoning.

---

## Fine-Tuning Setup

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training approach:** Standard sequence classification fine-tuning using HuggingFace `Trainer`. Dataset split 70% train / 15% validation / 15% test (stratified). Tokenization with `max_length=256` and dynamic padding via `DataCollatorWithPadding`.

**Hyperparameters:**

| Parameter | Value | Note |
|-----------|-------|------|
| `num_train_epochs` | 3 | Default; appropriate for a 140-example training set |
| `learning_rate` | 2e-5 | Standard BERT fine-tuning starting point |
| `per_device_train_batch_size` | 16 | Fits T4 GPU; no OOM observed |
| `weight_decay` | 0.01 | Light regularization for small dataset |
| `warmup_steps` | 50 | ~10% of training steps; stabilizes early training |

**Hyperparameter note:** No changes were made to the starter defaults. For a 200-example dataset, 3 epochs with a 2e-5 learning rate is well-supported by the literature on fine-tuning BERT-family models on small labeled sets. Increasing epochs beyond 3 on this dataset size risks overfitting.

---

## Groq Baseline Prompt

The zero-shot baseline used `llama-3.3-70b-versatile` via Groq with the following system prompt:

```
You are classifying posts from r/soccer.
Assign each post to exactly one of the following categories.

analysis: A post that makes a structured argument backed by statistics, tactical observation, historical comparison, or formation analysis. The evidence is specific and verifiable.
Example: "City's high press this season has forced 18 turnovers in the final third per game — highest in the league — which directly explains the 2.4 xG they're generating from transitions."

hot_take: A bold, confident opinion stated without supporting evidence. The claim might be true, but the post asserts rather than argues. May cite a vague or cherry-picked stat for effect.
Example: "Mbappe is overrated and will never win a Ballon d'Or. He just happens to play with the best players in the world."

reaction: An immediate emotional response to a specific match event or result. Little to no argument — the post is expressing a feeling in the moment.
Example: "THAT GOAL OMGGGG I cannot believe what I just witnessed. My heart is still racing!!!"

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction
```

---

## Evaluation Report

### Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | **1.000** |
| Fine-tuned DistilBERT | **0.867** |
| Difference | −0.133 (baseline outperformed fine-tuned) |

The baseline achieved perfect accuracy on the 30-example test set, which is the most important finding in this project. See the Reflection section for a full discussion.

### Per-Class Metrics — Baseline (Groq llama-3.3-70b-versatile)

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 1.000 | 1.000 | 1.000 | 10 |
| hot_take | 1.000 | 1.000 | 1.000 | 9 |
| reaction | 1.000 | 1.000 | 1.000 | 11 |

### Per-Class Metrics — Fine-Tuned DistilBERT

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 0.769 | 1.000 | 0.870 | 10 |
| hot_take | 1.000 | 0.556 | 0.714 | 9 |
| reaction | 0.917 | 1.000 | 0.957 | 11 |

*Precision for analysis = 10/(10+3) because 3 hot_takes were misclassified as analysis. Precision for reaction = 11/12 because 1 hot_take was misclassified as reaction. Hot_take recall = 5/9 because 4 of the 9 true hot_takes were missed.*

### Confusion Matrix — Fine-Tuned Model

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|--|---------------------|---------------------|---------------------|
| **True: analysis** | **10** | 0 | 0 |
| **True: hot_take** | 3 | **5** | 1 |
| **True: reaction** | 0 | 0 | **11** |

See `confusion_matrix.png` for the visual version. All 4 errors involve `hot_take` — the model never confuses `analysis` with `reaction` or vice versa.

### Error Analysis — 3 Wrong Predictions

All 4 errors were `hot_take` posts misclassified by the fine-tuned model. Three were predicted as `analysis`; one as `reaction`.

**Error 1 — hot_take → predicted analysis**
- **Text:** *"Mbappe is overrated — his Champions League knockout-stage conversion rate drops significantly against top-10 ranked defenses. He's a regular-season player."*
- **True label:** `hot_take` | **Predicted:** `analysis` (confidence: ~78%)
- **Analysis:** This is the clearest illustration of the model's core failure mode. The post contains a specific-sounding stat, which the model treats as a signal for `analysis`. But the stat is selected to support a preformed conclusion ("overrated") rather than to build toward one — the framing is accusatory, not investigative. The model learned "contains a number → analysis" rather than the intended distinction "evidence builds a conclusion vs. decorates one." This is a data distribution problem: the training set has many analysis posts with stats and many hot_take posts without stats, so the model anchored on statistical language as the primary signal. Fixing it would require more stat-containing hot_take examples in training to break that shortcut.

**Error 2 — hot_take → predicted analysis**
- **Text:** *"Klopp was never actually a great tactician — he just had Henderson and Wijnaldum running themselves into the ground for him. Give anyone those players pressing that hard and you'd get similar results."*
- **True label:** `hot_take` | **Predicted:** `analysis` (confidence: ~65%)
- **Analysis:** This post has a structural-argument tone — it makes a causal claim ("he just had X, which caused Y") that superficially resembles analytical reasoning. The model appears to have learned sentence structures associated with causal explanation as a proxy for analysis. The distinction the model missed is that the claim is unsupported — no data on press intensity, no comparison to other managers. The boundary here is genuine and difficult: persuasive-sounding prose without evidence is still a hot take. More examples of this pattern in training would help the model learn to look for evidence quality, not just argument structure.

**Error 3 — hot_take → predicted reaction**
- **Text:** *"Pathetic performance. No heart, no desire. The manager needs to go."*
- **True label:** `hot_take` | **Predicted:** `reaction` (confidence: ~71%)
- **Analysis:** This post's emotional register ("pathetic," "no heart") pulled the model toward `reaction`, even though it contains a normative claim ("the manager needs to go") that makes it debatable and therefore a hot take. The model learned emotional vocabulary as a strong signal for `reaction` — which is correct most of the time — but it failed to notice the mini-claim embedded in the emotional language. This is a genuine boundary difficulty: the post expresses feeling AND makes a claim. Short posts that do both are the hardest cases in this taxonomy. A decision rule making normative claims always qualify as hot_take regardless of emotional register is correct, but the model has no way to learn this rule from 200 examples — it needs to see many instances of emotional hot_takes to de-weight the emotional signal.

### Sample Classifications

| Post (truncated to 90 chars) | Predicted Label | Confidence | Correct? |
|-----------------------------|----------------|------------|----------|
| "City's high press this season has forced 18 turnovers in the final third per game..." | analysis | 96% | ✅ |
| "Haaland is overrated. He scores when the game is already won and disappears in tight..." | hot_take | 89% | ✅ |
| "LETSGOOOOO THAT'S THE WINNER IN THE 94TH MINUTE I AM GOING ABSOLUTELY MENTAL" | reaction | 99% | ✅ |
| "Mbappe is overrated — his Champions League knockout-stage conversion rate drops..." | analysis | 78% | ❌ (true: hot_take) |
| "I can't stop crying. We actually did it. 3-0. This club, man. This absolute club." | reaction | 97% | ✅ |

**Correct prediction walkthrough:** The model's 99% confidence on *"LETSGOOOOO THAT'S THE WINNER IN THE 94TH MINUTE..."* is entirely reasonable — all-caps text, exclamation, and the word "mental" are strong lexical signals for immediate emotional expression that the model learned reliably from the reaction examples in training.

### Reflection: What the Model Captured vs. What Was Intended

The confusion matrix tells a precise story: the fine-tuned model learned two of the three boundaries very well and completely failed on one.

**What it learned correctly:** The `analysis` / `reaction` boundary is perfectly captured — zero confusion between these two classes in either direction. The model correctly learned that detailed, structured prose belongs in one category and emotional, exclamatory text belongs in another. This is the easiest boundary and the model nailed it.

**What it missed:** The `hot_take` boundary is where the model broke down. The intended definition — "asserts without evidence" vs. "evidence builds a conclusion" — is a semantic distinction the model couldn't learn reliably from 200 examples. Instead, it learned a surface-level proxy: statistical language → analysis, emotional language → reaction. Hot_takes that sound analytical (contain a stat) get misclassified as analysis; hot_takes that sound emotional get misclassified as reaction. The model never learned what hot_take actually *is* — it learned what it *isn't*.

**Why this happened:** The `hot_take` category is defined by the *relationship* between claim and evidence, not by lexical features. DistilBERT, fine-tuned on 60 hot_take examples, doesn't have enough signal to learn this abstract semantic relationship — it overfits to surface tokens. A larger model with more training data could potentially learn it; alternatively, redefining hot_take to be more lexically distinct would make the classification task learnable at this scale.

**The baseline paradox:** The Groq baseline achieving 100% accuracy reveals that the task is not actually hard for a large language model with strong semantic understanding — it correctly applied the intended definitions zero-shot. The fine-tuned DistilBERT's underperformance is therefore not a task-difficulty problem; it's a model-capacity and data-size problem. The lesson: for semantically complex classification tasks with small datasets, a well-prompted large LLM may outperform a fine-tuned small model. Fine-tuning adds value when the task involves domain-specific jargon or when inference cost at scale matters more than peak accuracy.

---

## Spec Reflection

**One way the spec helped:** The spec's emphasis on writing label decision rules for ambiguous cases before annotating forced the most useful design work in the project. Specifically, the requirement to "find one post like this in your community and write the decision rule you'll use" prevented at least two label-definition problems that would have caused annotation inconsistency across 200 examples.

**One way my implementation diverged:** The spec assumes examples are collected by scraping or copy-pasting real posts from Reddit. In practice, the dataset examples were authored to ensure controlled label balance and deliberate coverage of edge cases (e.g., stat-backed hot takes, emotional posts with embedded claims). This produced a cleaner distribution than real scraping would have, but likely underrepresents naturalistic language variation — real posts have typos, slang, and mixed-language phrases that the synthetic dataset doesn't capture. This may partially explain why the fine-tuned model performs worse on nuanced cases than the zero-shot baseline, which was trained on far more naturalistic text.

---

## AI Usage

**1. Label stress-testing:** I directed Claude to generate 10–15 r/soccer posts that sit at the boundary between `analysis` and `hot_take`, given my label definitions and the "stat-backed hot take" edge case. Claude produced posts that cited stats in an accusatory framing. This prompted me to add the explicit decision rule: "Evidence must build toward a conclusion, not decorate a preformed one." Without this stress-test, the boundary would have been ambiguous during annotation.

**2. Failure pattern analysis:** After running Section 4, I pasted the wrong predictions into Claude and asked it to identify common themes across the 4 misclassified examples. Claude identified that all errors involved `hot_take` posts — specifically, that posts containing numbers or causal-sounding sentence structures were predicted as `analysis`, and posts with emotional vocabulary were predicted as `reaction`. I verified this pattern by re-reading the 4 misclassified examples myself and confirmed it: the model is using surface lexical features (presence of stats, emotional words) as proxies for the intended semantic boundary. This finding is the basis for the Reflection section above.

**3. Pre-labeling assistance:** A subset of dataset examples were pre-labeled by Claude given my label definitions. Each pre-assigned label was manually reviewed and corrected before being committed to `dataset.csv`. Pre-labeled examples are flagged with `"pre-labeled by LLM, reviewed"` in the `notes` column.

---

## How to Run

1. Open the [TakeMeter Colab notebook](https://colab.research.google.com/drive/1ilOny04QwR6CRUYLKvFycwzDsQLdPypI?usp=sharing) — click **File → Save a copy in Drive**
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU → Save)
3. Run **Section 1**: set `LABEL_MAP` (see below), upload `dataset.csv`
4. Run **Section 2**: verify split sizes and distribution
5. Run **Section 5**: add `GROQ_API_KEY` via Colab Secrets, paste the Groq prompt from this README
6. Run **Section 3**: fine-tune (~5–15 min on T4)
7. Run **Section 4**: evaluate and review wrong predictions
8. Run **Section 6**: download `evaluation_results.json` and `confusion_matrix.png`
9. Commit both files to this repo and fill in the placeholder sections above

**LABEL_MAP for the notebook:**
```python
LABEL_MAP = {
    "analysis": 0,
    "hot_take": 1,
    "reaction": 2,
}
```

---

## Repository Structure

```
TakeMeter/
├── README.md                  # This file — evaluation report and documentation
├── planning.md                # Design spec: labels, edge cases, data plan, AI tool plan
├── dataset.csv                # 200 labeled r/soccer examples (text, label, notes)
├── evaluation_results.json    # Downloaded from Colab after training
└── confusion_matrix.png       # Downloaded from Colab after training
```
