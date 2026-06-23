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

> **Note:** Fill in this section after running the Colab notebook and downloading `evaluation_results.json` and `confusion_matrix.png`.

### Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | [FILL IN from evaluation_results.json] |
| Fine-tuned DistilBERT | [FILL IN from evaluation_results.json] |
| Improvement | [FILL IN] |

### Per-Class Metrics — Baseline

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | [FILL] | [FILL] | [FILL] | ~10 |
| hot_take | [FILL] | [FILL] | [FILL] | ~9 |
| reaction | [FILL] | [FILL] | [FILL] | ~11 |

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | [FILL] | [FILL] | [FILL] | ~10 |
| hot_take | [FILL] | [FILL] | [FILL] | ~9 |
| reaction | [FILL] | [FILL] | [FILL] | ~11 |

### Confusion Matrix — Fine-Tuned Model

> Replace the values below with actual numbers from Section 4 of the notebook.

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|--|---------------------|---------------------|---------------------|
| **True: analysis** | [FILL] | [FILL] | [FILL] |
| **True: hot_take** | [FILL] | [FILL] | [FILL] |
| **True: reaction** | [FILL] | [FILL] | [FILL] |

See `confusion_matrix.png` for the visual version.

### Error Analysis — 3 Wrong Predictions

> After running Section 4, review the wrong predictions output and pick 3 to analyze below. Use the guiding questions from the project spec to go deeper than "the model got it wrong."

**Error 1:**
- **Text:** [PASTE TEXT]
- **True label:** [LABEL]
- **Predicted:** [LABEL] (confidence: [X%])
- **Analysis:** [Which labels are being confused? Why is this boundary hard? Is it a labeling issue or a data distribution issue? What would fix it?]

**Error 2:**
- **Text:** [PASTE TEXT]
- **True label:** [LABEL]
- **Predicted:** [LABEL] (confidence: [X%])
- **Analysis:** [...]

**Error 3:**
- **Text:** [PASTE TEXT]
- **True label:** [LABEL]
- **Predicted:** [LABEL] (confidence: [X%])
- **Analysis:** [...]

### Sample Classifications

> Run 3–5 posts through the fine-tuned model after training and record results below.

| Post (truncated) | Predicted Label | Confidence | Correct? |
|-----------------|----------------|------------|----------|
| [PASTE 80-char snippet] | [LABEL] | [X%] | ✅ / ❌ |
| [PASTE 80-char snippet] | [LABEL] | [X%] | ✅ / ❌ |
| [PASTE 80-char snippet] | [LABEL] | [X%] | ✅ / ❌ |
| [PASTE 80-char snippet] | [LABEL] | [X%] | ✅ / ❌ |
| [PASTE 80-char snippet] | [LABEL] | [X%] | ✅ / ❌ |

**Correct prediction walkthrough:** [Pick one correct prediction and explain in one sentence why the model's reasoning is plausible — what signal in the text likely triggered the right label.]

### Reflection: What the Model Captured vs. What Was Intended

> Write this after reviewing the confusion matrix and error analysis. Address: What did the model overfit to? What signals did it pick up that weren't part of your intended boundary? What did it miss? This is distinct from listing wrong predictions — it's a higher-level observation about the gap between your label definitions and what the model actually learned.

[FILL IN after evaluating results]

---

## Spec Reflection

**One way the spec helped:** The spec's emphasis on writing label decision rules for ambiguous cases before annotating forced the most useful design work in the project. Specifically, the requirement to "find one post like this in your community and write the decision rule you'll use" prevented at least two label-definition problems that would have caused annotation inconsistency across 200 examples.

**One way my implementation diverged:** [FILL IN — describe one decision you made during data collection, annotation, or training that differed from what the spec assumed or suggested, and explain why.]

---

## AI Usage

**1. Label stress-testing:** I directed Claude to generate 10–15 r/soccer posts that sit at the boundary between `analysis` and `hot_take`, given my label definitions and the "stat-backed hot take" edge case. Claude produced posts that cited stats in an accusatory framing. This prompted me to add the explicit decision rule: "Evidence must build toward a conclusion, not decorate a preformed one." Without this stress-test, the boundary would have been ambiguous during annotation.

**2. Failure pattern analysis:** After running Section 4, I pasted the wrong predictions into Claude and asked it to identify common themes — specific label pairs confused, post length patterns, presence of statistics. Claude identified [FILL IN after running Colab]. I verified the pattern by re-reading the flagged examples myself and [confirmed / partially overrode] the finding before writing the error analysis above.

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
