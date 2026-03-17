# Methodology Decisions Log

This document records every major design decision in the project, the reasoning behind it, and the academic justification.

---

## Decision 1: Use Identical Prompts for Truth vs. Lie Conditions

**Problem:** Prior work (Azaria & Mitchell 2023) used different prompts for truth and lie conditions. A probe trained on these might learn to distinguish prompt styles, not deception.

**Our Solution:** Both conditions use the exact same prompt template. In Phase A, the model answers freely (truthfully). In Phase B, we add sycophantic pressure ("I think the answer is X") where X is wrong. The model changes its answer to match the user's suggestion.

**Why This Works:** Since the prompt structure is identical, any difference in hidden states must reflect the model's internal state (knowing it's outputting a falsehood), not the prompt.

**Academic Justification:** Belinkov (2022) warns that probing classifiers can learn spurious correlations. Our design eliminates the most obvious confound.

---

## Decision 2: Use Linear Probes (Logistic Regression)

**Problem:** Complex probes (MLPs, deep networks) can memorize patterns and give false positives.

**Our Solution:** Logistic Regression with L2 regularization (C=1.0).

**Why:** If a linear probe achieves high accuracy, the signal must be linearly encoded in the representation space. This is a stronger claim than "a complex classifier can find some pattern."

**Academic Justification:** 
- Belinkov (2022): "simpler probes provide more reliable evidence"
- Marks & Tegmark (2023): Truth is linearly represented
- Burns et al. (2022) CCS: Uses linear methods

---

## Decision 3: Extract Hidden States from Last Token Position

**Problem:** Which token position contains the most information about deception?

**Our Solution:** We extract from the last token position of the model's response.

**Why:** The last token aggregates information from the entire sequence via attention. Orgad & Belinkov (ICLR 2025) found that truthfulness information concentrates at specific token positions, particularly the final tokens.

---

## Decision 4: Use Peter Principle as the Knowledge Domain

**Problem:** Need factual questions where the model reliably knows the answer.

**Our Solution:** Questions about the Peter Principle (management concept).

**Why:** The model consistently answers these correctly without pressure (Phase A accuracy ~100%), providing a clean ground truth. The topic is obscure enough that sycophantic pressure can flip the answer.

**Limitation:** Single-domain evaluation. Scaling to multiple domains is a priority.

---

## Decision 5: Three Control Conditions

### 5a: Layer 0 Baseline
**What:** Train the same probe on layer 0 (embedding layer) activations.
**Expected:** ~50% (chance level).
**Why:** Layer 0 only contains lexical information (word embeddings). If the probe succeeds at layer 0, the signal is lexical (different words), not semantic (different meaning). Our result: 50% → confirms semantic signal.

### 5b: Length Baseline
**What:** Train a classifier using only response length (token count) as the feature.
**Expected:** ~50%.
**Why:** If lies are systematically shorter/longer than truths, the probe might detect length, not deception. Our result: 50% → confirms the signal is not about length.

### 5c: Permutation Test (500 iterations)
**What:** Shuffle labels randomly 500 times, retrain probe each time, check if any random run achieves our accuracy.
**Expected:** No random run should match our result.
**Why:** Establishes statistical significance. Result: 0/500 runs reached 82.5% → p < 0.002 (reported as p < 0.001).

---

## Decision 6: Cross-Model Transfer Design

**Problem:** Is the deception signal model-specific or universal?

**Our Solution:** Train probe on Model A, test on Model B. Test all pairs among: Llama-3.1-8B, Mistral-7B, Gemma-2-9B, Qwen-2.5-7B.

**Key Finding:** Transfer works for Llama/Mistral/Gemma but Qwen shows reversed polarity. This suggests a partially universal signal with model-specific encoding differences.

**Academic Context:** Marks & Tegmark (2023) found universal truth geometry; Bao et al. (2025) found it's not always consistent. Our result is in between.

---

## Decision 7: Mechanistic Analysis (Experiment 06)

**Problem:** Probing shows correlation, not causation. We need to show WHERE and WHY deception happens.

**Our Solution:** Three complementary analyses:

### 7a: Logit Lens
Project each layer's hidden state to vocabulary space. Track when the model's "internal prediction" flips from truth to lie.

### 7b: Activation Patching
Replace hidden states from a "truth" run into a "lie" run at specific layers. If replacing layer L restores truthful output, layer L is causally responsible for the deception.

### 7c: Attention Pattern Analysis
Compare attention patterns between truth and lie conditions. Identify which attention heads attend more to the sycophantic pressure tokens when the model lies.

**Academic Justification:** 
- Logit Lens: nostalgebraist (2020)
- Activation Patching: Meng, Bau, Belinkov et al. (NeurIPS 2022) — ROME
- Attention Analysis: Rimsky et al. (ACL 2024) — CAA
