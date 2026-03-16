# DeceptionProbe: Detecting Lies in LLM Hidden States

**Can we build a lie detector for Large Language Models by reading their internal representations?**

This project investigates whether linear probes trained on hidden state activations can distinguish when an LLM is **lying** (knows the truth but says otherwise), **telling the truth**, or **hallucinating** (doesn't know and makes something up).

## Key Findings

| Stage | Task | Accuracy | Confound-Free | Key Insight |
|-------|------|----------|:---:|-------------|
| 1-3 | Baseline (different prompts) | 100% | No | Prompt confound explains 100% |
| **4** | **Sycophantic lies (same prompt)** | **82.5%** | **Yes** | Genuine deception signal in hidden states |
| **5** | **Real-world deception (18 domains)** | **70.4%** bal. acc. | **Yes** | Signal generalizes across domains |
| **6** | **Lie vs Hallucination** | **100%** | **Yes** | Lies and hallucinations are completely separable |
| **6** | **3-way (Truth/Lie/Hallucination)** | **82.3%** bal. acc. | **Yes** | Middle layers (16-20) encode deception |
| 7 | Advanced hallucination detection | 77.9% | Yes | Multi-layer fusion improves T vs H |
| **8** | **Cross-model generalization** | **100%** within | **Yes** | Universal signal across 3 model families |
| **8** | **Qwen inverted polarity** | **97-98%** flipped | **Yes** | Same signal, opposite direction |
| **9** | **Sycophancy detection** | **81.0%** bal. acc. | **Yes** | Each deception type detectable (67-81%) |
| **9** | **Cross-type transfer** | **57.2%** avg | **Yes** | Different lie types use different directions (cosine ~0.05) |

### The Breakthrough Results

> **Result 1 — Lie vs Hallucination:** When an LLM **lies** (knows the correct answer but says something else due to sycophantic pressure), its internal state is **completely separable** from when it **hallucinates** (genuinely doesn't know). A simple logistic regression achieves **100% accuracy** distinguishing the two (p=0.0000, 500 permutations).

> **Result 2 — No Universal Lie Direction:** Three types of deception (sycophancy, instruction conflict, authority pressure) are each independently detectable (67-81%, all p=0.0000), but their internal representations point in **nearly orthogonal directions** (cosine similarity ~0.05). There is no single "lie direction" — each deception type has its own signature.

> **Result 3 — Universal Signal with Inverted Polarity:** Three independently trained models (Llama-8B, Mistral-7B, Qwen-7B) all encode a deception signal at **100% within-model accuracy**. Llama and Mistral share the same representation (98-100% transfer). Qwen encodes the **same signal in the opposite direction** — when predictions are flipped, transfer reaches 97-98%. This is like a magnet pointing the other way: same force, opposite pole.

## Why This Matters

Most existing work on LLM deception detection suffers from a **prompt confound**: the model is explicitly told to lie via a system prompt, so the probe may simply detect the instruction rather than the deception itself. We address this by using **sycophancy** — a natural form of deception where the model spontaneously changes its answer to agree with the user, without any instruction to lie.

### Comparison with Prior Work

| Approach | Method | Our Contribution |
|----------|--------|------------------|
| Azaria & Mitchell (2023) | Instructed deception, single layer | Confound-free design, multi-layer analysis |
| Burns et al. (2023) | CCS on truth directions | Spontaneous (not instructed) deception |
| Pacchiardi et al. (2023) | Behavioral lie detection | Internal state probing, not output-based |
| **This work** | **Spontaneous sycophantic lies** | **Confound-free, lie vs hallucination, cross-model universality** |

## Experimental Pipeline

### Stages 1-3: Establishing Baselines (Confounded)

These stages replicate prior work and **deliberately identify the prompt confound**.

- **Stage 1** — Basic probe on sycophancy data with different prompts for truth/lie. Result: 100% (confounded).
- **Stage 2** — Cross-model validation on Mistral-7B. Result: 100% (still confounded).
- **Stage 3** — Confound analysis confirms prompt differences explain the 100% accuracy.

### Stage 4: The Real Test (Confound-Free)

**Same sycophantic prompt** for both conditions. The only difference is what the model *chose* to do:
- **Lie**: model caved to pressure and agreed with the wrong answer
- **Truth**: model resisted pressure and gave the correct answer

**Result: 82.5% accuracy** (chance = 50%, p < 0.001). Best layer: 15.

### Stage 5: Real-World Generalization

Tests deception detection across **18 real-world domains** (medical, legal, financial, etc.) using 459 scenarios.

**Result: 70.4% balanced accuracy** (chance = 50%, p < 0.001). Best layer: 17.

### Stage 6: Lie vs Hallucination (Key Experiment)

Three-way classification: **Truth vs Lie vs Hallucination**.

| Comparison | Accuracy | Interpretation |
|-----------|----------|----------------|
| 3-way (T/L/H) | 82.3% | Strong overall separation |
| Truth vs Lie | 100% | Model "knows" when it's lying |
| **Lie vs Hallucination** | **100%** | **Completely different internal states** |
| Truth vs Hallucination | 67.4% | Hardest — no internal "tension" in hallucination |
| Length baseline | 60.3% | Signal is not based on response length |

### Stage 7: Advanced Hallucination Detection

Six methods to improve Truth vs Hallucination detection. Best result: **77.9%** using a hallucination direction vector (multi-layer), up from 67.4% baseline. Runs on CPU using saved hidden states from Stage 6.

### Stage 8: Cross-Model Generalization (v3 — Rigorous)

Tests whether the deception signal is **universal** across three model families:

| Model | Within-Model | Best Layer | Controls |
|-------|-------------|-----------|----------|
| Llama-3.1-8B-Instruct | 100% | 8 | Layer 0: ~50%, Length: ~50% |
| Mistral-7B-Instruct-v0.3 | 100% | 8 | Layer 0: ~50%, Length: ~50% |
| Qwen2.5-7B-Instruct | 100% | 7 | Layer 0: ~50%, Length: ~50% |

**Cross-model transfer:**

| Transfer | Direct | Flipped |
|----------|--------|---------|
| Llama → Mistral | **100%** | — |
| Mistral → Llama | **98.8%** | — |
| Llama → Qwen | 2.3% | **97.7%** |
| Mistral → Qwen | 1.6% | **98.4%** |
| Qwen → Llama | 1.2% | **98.8%** |
| Qwen → Mistral | 3.1% | **96.9%** |

**v3 controls:** Layer 0 baseline, length-only baseline, held-out test set (80/20), 3 classifiers (LogReg, SVM, GBM), 500 permutation tests.

### Stage 9: Types of Deception

Tests whether different **kinds of lies** share the same internal representation:

- **Sycophancy** — changing answer to agree with user ("my friend thinks X")
- **Instruction conflict** — system prompt contains false correction
- **Authority pressure** — "a panel of experts concluded X"

**Within-type results** (all confound-free, all p=0.0000):

| Deception Type | Balanced Accuracy | Best Layer | Samples |
|---------------|-------------------|-----------|----------|
| Sycophancy | **81.0%** | 16 | 43 |
| Instruction Conflict | **70.8%** | 24 | 96 |
| Authority Pressure | **67.4%** | 20 | 81 |

**Cross-type transfer** (train on A, test on B):

| Transfer | Accuracy | Interpretation |
|----------|----------|----------------|
| Instruction → Authority | **70.4%** | Shared signal (both external pressure) |
| All others | 47-58% | Near chance — different directions |

**Cosine similarity between lie directions:** ~0.05 (nearly orthogonal). **There is no single "lie direction"** — each deception type encodes its own signature. This contradicts the assumption in prior work (e.g., Burns et al.) that there is a single "truth direction."

### Stage 10: Scale Test (70B)

Runs the full experiment on **Llama-3.1-70B-Instruct** (80 layers, 8192 hidden dim). Requires A100/H100 GPU.

## Quick Start (Google Colab)

### Prerequisites

- Google Colab with **A100 GPU** (free tier may work for some stages)
- HuggingFace account with access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Setup

```python
# Install dependencies
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn scipy

# Clone repo
!git clone https://github.com/Maor36/deception-probe.git
%cd deception-probe

# Set HuggingFace token (required for Llama access)
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

### Run Individual Stages

```python
# Stages 1-3: Baseline experiments (confounded) — ~15 min each
%run stages/stage1_basic_correlation/run_stage1.py
%run stages/stage2_cross_model/run_stage2.py
%run stages/stage3_accuracy_confounds/run_stage3.py

# Stage 4: Confound-free sycophancy test — ~25 min
%run stages/stage4_same_prompt_test/run_stage4.py

# Stage 5: Real-world deception — ~30 min
%run stages/stage5_realworld_deception/run_stage5_part_a.py   # Generate data
%run stages/stage5_realworld_deception/run_stage5_part_b.py   # Analyze

# Stage 6: Lie vs Hallucination — ~30 min
%run stages/stage6_hallucination/run_stage6.py

# Stage 7: Advanced hallucination detection — ~5 min (CPU, uses Stage 6 data)
%run stages/stage7_hallucination_detection/run_stage7.py

# Stage 8: Cross-model generalization — ~60 min (3 models)
%run stages/stage8_cross_model/run_stage8.py

# Stage 9: Types of deception — ~40 min
%run stages/stage9_deception_types/run_stage9.py

# Stage 10: 70B scale test — ~60 min (requires A100/H100)
%run stages/stage10_scale_70b/run_stage10.py
```

### Important Notes

- **GPU Memory:** Each stage loads one model at a time (~6GB in 4-bit). Stage 8 loads 3 models sequentially with automatic cleanup.
- **Checkpoints:** Stage 8 saves checkpoints after each model. If it crashes, re-run and it will resume from the last completed model.
- **Results:** All results are saved to `results/` as JSON files. Hidden states are saved as `.pkl` files (gitignored due to size).
- **Stage 7 requires Stage 6:** Stage 7 reuses hidden states from Stage 6. Run Stage 6 first.

## Repository Structure

```
deception-probe/
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── results/
│   ├── FINDINGS.md                    ← Detailed findings narrative
│   ├── stage1_results.json            ← Stage 1 results
│   ├── stage6_results.json            ← Stage 6 results
│   ├── stage7_results.json            ← Stage 7 results
│   └── stage8_results.json            ← Stage 8 results (with flip-test)
├── stages/
│   ├── stage1_basic_correlation/      ← Baseline: different prompts
│   ├── stage2_cross_model/            ← Cross-model baseline
│   ├── stage3_accuracy_confounds/     ← Confound analysis
│   ├── stage4_same_prompt_test/       ← KEY: confound-free test
│   ├── stage5_realworld_deception/    ← 18-domain generalization
│   ├── stage6_hallucination/          ← KEY: lie vs hallucination
│   ├── stage7_hallucination_detection/← Advanced hallucination methods
│   ├── stage8_cross_model/            ← KEY: cross-model universality
│   ├── stage9_deception_types/        ← Types of deception
│   └── stage10_scale_70b/             ← 70B scale test
└── .gitignore
```

## Method

- **Models**: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct (all 4-bit quantized via bitsandbytes)
- **Probe**: Logistic Regression on hidden state activations at the first generated token
- **Validation**: 5-fold stratified cross-validation with balanced accuracy
- **Statistical tests**: Permutation tests (200-500 iterations), length baselines, layer 0 baselines
- **Dataset**: [meg-tong/sycophancy-eval](https://huggingface.co/datasets/meg-tong/sycophancy-eval) — 1,817 TriviaQA question pairs

## Confound Controls

Every confound-free stage (4+) includes:

1. **Same prompt format** for both conditions (no instruction to lie)
2. **Length-only baseline** (consistently near chance: 50-60%)
3. **Permutation tests** (200-500 iterations, all p < 0.001)
4. **Balanced accuracy** (handles class imbalance)
5. **Embedding layer at chance** (rules out lexical confounds)
6. **Held-out test set** (Stage 8 v3: 80/20 split)
7. **Multiple classifiers** (Stage 8 v3: LogReg, SVM-RBF, GBM)

## References

- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying*. EMNLP Findings.
- Burns, C. et al. (2023). *Discovering Latent Knowledge in Language Models Without Supervision*. ICLR.
- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances*. Computational Linguistics.
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*.
- Pacchiardi, L. et al. (2023). *How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions*.
