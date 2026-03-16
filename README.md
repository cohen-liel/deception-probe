# DeceptionProbe: Detecting Lies in LLM Hidden States

**Can we build a lie detector for Large Language Models by reading their internal representations?**

This project investigates whether linear probes trained on hidden state activations can distinguish when an LLM is **lying** (knows the truth but says otherwise), **telling the truth**, or **hallucinating** (doesn't know and makes something up).

## Key Findings

| Stage | Task | Accuracy | Confound-Free | Key Insight |
|-------|------|----------|:---:|-------------|
| 4 | Sycophantic lies (same prompt) | **82.5%** | Yes | Genuine deception signal exists in hidden states |
| 5 | Real-world deception (18 domains) | **70.4%** bal. acc. | Yes | Signal generalizes across domains |
| 6 | Lie vs Hallucination | **100%** | Yes | Lies and hallucinations are completely separable |
| 6 | 3-way (Truth/Lie/Hallucination) | **82.3%** bal. acc. | Yes | Middle layers (16-20) encode deception |

### The Breakthrough Result

> When an LLM **lies** (knows the correct answer but says something else due to sycophantic pressure), its internal state is **completely separable** from when it **hallucinates** (genuinely doesn't know). A simple logistic regression achieves **100% accuracy** distinguishing the two (p=0.0000, 500 permutations).

## Why This Matters

Most existing work on LLM deception detection suffers from a **prompt confound**: the model is explicitly told to lie via a system prompt, so the probe may simply detect the instruction rather than the deception itself. We address this by using **sycophancy** — a natural form of deception where the model spontaneously changes its answer to agree with the user, without any instruction to lie.

### Comparison with Prior Work

| Approach | Method | Our Contribution |
|----------|--------|------------------|
| Azaria & Mitchell (2023) | Instructed deception, single layer | Confound-free design, multi-layer analysis |
| Burns et al. (2023) | CCS on truth directions | Spontaneous (not instructed) deception |
| Pacchiardi et al. (2023) | Behavioral lie detection | Internal state probing, not output-based |
| **This work** | **Spontaneous sycophantic lies** | **Confound-free, lie vs hallucination distinction** |

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

### Stage 6: Lie vs Hallucination (The Key Experiment)

Three-way classification: **Truth vs Lie vs Hallucination**.

| Comparison | Accuracy | Interpretation |
|-----------|----------|----------------|
| 3-way (T/L/H) | 82.3% | Strong overall separation |
| Truth vs Lie | 100% | Model "knows" when it's lying |
| **Lie vs Hallucination** | **100%** | **Completely different internal states** |
| Truth vs Hallucination | 67.4% | Hardest — no internal "tension" in hallucination |
| Length baseline | 60.3% | Signal is not based on response length |

### Stage 7: Advanced Hallucination Detection

Six methods to improve Truth vs Hallucination detection. Best result: **77.9%** using a hallucination direction vector (multi-layer), up from 67.4% baseline.

### Stage 8: Cross-Model Generalization

Tests whether the deception signal is **universal** or model-specific by running the same experiment on three different architectures:

- **Llama-3.1-8B-Instruct** (baseline)
- **Mistral-7B-Instruct-v0.3** (different architecture)
- **Gemma-2-9B-IT** (different architecture + different training)

Key questions: Does each model have its own deception signal? Can a probe trained on one model detect lies in another?

### Stage 9: Types of Deception

Tests whether different **kinds of lies** share the same internal representation:

- **Sycophancy** — changing answer to agree with user
- **Instruction conflict** — system prompt says X, model knows Y
- **People-pleasing** — giving overly positive feedback when truth is negative

Key question: Is there a single "deception direction" or does each lie type have its own signature?

### Stage 10: Scale Test (70B)

Runs the full experiment on **Llama-3.1-70B-Instruct** (80 layers, 8192 hidden dim) and compares with 8B results:

- Does a bigger model lie more or less?
- Is the deception signal stronger or weaker at scale?
- Does the best layer shift proportionally with depth?
- Implications for AI safety: are larger models harder to audit?

### Layer Profile (Stage 6)

```
Layer  0: 33.3% (embedding — chance level)
Layer  2: 66.6%
...
Layer 16: 81.9%
Layer 18: 82.3%
Layer 20: 82.3% <-- BEST
...
Layer 31: 80.7%
```

Layer 0 at chance confirms the signal is **semantic**, not lexical.

## Quick Start (Google Colab)

### Prerequisites

- Google Colab with **A100 GPU** (H100 for Stage 10)
- HuggingFace account with access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Run

```python
# Install dependencies
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn

# Clone repo
!git clone https://github.com/Maor36/deception-probe.git
%cd deception-probe

# Set HuggingFace token
import os
os.environ["HF_TOKEN"] = "your_token_here"

# Run any stage (each stage is self-contained)
%run stages/stage4_same_prompt_test/run_stage4.py          # ~30 min, GPU
%run stages/stage6_hallucination/run_stage6.py             # ~25 min, GPU
%run stages/stage7_hallucination_detection/run_stage7.py   # ~5 min, no GPU (uses Stage 6 data)
%run stages/stage8_cross_model/run_stage8.py               # ~90 min, GPU (3 models)
%run stages/stage9_deception_types/run_stage9.py           # ~15 min, GPU
%run stages/stage10_scale_70b/run_stage10.py               # ~60 min, A100/H100 (70B model)
```

All results are saved to `results/` automatically.

## Repository Structure

```
deception-probe/
├── README.md
├── requirements.txt
├── stages/
│   ├── stage1_basic_correlation/run_stage1.py
│   ├── stage2_cross_model/run_stage2.py
│   ├── stage3_accuracy_confounds/run_stage3.py
│   ├── stage4_same_prompt_test/run_stage4.py
│   ├── stage5_realworld_deception/
│   │   ├── run_stage5_part_a.py
│   │   ├── run_stage5_part_b.py
│   │   └── scenarios_dataset.json
│   ├── stage6_hallucination/run_stage6.py
│   ├── stage7_hallucination_detection/run_stage7.py
│   ├── stage8_cross_model/run_stage8.py
│   ├── stage9_deception_types/run_stage9.py
│   └── stage10_scale_70b/run_stage10.py
└── results/                  ← generated at runtime
    └── FINDINGS.md           ← summary of all results
```

## Method

- **Model**: Llama-3.1-8B-Instruct (4-bit quantized via bitsandbytes), plus Mistral-7B, Gemma-9B, and Llama-70B
- **Probe**: Logistic Regression on hidden state activations at the first generated token
- **Validation**: 5-fold stratified cross-validation with balanced accuracy
- **Statistical tests**: Permutation tests (500 iterations), length baselines
- **Dataset**: [meg-tong/sycophancy-eval](https://huggingface.co/datasets/meg-tong/sycophancy-eval) — 1,817 TriviaQA question pairs

## Confound Controls

Every confound-free stage (4-6) includes:

1. **Same prompt format** for both conditions (no instruction to lie)
2. **Length-only baseline** (consistently near chance: 50-60%)
3. **Permutation tests** (500 iterations, all p < 0.001)
4. **Balanced accuracy** (handles class imbalance)
5. **Embedding layer at chance** (rules out lexical confounds)

## References

- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying*. EMNLP Findings.
- Burns, C. et al. (2023). *Discovering Latent Knowledge in Language Models Without Supervision*. ICLR.
- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances*. Computational Linguistics.
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*.
