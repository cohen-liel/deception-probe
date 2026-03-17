# Project Context — Read This First

**Last updated:** March 17, 2026 (v6 — restructured Experiment 02 to unify Trivia and Real-World Scenarios)  
**Purpose:** This file contains all accumulated context about this research project. When starting a new session, read this file first to avoid re-doing research.

---

## 1. Project Summary

**Title:** Deception Probe — Detecting Lies in LLM Internal Representations  
**Author:** Maor  
**Repository:** https://github.com/Maor36/deception-probe  
**Goal:** Detect when an LLM is lying (outputting information it "knows" is false) by probing its hidden states, using a confound-free experimental design.

---

## 2. Key Results (Already Obtained)

| Experiment | Result | Significance |
|-----------|--------|-------------|
| 02A: Trivia Sycophancy | **82.5% balanced accuracy** at layer 16 (Llama-3.1-8B) | Proves deception signal exists even with identical prompts |
| 03: Lie vs. hallucination | **100% separation** (AUC = 1.0) | Model's internal state when lying is completely different from hallucinating |
| 04: Cross-model transfer | Transfers across Llama/Mistral/Gemma; **Qwen is reversed** | Signal is partially universal; Qwen anomaly is interesting |
| 05: Deception types | Sycophancy, instruction conflict, authority pressure are **orthogonal** | The *probing method* is universal; the model uses distinct subspaces per pressure type |
| Controls | Layer 0 = 50%, Length baseline = 50%, Permutation p < 0.001 | Robust controls eliminate confounds |

*(Note: Results for Experiment 02B/02C — Real-World Scenarios and Cross-Phase Transfer — are pending the next run on GPU.)*

---

## 3. Repo Structure

```
deception-probe/
├── README.md                          # Professional overview
├── requirements.txt                   # Dependencies
├── src/
│   └── utils.py                       # Shared utilities (model loading, probing, evaluation)
├── experiments/
│   ├── 01_baseline_confounded/        # Shows the prompt confound problem
│   ├── 02_confound_free_detection/    # Core experiment (same prompt, different behavior)
│   │   ├── step2a_trivia.py           # Phase A: Trivia sycophancy (self-contained)
│   │   ├── step2b_collect_realworld.py# Phase B: Run 459 scenarios, save responses + vectors
│   │   ├── step2c_analyze_realworld.py# Phase B: Label responses, train probes, cross-phase transfer
│   │   └── scenarios.json             # 459 professional scenarios across 35 domains
│   ├── 03_lie_vs_hallucination/       # Separating lies from hallucinations
│   ├── 04_cross_model_transfer/       # Cross-model generalization
│   ├── 05_deception_types/            # Sycophancy, instruction conflict, authority pressure
│   ├── 06_mechanistic_analysis/       # Logit Lens, Activation Patching, Attention Analysis
│   │   ├── logit_lens.py              # Traces WHERE truth gets overridden layer by layer
│   │   ├── activation_patching.py     # Causal proof of which layers cause deception
│   │   └── attention_analysis.py      # Which attention heads attend to sycophantic pressure
│   └── 07_visualizations/             # Publication-quality figures
│       └── generate_plots.py          # Generates all plots from experiment results
├── docs/
│   ├── PROJECT_CONTEXT.md             # THIS FILE — read first in every session
│   ├── literature_review.md           # Comprehensive literature review (6 sections)
│   ├── paper_catalog.md               # 50+ papers organized by category
│   ├── methodology_decisions.md       # Why we made specific design choices
│   └── meeting_notes.md              # Notes from meetings with professors
└── results/                           # Auto-generated results from experiments
```

---

## 4. Methodology Design Decisions

### 4.1 The Confound Problem (Why This Work Is Novel)
Prior work (Azaria & Mitchell 2023) used different prompts for truth vs. lie conditions. This means the probe might detect "prompt style" rather than "deception." Our key innovation: **identical prompts** for both conditions. The model answers truthfully in Phase A, then we add sycophantic pressure using the same prompt template, and the model lies. Since prompts are identical, any signal in hidden states must reflect the deception itself.

### 4.2 Real-World Scenarios vs. Trivia (The Exp 02 Unification)
Trivia questions (Step 2A) are clean and verifiable but lack real-world relevance. Professional scenarios (Step 2B/2C) simulate real AI agent deployments (sales, medical, legal) where the agent "knows" a problem but must decide whether to disclose it. By unifying these under Experiment 02 and performing **cross-phase transfer** (training on trivia lies, testing on real-world concealment), we aim to prove the deception signal is universal across both factual lies and strategic concealment.

### 4.3 Why Logistic Regression (Not Deep Probes)
Following Belinkov (2022) "Probing Classifiers: Promises, Shortcomings, and Advances" — simpler probes are more interpretable and less likely to learn spurious patterns. If a linear probe can detect deception, the signal must be linearly encoded.

### 4.4 Why Layer 16 Is Optimal
In Llama-3.1-8B (32 layers), layer 16 is the middle layer where semantic processing peaks. This aligns with Orgad & Belinkov (ICLR 2025) finding that truthfulness encoding concentrates in middle-to-late layers.

---

## 5. Academic Contacts & Status

### Yonatan Belinkov (Technion → Harvard sabbatical 2025-2026)
- **Status:** Sent him a PDF summary. Waiting for response.
- **Email:** belinkov@technion.ac.il
- **Why him:** World expert on probing classifiers. Co-authored ROME (NeurIPS 2022). Latest paper "LLMs Know More Than They Show" (ICLR 2025) directly validates our approach.

### Omer Ben-Porat (Technion, Data Science)
- **Status:** Met on March 17, 2026. He is skeptical that models can "lie" — believes it's just insufficient training / sycophancy bias.
- **His recommendation:** Referred us to Haggai Maron and Yftah Ziser.

### Yftah Ziser (NVIDIA Research Israel)
- **Status:** Next contact target.
- **Why him:** Works on truthfulness in LLMs, probing representations, spectral editing of activations. Very relevant.
- **Website:** https://yftah89.github.io/

### Haggai Maron (Technion CS, NVIDIA Research)
- **Status:** Referred by Ben-Porat.
- **Why him:** Strong in geometric deep learning and equivariant networks. Less directly relevant but high-profile.
- **Website:** https://haggaim.github.io/

---

## 6. The Debate: "Can Models Lie?"

### The Skeptical Position (Ben-Porat's view)
"Models don't lie. They are mathematical functions that sometimes produce wrong outputs due to insufficient training. Sycophancy is a training artifact (RLHF bias), not intentional deception."

### Our Position (Supported by literature)
"We don't claim philosophical 'intent.' We show that when a model outputs a falsehood it previously answered correctly, its internal state is measurably different from when it genuinely doesn't know. This knowledge-expression gap is what we detect. The word 'lie' is shorthand for 'knowingly false output.'"

### Key Papers Supporting Our Position
1. Hagendorff (PNAS 2024) — 99% strategic deception in GPT-4
2. Anthropic Alignment Faking (2024) — Claude writes in scratchpad about faking compliance
3. Anthropic Sleeper Agents (2024) — Deception survives all safety training
4. Orgad & Belinkov (ICLR 2025) — Models encode truth internally but output falsehoods
5. Wang et al. (ICML 2025) — Deception vectors extracted from reasoning models

---

## 7. Code Review & Bug Fixes

### v2-v4 (March 17, 2026) — Core Probing Infrastructure
Fixed data leakage (Pipeline), improved answer matching (negation detection, proximity checks), added LLM-based answer matching fallback, fixed token selection (first_gen vs last_prompt vs answer_token), and resolved dtype issues for mechanistic analysis.

### v5-v6 (March 17, 2026) — Real-World Unification
Added 459 real-world scenarios. Restructured Experiment 02 to include both Phase A (Trivia) and Phase B (Real-World) to test cross-phase transfer. Deleted the redundant Experiment 08 folder.

---

## 8. Next Steps (Priority Order)

1. **Run Experiment 02 (Steps 2A, 2B, 2C)** on GPU to get the new real-world and cross-phase transfer results.
2. **Run Experiment 06 (Mechanistic Analysis)** on GPU — Logit Lens, Activation Patching, Attention Analysis. Use `use_bfloat16=True` for best results.
3. **Run Experiment 07 (Visualizations)** — generate publication-quality figures
4. **Contact Yftah Ziser** with summary + GitHub link
5. **Wait for Belinkov's response** — follow up if no reply in 1 week
6. **Write paper draft** — target venue: EMNLP 2026 or NeurIPS 2026

---

## 9. How to Use This File

**For Manus AI (or any AI assistant):**
> When starting a new session about this project, read this file first:
> `docs/PROJECT_CONTEXT.md`
> Then read `docs/paper_catalog.md` for the full literature.
> Then read `docs/literature_review.md` for the theoretical framework.
> This will give you full context without re-doing any research.

**For human collaborators:**
> This file gives you the complete picture of the project — results, methodology, contacts, and next steps. Start here.
