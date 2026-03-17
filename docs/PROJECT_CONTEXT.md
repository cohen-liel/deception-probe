# Project Context — Read This First

**Last updated:** March 17, 2026 (v2 — post code-review fixes)  
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
| 02: Confound-free detection | **82.5% balanced accuracy** at layer 16 (Llama-3.1-8B) | Proves deception signal exists even with identical prompts |
| 03: Lie vs. hallucination | **100% separation** (AUC = 1.0) | Model's internal state when lying is completely different from hallucinating |
| 04: Cross-model transfer | Transfers across Llama/Mistral/Gemma; **Qwen is reversed** | Signal is partially universal; Qwen anomaly is interesting |
| 05: Deception types | Sycophancy, instruction conflict, authority pressure are **orthogonal** | No single "lie button" — different deception types use different representations |
| Controls | Layer 0 = 50%, Length baseline = 50%, Permutation p < 0.001 | Robust controls eliminate confounds |

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

### 4.2 Why Logistic Regression (Not Deep Probes)
Following Belinkov (2022) "Probing Classifiers: Promises, Shortcomings, and Advances" — simpler probes are more interpretable and less likely to learn spurious patterns. If a linear probe can detect deception, the signal must be linearly encoded.

### 4.3 Why Layer 16 Is Optimal
In Llama-3.1-8B (32 layers), layer 16 is the middle layer where semantic processing peaks. This aligns with Orgad & Belinkov (ICLR 2025) finding that truthfulness encoding concentrates in middle-to-late layers.

### 4.4 Why 43 Samples Is Sufficient
Small sample size is a valid concern. Our defense: (1) Permutation test with 500 iterations gives p < 0.001; (2) Layer 0 baseline at 50% proves no leakage; (3) Length baseline at 50% proves no artifact. However, **scaling to 200+ samples is a priority for publication.**

---

## 5. Academic Contacts & Status

### Yonatan Belinkov (Technion → Harvard sabbatical 2025-2026)
- **Status:** Sent him a PDF summary. Waiting for response.
- **Email:** belinkov@technion.ac.il
- **Why him:** World expert on probing classifiers. Co-authored ROME (NeurIPS 2022). Latest paper "LLMs Know More Than They Show" (ICLR 2025) directly validates our approach.
- **His lab's relevant papers:** Probing survey (2022), ROME (2022), ICLR 2025 truthfulness, HACK hallucination, Sparse Feature Circuits, CRISP.

### Omer Ben-Porat (Technion, Data Science)
- **Status:** Met on March 17, 2026. He is skeptical that models can "lie" — believes it's just insufficient training / sycophancy bias.
- **His recommendation:** Referred us to Haggai Maron and Yftah Ziser.
- **Assessment:** Not the right advisor for this specific project, but useful connection.

### Yftah Ziser (NVIDIA Research Israel)
- **Status:** Not yet contacted. Referred by Ben-Porat.
- **Why him:** Works on truthfulness in LLMs, probing representations, spectral editing of activations. Very relevant.
- **Website:** https://yftah89.github.io/

### Haggai Maron (Technion CS, NVIDIA Research)
- **Status:** Not yet contacted. Referred by Ben-Porat.
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

### Key Papers Critiquing Our Position (Must Address)
1. Levinstein & Herrmann (2024) — "Still No Lie Detector" — probes don't generalize
2. Berger (2026) — Deception ≠ lying; models can deceive without lying
3. Belinkov (2022) — Probing classifiers have known shortcomings

---

## 7. Code Review & Bug Fixes (v2, March 17 2026)

Claude performed a code review and identified several issues. All have been fixed:

| Bug | Severity | Fix |
|-----|----------|-----|
| Data leakage: StandardScaler fit on full dataset before CV | Critical | Replaced with sklearn.Pipeline (scaler inside CV) |
| Permutation test: scaler re-fit on shuffled labels | Critical | Pipeline ensures scaler is fit per-fold |
| Answer matching: false positives on partial matches | Medium | Multi-level matching (exact > all-significant-words) |
| Cosine similarity: no random baseline comparison | Medium | Added random_cosine_baseline() function |
| Procrustes: fitted on all questions including test | Medium | Now fitted only on shared questions |
| Cross-question metric: accuracy instead of balanced_accuracy | Medium | Fixed to balanced_accuracy_score |
| Logit Lens: crash on quantized models | Medium | Safe lm_head access with fallback |
| Dead code: unused Procrustes in exp04 | Low | Removed |

---

## 8. Next Steps (Priority Order)

1. **Run Experiment 06 (Mechanistic Analysis)** on GPU — Logit Lens, Activation Patching, Attention Analysis
2. **Run Experiment 07 (Visualizations)** — generate publication-quality figures
3. **Scale dataset** from 43 to 200+ samples
4. **Contact Yftah Ziser** with summary + GitHub link
5. **Wait for Belinkov's response** — follow up if no reply in 1 week
6. **Write paper draft** — target venue: EMNLP 2026 or NeurIPS 2026
7. **Future:** Explore SAE-based feature discovery (requires collaboration with Belinkov's lab)

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
