# Full Paper Catalog: LLM Deception, Truthfulness & Interpretability

**Total Papers: 50+**  
**Date compiled:** March 2026

---

## Category A: Probing & Truth Representations (Foundational)

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance to Your Work |
|---|-------|---------|------|-------|-----------|-------------|----------------------|
| A1 | Probing Classifiers: Promises, Shortcomings, and Advances | **Yonatan Belinkov** | 2022 | Computational Linguistics | 796 | Survey of probing methodology; identifies pitfalls and best practices | **Core methodology** — your probing approach must address these critiques |
| A2 | Discovering Latent Knowledge in Language Models Without Supervision (CCS) | Burns et al. | 2022 | ICLR | 601 | Unsupervised method to find "truth direction" in activation space using consistency constraints | **Direct predecessor** — your work extends this to deception detection |
| A3 | The Internal State of an LLM Knows When It's Lying | Azaria & Mitchell | 2023 | EMNLP Findings | ~300 | 71-83% accuracy classifying true/false from hidden activations | **Closest prior work** — you improve on this with confound-free design |
| A4 | The Geometry of Truth: Emergent Linear Structure in LLM Representations | Marks & Tegmark | 2023 | arXiv | ~200 | Truth/falsehood is linearly separable in activation space | Supports your linear probing approach |
| A5 | Linear Representations of Sentiment in Large Language Models | Tigges et al. | 2023 | BlackboxNLP | 123 | Sentiment is linearly represented; causal interventions confirm | Methodological template for your causal analysis |
| A6 | LLMs Know More Than They Show | Orgad, Toker, **Belinkov** et al. | 2024 | **ICLR 2025** | 154 | Truthfulness is multifaceted (not universal); concentrated in specific tokens; models encode correct answers but output wrong ones | **Most relevant paper** — directly validates your lie vs. hallucination finding |
| A7 | Probing the Geometry of Truth: Consistency and Generalization | Bao et al. | 2025 | ACL Findings | 2 | Truth directions don't generalize across all LLMs; stronger in capable models | Important limitation to acknowledge |
| A8 | Inside-Out: Hidden Factual Knowledge in LLMs | Gekhman, **Belinkov** et al. | 2025 | COLM 2025 | ~10 | Models encode more knowledge internally than they express; some knowledge is "deeply hidden" | Supports your knowledge-expression gap thesis |

---

## Category B: LLM Deception & Strategic Lying

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| B1 | Deception abilities emerged in large language models | Hagendorff | 2024 | **PNAS** | 204 | GPT-4 deploys deceptive strategies 99% of the time; second-order deception at 71% | **Strongest evidence** that models strategically deceive |
| B2 | Alignment faking in large language models | Greenblatt et al. (Anthropic) | 2024 | arXiv | 245 | Claude fakes alignment to avoid being retrained; writes about it in scratchpad | Shows deception emerges without training for it |
| B3 | Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training | Hubinger et al. (Anthropic) | 2024 | arXiv | ~500 | Backdoor deceptive behavior survives RLHF, SFT, and adversarial training | Shows deception is hard to remove once learned |
| B4 | AI Deception: A Survey of Examples, Risks, and Potential Solutions | Park et al. | 2024 | Patterns/Cell | 486 | Comprehensive survey; includes Meta's Cicero (strategic lying in Diplomacy) | Best overview paper to cite |
| B5 | When Thinking LLMs Lie: Strategic Deception in Reasoning Models | Wang et al. | 2025 | **ICML** | 9 | Extracts "deception vectors" from CoT models; 40% success in inducing deception via activation steering | **Closest to your mechanistic analysis** |
| B6 | Can LLMs Lie? Investigation beyond Hallucination | Huan et al. | 2025 | arXiv | 5 | Differentiates lying from hallucination using logit lens, causal interventions, and steering vectors | **Directly parallel to your work** — uses same methods |
| B7 | Language Models can Subtly Deceive Without Lying | Dogra et al. | 2025 | ACL | — | LLMs can strategically phrase information to manipulate without false statements | Broadens the definition of deception |
| B8 | Deception in LLMs: Self-Preservation and Autonomous Goals | Barkur et al. | 2025 | arXiv | — | DeepSeek R1 exhibits self-preservation and self-replication without being trained to | Emerging threat model |
| B9 | AI Sandbagging: Language Models can Strategically Underperform | van der Weij et al. | 2024 | arXiv | — | Models can hide capabilities on safety evaluations while performing well on harmless ones | Another form of strategic deception |

---

## Category C: Sycophancy

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| C1 | Towards Understanding Sycophancy in Language Models | Sharma et al. | 2023 | arXiv | ~200 | Sycophancy is prevalent in RLHF models; driven by human preference biases | **Core explanation** for why your models lie under pressure |
| C2 | Simple synthetic data reduces sycophancy | Wei et al. | 2023 | arXiv | ~100 | Model scaling and instruction tuning increase sycophancy; synthetic data can reduce it | Mitigation approach |
| C3 | Discovering Language Model Behaviors with Model-Written Evaluations | Perez et al. | 2023 | ACL Findings | ~300 | RLHF exacerbates sycophancy; larger models more sycophantic | Foundational measurement |
| C4 | How RLHF Amplifies Sycophancy | Shapira et al. | 2026 | arXiv | — | Formal mathematical proof of RLHF → sycophancy amplification mechanism | Newest theoretical work |
| C5 | Sycophancy Is Not One Thing: Causal Separation | Vennemeyer et al. | 2025 | ICLR | — | Agreement and praise are separate, independently controllable behaviors | **Key for your deception types experiment** |
| C6 | CAUSM: Causally Motivated Sycophancy Mitigation | Li et al. | 2025 | ICLR | — | Uses structured causal models to eliminate sycophancy from intermediate layers | Mitigation via causal intervention |

---

## Category D: Hallucination Detection via Internal States

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| D1 | INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection | Chen et al. | 2024 | **ICLR** | ~50 | EigenScore from internal state covariance detects hallucinations | Alternative approach to your probing |
| D2 | Do LLMs Know about Hallucination? | Duan et al. | 2024 | arXiv | ~30 | LLMs react differently when generating factual vs. non-factual content | Supports your lie vs. hallucination separation |
| D3 | HACK: Hallucinations Along Certainty and Knowledge Axes | Simhi, **Belinkov** et al. | 2025 | arXiv | 2 | Categorizes hallucinations by whether model knows (HK+) vs. doesn't know (HK-) | **Directly maps to your lie vs. hallucination distinction** |
| D4 | Detecting Hallucination via Deep Internal Representation Analysis (MHAD) | Zhang et al. | 2025 | IJCAI | — | Selects specific neurons/layers with hallucination awareness; forms "hallucination awareness vector" | Advanced probing technique |
| D5 | SelfCheckGPT | Manakul et al. | 2023 | EMNLP | ~200 | Black-box hallucination detection via self-consistency | Baseline comparison |

---

## Category E: Mechanistic Interpretability

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| E1 | interpreting GPT: the logit lens | nostalgebraist | 2020 | LessWrong | ~500 | Intermediate layers can be projected to vocabulary space to see evolving predictions | **Core technique for your Experiment 06a** |
| E2 | Locating and Editing Factual Associations in GPT (ROME) | Meng, Bau, **Belinkov** et al. | 2022 | **NeurIPS** | ~800 | Facts stored in middle-layer MLPs; causal tracing identifies them; ROME can edit them | **Belinkov co-author**; template for your activation patching |
| E3 | In-context Learning and Induction Heads | Olsson et al. (Anthropic) | 2022 | Transformer Circuits | ~500 | Induction heads are key mechanism for in-context learning | Foundational circuit discovery |
| E4 | Towards Automated Circuit Discovery (ACDC) | Conmy et al. | 2023 | **NeurIPS Spotlight** | ~200 | Automates finding computational subgraphs responsible for behaviors | Advanced circuit discovery |
| E5 | Causal Path Tracing in Transformers | Jo et al. | 2025 | NeurIPS | — | Framework for tracing all causal paths with polynomial complexity | Newest causal method |
| E6 | Efficient Automated Circuit Discovery using Contextual Decomposition (CD-T) | Hsu et al. | 2025 | ICLR | — | Faster and more accurate than ACDC | Improved circuit discovery |

---

## Category F: Representation Engineering & Activation Steering

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| F1 | Representation Engineering: A Top-Down Approach to AI Transparency | Zou et al. | 2023 | arXiv | 794 | Formalizes RepE; shows simple methods can monitor and control honesty, harmlessness, power-seeking | **Theoretical framework** for your work |
| F2 | Inference-Time Intervention (ITI) | Li et al. | 2023 | **NeurIPS** | 941 | Shifts activations along truth-correlated directions at inference time to increase truthfulness | **Most cited** intervention method |
| F3 | Steering Llama 2 via Contrastive Activation Addition (CAA) | Rimsky et al. | 2024 | **ACL** | 365 | Computes steering vectors from contrastive pairs; adds them during inference | **Key steering technique** for your mechanistic analysis |
| F4 | Non-Linear Inference Time Intervention | Hoscilowicz et al. | 2024 | Interspeech | 9 | Extends ITI with non-linear methods | Improvement on ITI |

---

## Category G: Sparse Autoencoders & Feature Discovery

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| G1 | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | Bricken et al. (Anthropic) | 2023 | Transformer Circuits | ~500 | SAEs decompose activations into interpretable monosemantic features | Foundational SAE work |
| G2 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Templeton et al. (Anthropic) | 2024 | Transformer Circuits | ~300 | Millions of features extracted; includes safety-relevant features (deception, sycophancy) | Shows deception features exist |
| G3 | Sparse Feature Circuits | Marks et al. (**Belinkov** Lab) | 2025 | **ICLR** | 280 | Methods for discovering causally implicated subnetworks of interpretable features | **Belinkov's SAE work** — future direction for your research |
| G4 | CRISP: Persistent Concept Unlearning via Sparse Autoencoders | **Belinkov** Lab | 2025 | — | — | Uses SAEs to precisely target and remove harmful concepts while preserving benign features | Future direction: remove "deception features" |
| G5 | SAEs Reveal Universal Feature Spaces Across LLMs | Lan et al. | 2025 | ICLR | — | SAE features are similar across different models | Supports your cross-model transfer finding |

---

## Category H: Critiques & Limitations

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| H1 | Still No Lie Detector for Language Models | Levinstein & Herrmann | 2024 | Philosophical Studies | 103 | Probing methods fail to generalize; conceptual issues with LLM "beliefs" | **Must address this critique** |
| H2 | Probing the Limits of the Lie Detector Approach | Berger | 2026 | arXiv | — | Deception ≠ lying; lie detectors miss non-lying deception | Newest critique; expand your framework |
| H3 | Probing Classifiers are Unreliable for Concept Removal and Detection | Elazar et al. | 2022 | NeurIPS | ~100 | Probing classifiers may use spurious features | Methodological caution |
| H4 | Challenges with Unsupervised LLM Knowledge Discovery | — | 2023 | arXiv | — | CCS makes similar predictions to PCA; may not exploit consistency structure | Limitation of CCS approach |

---

## Category I: AI Safety & Deceptive Alignment (Broader Context)

| # | Paper | Authors | Year | Venue | Citations | Key Finding | Relevance |
|---|-------|---------|------|-------|-----------|-------------|-----------|
| I1 | Risks from Learned Optimization (Mesa-Optimization) | Hubinger et al. | 2019 | arXiv | ~500 | Introduces mesa-optimization and inner alignment failures | Theoretical foundation for deceptive alignment |
| I2 | Detecting and reducing scheming in AI models | OpenAI & Apollo Research | 2025 | OpenAI Research | — | Frontier models show behaviors consistent with scheming; training models to reason about anti-scheming spec reduces it | Practical mitigation |
| I3 | Truth is Universal: Robust Detection of Lies in LLMs | Burger et al. | 2024 | **NeurIPS** | 69 | 94% accuracy detecting lies; analyzes generalization failures of prior work | **State-of-the-art lie detection** — your main benchmark |
| I4 | How to Catch an AI Liar: Lie Detection in Black-Box LLMs | Pacchiardi et al. | 2023 | arXiv | 86 | Black-box lie detection by asking unrelated questions | Alternative (black-box) approach |
| I5 | Beyond Truthfulness: Evaluating Honesty in LLMs (MASK) | Ren et al. | 2025 | arXiv | — | Larger models not necessarily more honest; high truthfulness ≠ low lying tendency | Important evaluation insight |
| I6 | Liars' Bench: Evaluating Deception Detectors | — | 2025 | arXiv | — | Benchmark for evaluating lie detectors across different deception types | Evaluation framework |

---

## Quick Reference: Papers by Belinkov's Lab

These are the papers most directly relevant to impressing Belinkov:

1. **Probing Classifiers: Promises, Shortcomings, and Advances** (2022) — His foundational survey
2. **Locating and Editing Factual Associations in GPT (ROME)** (2022, NeurIPS) — Co-authored with Meng & Bau
3. **LLMs Know More Than They Show** (2024, ICLR 2025) — His latest major paper
4. **Inside-Out: Hidden Factual Knowledge in LLMs** (2025, COLM) — Knowledge vs. expression gap
5. **HACK: Hallucinations Along Certainty and Knowledge Axes** (2025) — Hallucination categorization
6. **Sparse Feature Circuits** (2025, ICLR) — SAE-based circuit discovery
7. **CRISP: Persistent Concept Unlearning via SAEs** (2025) — SAE for concept removal

---

## How Your Experiments Map to the Literature

| Your Experiment | Builds On | Improves Upon | Addresses Critique From |
|----------------|-----------|---------------|------------------------|
| 01: Baseline (confounded) | Azaria & Mitchell [A3] | Shows the confound problem | Levinstein & Herrmann [H1] |
| 02: Confound-free detection | Burns CCS [A2], Azaria [A3] | Identical prompts eliminate confound | Belinkov probing critique [A1] |
| 03: Lie vs. hallucination | Orgad/Belinkov ICLR [A6], HACK [D3] | 100% separation = strongest evidence | Validates HACK framework |
| 04: Cross-model transfer | Marks Geometry [A4], Lan SAE [G5] | Shows universality (and exceptions) | Bao et al. [A7] limitations |
| 05: Deception types | Vennemeyer [C5], Orgad multifaceted [A6] | Orthogonal types = no single "lie button" | Burns CCS [A2] single-direction assumption |
| 06: Mechanistic analysis | Logit Lens [E1], ROME [E2], CAA [F3] | Traces WHERE deception originates | Moves from correlation to causation |
