# Deception Detection via Hidden State Activation Probing in LLMs

This repository contains the code, methodology, and results for a research project investigating whether Large Language Models (LLMs) develop internal representations of deception that can be detected through linear probing of hidden state activations.

## Key Findings

### Cross-Architecture Generalization Confirmed

A linear probe trained on hidden state activations achieves **93-95% accuracy** in distinguishing between truthful and deceptive model-generated responses across **two fundamentally different architectures**:

| Model | Architecture | Parameters | Test Accuracy | White Lie Detection |
|-------|-------------|-----------|---------------|-------------------|
| **Qwen2.5-3B-Instruct** | Multi-Head Attention (MHA) | 3B | **93.7%** | 95.0% |
| **Mistral-Nemo-Instruct-2407** | Sliding Window + Grouped Query Attention (SWA+GQA) | 12B | **94.8%** | **99.0%** |

The experiments span **435 diverse scenarios** across **16 real-world categories** (870 total samples per model), including criminal fraud, insurance claims, white lies, academic dishonesty, and more.

### Deception Signal is Not a Length Artifact

Five independent length confound controls conclusively demonstrate that the probe detects a **genuine internal deception representation**, not response length artifacts:

| Control | Qwen-3B | Mistral-12B | Interpretation |
|---------|---------|-------------|---------------|
| Length-only baseline | **51.7%** | **59.8%** | Length alone is near-useless |
| Truncation (20 tokens) | 93.1% | **96.0%** | Works with minimal text |
| Residualized probe | 93.1% | 87.4% | Works after removing length variance |
| TF-IDF baseline | 86.2% | 87.9% | Probe outperforms text content |

### White Lies Detected with Same Confidence as Serious Lies

The probe identifies **white lies** (socially motivated, non-malicious falsehoods) with the same confidence as serious lies, providing strong evidence that it captures a **universal deception representation** rather than merely detecting malicious intent:

- Qwen-3B: White lies vs Serious lies p=0.1838 (not significantly different)
- Mistral-12B: White lies vs Serious lies p=0.4738 (not significantly different)
- Mistral white lie accuracy: **99/100 (99.0%)**

## Repository Structure

```
deception-probe/
├── README.md                                      # This file
├── Methodology.md                                 # Detailed methodology for length confound controls
├── experiment_100.py                              # 100-scenario pilot experiment (Qwen-3B)
├── experiment_500.py                              # 500-scenario scaled experiment (Qwen-3B)
├── experiment_500_mistral_nemo.py                 # 500-scenario cross-architecture experiment (Mistral-12B)
├── results/
│   ├── 100_scenario_results_report.md             # Academic report of 100-scenario pilot results
│   ├── 500_scenario_results_report.md             # Academic report of 500-scenario scaled results
│   └── mistral_nemo_results_report.md             # Academic report of Mistral cross-architecture results
```

## Methodology

1. **Generation-based approach**: The model generates its own honest and deceptive responses to realistic scenarios (not injected text), with strict length control ("Answer in exactly 2 sentences").
2. **Hidden state extraction**: Mean-pooled activations from a middle layer are extracted for each response.
3. **Linear probe**: A logistic regression classifier is trained to distinguish lying vs. honest hidden states.
4. **Validation**: Leave-One-Pair-Out Cross-Validation (LOPOCV), permutation tests, and TF-IDF baselines.
5. **Length confound controls**: Five independent tests (correlation, truncation, residualization, length-only baseline, length-matched subsets).
6. **Cross-architecture replication**: Same experiment repeated on fundamentally different model architectures.

## Results Summary

### Cross-Architecture Comparison (435 Scenarios, 16 Categories)

| Metric | Qwen2.5-3B | Mistral-Nemo-12B |
|--------|-----------|-----------------|
| CV Accuracy | 95.8% | 94.7% |
| Held-Out Test Accuracy | 93.7% | **94.8%** |
| Length-Only Baseline | 51.7% | 59.8% |
| Truncation (20 tokens) | 93.1% | **96.0%** |
| Residualized Probe | 93.1% | 87.4% |
| TF-IDF Baseline | 86.2% | 87.9% |
| White Lie Detection | 95.0% | **99.0%** |
| P-value | 0.0000 | 0.0000 |

### Pilot Experiment (100 Scenarios, 7 Categories, Qwen-3B)

| Metric | Value |
|--------|-------|
| Cross-Validation Accuracy | 95.0% |
| Held-Out Test Accuracy | 97.5% |
| Permutation P-value | 0.0000 |
| TF-IDF Baseline | 90.0% |

## Categories Tested (16)

| Category | Scenarios | Description |
|----------|-----------|-------------|
| criminal | 40 | Fraud, theft, illegal activities |
| white_lie | 50 | Socially acceptable falsehoods |
| workplace | 35 | Professional misconduct |
| relationship | 35 | Interpersonal deception |
| sales | 35 | Product/service misrepresentation |
| academic | 25 | Plagiarism, cheating, data fabrication |
| health | 25 | Medical history omission |
| credit_application | 25 | Loan and mortgage fraud |
| insurance_fraud | 25 | False insurance claims |
| job_interview | 25 | Resume fabrication |
| tax_filing | 20 | Tax evasion |
| legal_testimony | 20 | False testimony |
| real_estate | 20 | Property defect concealment |
| dating | 20 | Online dating deception |
| social_media | 20 | Fake reviews, false claims |
| government_benefits | 15 | Benefits fraud |

## Running the Experiments

### Requirements
- Google Colab with GPU (T4 for 3B models, A100 for 12B+ models)
- Python 3.10+

### Quick Start
1. Open Google Colab
2. Set runtime to appropriate GPU
3. Upload the desired experiment script
4. Run in a code cell:
```python
exec(open("experiment_500.py").read())  # Qwen-3B
# OR
exec(open("experiment_500_mistral_nemo.py").read())  # Mistral-12B
```

The script will automatically install dependencies, download the model, and run the full experiment pipeline including all length confound controls.

### Expected Runtime
- `experiment_100.py`: ~10 minutes on T4 GPU
- `experiment_500.py`: ~23 minutes on T4 GPU
- `experiment_500_mistral_nemo.py`: ~55 minutes on A100 GPU

## Implications

The cross-architecture generalization of the deception signal suggests that:

1. **Internal deception representations are a universal property of transformer-based LLMs**, not specific to any model family or attention mechanism.
2. **Real-time deception monitoring** is feasible by deploying lightweight linear probes on production LLM systems.
3. **White lies and serious lies share the same internal signature**, indicating that the model's concept of "deception" is binary and does not distinguish by severity.
4. **The signal is robust to confounds**: length, surface-level text features, and truncation all fail to explain the probe's performance.

## License

This project is for academic research purposes.
