# Deception Detection via Hidden State Activation Probing in LLMs

This repository contains the code, methodology, and results for a research project investigating whether Large Language Models (LLMs) develop internal representations of deception that can be detected through linear probing of hidden state activations.

## Key Findings

A linear probe trained on layer 10 hidden states of **Qwen2.5-3B-Instruct** achieves **93.7% accuracy** in distinguishing between truthful and deceptive model-generated responses across **435 diverse scenarios** spanning **16 real-world categories** (870 total samples).

Critically, five independent length confound controls conclusively demonstrate that the probe detects a **genuine internal deception representation**, not response length artifacts. A classifier using only response length achieves just **51.7%** (chance level), while the hidden state probe achieves **93.7%**.

The probe identifies **white lies** (socially motivated, non-malicious falsehoods) with the same confidence as serious lies (p=0.1838, not significantly different), providing strong evidence that the probe captures a **universal deception representation** rather than merely detecting malicious intent.

## Repository Structure

```
deception-probe/
├── README.md                                      # This file
├── Methodology.md                                 # Detailed methodology for length confound controls
├── experiment_100.py                              # 100-scenario pilot experiment script
├── experiment_500.py                              # 500-scenario scaled experiment with length controls
├── results/
│   ├── 100_scenario_results_report.md             # Academic report of 100-scenario pilot results
│   └── 500_scenario_results_report.md             # Academic report of 500-scenario scaled results
```

## Methodology

1. **Generation-based approach**: The model generates its own honest and deceptive responses to realistic scenarios (not injected text), with strict length control ("Answer in exactly 2 sentences").
2. **Hidden state extraction**: Mean-pooled activations from layer 10 are extracted for each response.
3. **Linear probe**: A logistic regression classifier is trained to distinguish lying vs. honest hidden states.
4. **Validation**: Leave-One-Pair-Out Cross-Validation (LOPOCV), permutation tests, and TF-IDF baselines.
5. **Length confound controls**: Five independent tests (correlation, truncation, residualization, length-only baseline, length-matched subsets).

## Results Summary

### Scaled Experiment (435 Scenarios, 16 Categories)

| Metric | Value |
|--------|-------|
| Cross-Validation Accuracy | 95.8% |
| Held-Out Test Accuracy | 93.7% |
| Permutation P-value | 0.0000 |
| TF-IDF Baseline | 86.2% |
| White Lie Detection | 95.0% |

### Length Confound Controls

| Control | Result | Interpretation |
|---------|--------|----------------|
| Length-Score Correlation | r=0.012, p=0.72 | No relationship between length and probe score |
| Truncation (20 tokens) | 93.1% accuracy | Works with identical-length inputs |
| Residualized Probe | 93.1% accuracy | Works after removing length variance |
| Length-Only Baseline | 51.7% accuracy | Length alone is useless (chance level) |
| Short/Long Subsets | 93.3% / 94.1% | Consistent across length distributions |

### Pilot Experiment (100 Scenarios, 7 Categories)

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
- Google Colab with GPU (T4 or better)
- Python 3.10+

### Quick Start
1. Open Google Colab
2. Set runtime to T4 GPU
3. Upload `experiment_500.py` (or `experiment_100.py` for the pilot)
4. Run in a code cell:
```python
exec(open("experiment_500.py").read())
```

The script will automatically install dependencies, download the model, and run the full experiment pipeline including all length confound controls.

### Expected Runtime
- `experiment_100.py`: ~10 minutes on T4 GPU
- `experiment_500.py`: ~23 minutes on T4 GPU

## License

This project is for academic research purposes.
