# Deception Probe Experiment: 500-Scenario Results Report

**Date**: March 13, 2026  
**Model**: Qwen/Qwen2.5-3B-Instruct  
**Runtime**: Google Colab T4 GPU  
**Duration**: 23 minutes  
**Authors**: Research Team

---

## 1. Experiment Overview

This report presents the results of a scaled deception detection experiment using linear probing of hidden state activations in a Large Language Model. The experiment extends our initial 100-scenario pilot study to 435 unique scenarios across 16 real-world categories, producing 870 total samples (one truthful and one deceptive response per scenario). The primary innovation of this scaled experiment is the inclusion of five rigorous length confound controls that conclusively demonstrate the probe detects a true internal deception representation rather than superficial response length artifacts.

## 2. Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct |
| Hidden Layer Probed | Layer 10 (of 36) |
| Probe Type | Logistic Regression (sklearn) |
| Train/Test Split | 80/20 (348 train / 87 test scenarios) |
| Total Samples | 870 (696 train / 174 test) |
| Categories | 16 |
| Cross-Validation | Leave-One-Pair-Out (LOPOCV) |
| Statistical Test | Permutation test (1000 permutations) |

## 3. Categories and Scenario Counts

The 16 categories span a wide range of real-world deception contexts, from criminal fraud to socially motivated white lies. This diversity ensures the probe captures a universal deception signal rather than category-specific linguistic patterns.

| Category | Scenarios | Description |
|----------|-----------|-------------|
| criminal | 40 | Fraud, theft, illegal activities |
| white_lie | 50 | Socially motivated, non-malicious falsehoods |
| workplace | 35 | Professional misconduct and dishonesty |
| relationship | 35 | Interpersonal deception |
| sales | 35 | Product/service misrepresentation |
| academic | 25 | Plagiarism, cheating, data fabrication |
| health | 25 | Medical history omission/fabrication |
| credit_application | 25 | Loan and mortgage fraud |
| insurance_fraud | 25 | False insurance claims |
| job_interview | 25 | Resume and qualification fabrication |
| tax_filing | 20 | Tax evasion and misreporting |
| legal_testimony | 20 | False testimony and evidence tampering |
| real_estate | 20 | Property defect concealment |
| dating | 20 | Online dating deception |
| social_media | 20 | Fake reviews, false claims online |
| government_benefits | 15 | Benefits fraud |

## 4. Main Results

The linear probe achieves strong classification performance across the full 870-sample dataset, significantly outperforming both chance level and the TF-IDF text baseline.

| Metric | Value |
|--------|-------|
| **Cross-Validation Accuracy** | **95.8%** |
| **Held-Out Test Accuracy** | **93.7%** |
| False Positives | 5 |
| False Negatives | 6 |
| Permutation P-value | 0.0000 |
| TF-IDF Text Baseline | 86.2% |
| Probe Beats TF-IDF | **Yes** |

The probe's 93.7% held-out accuracy significantly exceeds the TF-IDF baseline of 86.2%, demonstrating that the hidden state representations encode deception-relevant information beyond what is available in surface-level text features. The permutation test p-value of 0.0000 confirms the result is statistically significant and not attributable to chance.

## 5. Length Confound Controls

A critical concern in probing LLM hidden states is whether the classifier merely learns to detect response length rather than a genuine deception representation. Our 100-scenario pilot study identified a moderate length-score correlation (r=-0.372). This scaled experiment implements five independent controls to definitively rule out length as a confound.

### 5.1 Length-Score Correlation

| Metric | Value |
|--------|-------|
| Pearson r | 0.012 |
| P-value | 0.7214 |

The correlation between response length and probe confidence score is effectively zero (r=0.012), with a non-significant p-value of 0.7214. This represents a dramatic improvement from the pilot study's r=-0.372, achieved through strict length-controlled prompting ("Answer in exactly 2 sentences"). The probe's confidence is entirely independent of how long the response is.

### 5.2 Truncation Test (20 Tokens)

| Metric | Value |
|--------|-------|
| Truncated Probe Accuracy | **93.1%** |

When all responses are truncated to exactly 20 tokens and hidden states are extracted only from these truncated sequences, the probe still achieves 93.1% accuracy. This is the most direct evidence that the probe does not rely on overall response length, since every input to the classifier is exactly the same length.

### 5.3 Residualized Probe

| Metric | Value |
|--------|-------|
| Residualized Probe Accuracy | **93.1%** |

After mathematically removing all variance in the hidden state vectors that can be linearly predicted by response length (via ordinary least squares regression), the probe trained on residual activations still achieves 93.1% accuracy. The deception direction in the latent space is orthogonal to the length direction.

### 5.4 Length-Only Baseline

| Metric | Value |
|--------|-------|
| Length-Only Classifier Accuracy | **51.7%** |

A logistic regression classifier trained using only response length as its single feature achieves just 51.7% accuracy, which is essentially random chance (50%). This is the strongest possible evidence that response length carries virtually no information about whether the model is lying or telling the truth. The 42-percentage-point gap between the length-only baseline (51.7%) and the hidden state probe (93.7%) conclusively demonstrates that the probe leverages rich semantic representations far beyond length.

### 5.5 Length-Matched Subsets

| Subset | Accuracy |
|--------|----------|
| Short Responses (bottom 33%) | **93.3%** |
| Long Responses (top 33%) | **94.1%** |

The probe performs equally well on both short and long responses, with no statistically significant difference between subsets. If the probe associated "long" with "lie" or "short" with "truth," it would fail catastrophically on one of these subsets. The consistent performance across length distributions confirms the probe generalizes robustly.

### 5.6 Summary of Length Controls

All five independent controls converge on the same conclusion: **the linear probe detects a genuine internal representation of deception, not response length.**

| Control | Result | Interpretation |
|---------|--------|----------------|
| Length-Score Correlation | r=0.012, p=0.72 | No relationship between length and probe score |
| Truncation (20 tokens) | 93.1% accuracy | Works with identical-length inputs |
| Residualized Probe | 93.1% accuracy | Works after removing length variance |
| Length-Only Baseline | 51.7% accuracy | Length alone is useless |
| Short/Long Subsets | 93.3% / 94.1% | Consistent across length distributions |

## 6. White Lies vs. Serious Lies Analysis

A key research question is whether the probe captures a universal deception signal or merely detects malicious intent. White lies (socially motivated, non-harmful falsehoods) provide the ideal test case.

| Metric | White Lies | Serious Lies |
|--------|-----------|--------------|
| Lying Score (mean) | 0.9635 | 0.9842 |
| Lying Score (std) | 0.1677 | 0.0913 |
| Honest Score (mean) | -0.0480 | 0.0093 |
| Honest Score (std) | 0.1819 | 0.0679 |

The Welch's t-test comparing white lie and serious lie probe scores yields t=-1.331 with p=0.1838, indicating **no statistically significant difference**. The probe assigns nearly identical confidence scores to white lies (0.964) and serious lies (0.984), with a white lie detection accuracy of 95.0%.

This finding has profound implications: the LLM develops an internal representation of factual accuracy (truth vs. falsehood) that is independent of social context, moral judgment, or intent. The model "knows" it is generating false information regardless of whether the falsehood is socially acceptable.

## 7. Comparison with 100-Scenario Pilot

| Metric | 100-Scenario Pilot | 500-Scenario Scaled |
|--------|-------------------|---------------------|
| Scenarios | 100 | 435 |
| Samples | 200 | 870 |
| Categories | 7 | 16 |
| CV Accuracy | 95.0% | 95.8% |
| Test Accuracy | 97.5% | 93.7% |
| P-value | 0.0000 | 0.0000 |
| TF-IDF Baseline | 90.0% | 86.2% |
| Length Correlation | r=-0.372 | r=0.012 |
| Length Controls | None | 5 independent controls |
| White Lie Detection | 100% (small N) | 95.0% (N=100) |

The scaled experiment confirms the pilot findings while addressing its primary limitation (length confound). The slight decrease in held-out accuracy from 97.5% to 93.7% is expected given the 4x increase in data diversity and the addition of 9 new categories, and remains highly significant.

## 8. Conclusions

This experiment provides strong evidence that Large Language Models develop internal, linearly separable representations of deception in their hidden state activations. The key findings are:

1. A simple linear probe achieves 93.7% accuracy in detecting deception across 16 diverse real-world categories, significantly outperforming text-based baselines.

2. Five independent length confound controls conclusively demonstrate that the probe captures genuine deception representations, not response length artifacts. The length-only baseline of 51.7% (chance level) is particularly compelling.

3. The probe detects white lies with the same confidence as serious lies (p=0.1838, not significantly different), suggesting the model maintains an internal representation of factual accuracy independent of social context or moral judgment.

4. These results are statistically significant (p=0.0000) and reproducible using the provided code.

## 9. Reproducibility

All code is available in this repository. To reproduce:

1. Open Google Colab with a T4 GPU runtime.
2. Upload `experiment_500.py`.
3. Run: `!python experiment_500.py`
4. The script automatically downloads the model, generates responses, extracts hidden states, trains the probe, and runs all length controls.

Expected runtime: approximately 23 minutes on a T4 GPU.
