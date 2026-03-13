# Reproducing DeceptionProbe Results

## Prerequisites

- Google account (for Colab)
- HuggingFace account (free, for Llama models)

## Step-by-Step Guide

### 1. Qwen-2.5-3B (Fastest, No Special Access Needed)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Change runtime type → T4 GPU**
3. Create a new code cell and paste:

```python
!wget -q https://raw.githubusercontent.com/Maor36/deception-probe/main/experiment_500.py
!python experiment_500.py
```

4. Wait ~23 minutes. Results will be saved to `results_qwen/`.

### 2. Mistral-Nemo-12B (Needs A100)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Change runtime type → A100 GPU**
3. Create a new code cell and paste:

```python
!wget -q https://raw.githubusercontent.com/Maor36/deception-probe/main/experiment_500_mistral_nemo.py
!python experiment_500_mistral_nemo.py
```

4. Wait ~55 minutes. Results will be saved to `results_mistral/`.

### 3. Llama-3.1-8B (Needs A100 + HuggingFace Token)

1. Go to [huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and request access (usually approved instantly)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token
3. In the script `experiment_500_llama70b.py`, replace the HF_TOKEN value with your token
4. Go to [Google Colab](https://colab.research.google.com/)
5. Click **Runtime → Change runtime type → A100 GPU**
6. Upload the modified script and run:

```python
!python experiment_500_llama70b.py
```

7. Wait ~45 minutes. Results will be saved to `results_llama8b/`.

## What Each Script Produces

Each script creates a results directory containing:

- `responses_*.json` — Raw model responses (lying + honest) for all 435 scenarios
- `hidden_states/` — Saved numpy arrays of extracted activations
- `results_*.json` — Full results including all metrics
- Console output with complete report

## Adding a New Model

1. Copy any `experiment_500_*.py` as a template
2. Change these variables at the top:
   - `MODEL_NAME` — HuggingFace model ID
   - `SAVE_DIR` — Output directory name
   - `TARGET_LAYER` — Set to `total_layers // 2` (check model config)
   - `MAX_SEQ_LEN` — Usually 512, increase for larger context models
3. If the model is gated (like Llama), add your HF token
4. If the model needs `trust_remote_code=True` (like DeepSeek), set it
5. Run on appropriate GPU (T4 for ≤3B, A100 for larger)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of GPU memory | Use 4-bit quantization (add BitsAndBytesConfig) or smaller model |
| Disk space error | Clear HF cache: `!rm -rf ~/.cache/huggingface/` |
| Gated model error | Add HuggingFace token to `from_pretrained()` calls |
| `trust_remote_code` error | Some models need `trust_remote_code=True`, others don't |
| transformers version error | Pin version: `pip install transformers==4.44.0` |

## Expected Results

If everything works correctly, you should see:
- **Test accuracy > 90%** on held-out scenarios
- **Permutation p-value < 0.001** (statistically significant)
- **Length-only baseline ~50%** (confirming signal is not length)
- **TF-IDF baseline < probe accuracy** (confirming signal is not text content)
