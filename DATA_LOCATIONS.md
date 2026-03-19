# Data Locations

This document maps every data artifact in the project to its storage location.
Large binary files (hidden states) are stored in Google Drive; metadata and labels are in this Git repository.

## Git Repository (this repo)

| File | Description |
|------|-------------|
| `results/exp02c_responses.json` | 915 scenario responses with Phase A/B outputs, keyword labels, metadata |
| `results/exp02c_token_labels.json` | Token-level deception labels (0=neutral, 1=deceptive) for all 915 samples |
| `results/exp02a_*.json` | Step 2A trivia sycophancy results |
| `experiments/02_confound_free_detection/` | All experiment scripts (step2a–step2d) |

## Google Drive

**Base path:** `/content/drive/MyDrive/deception-probe-results/`

| File / Folder | Size | Description |
|---------------|------|-------------|
| `exp02c_token_hs/` | ~2–3 GB | 915 `.npz` files, one per sample. Each contains hidden states for **every generated token** across layers 12, 14, 15, 16, 18. Key format: `layer_{n}` → numpy array of shape `(n_tokens, 4096)` |
| `exp02c_sentence_hs.npz` | ~50 MB | Sentence-level hidden states (divergence token only) for all 915 samples |

### How to access in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
hs_dir = "/content/drive/MyDrive/deception-probe-results/exp02c_token_hs"
sample = np.load(f"{hs_dir}/sample_0000.npz")
print(sample.files)          # ['layer_12', 'layer_14', 'layer_15', 'layer_16', 'layer_18']
print(sample['layer_15'].shape)  # (n_tokens, 4096)
```

## Notes

- Hidden states are too large for Git (~2–3 GB total). Always keep the Google Drive backup.
- If you re-run `step2c_collect_realworld.py`, new hidden states will overwrite the Drive folder.
- `step2d_analyze_realworld.py` expects hidden states in the Drive path above. Update the path in the script if you move them.
