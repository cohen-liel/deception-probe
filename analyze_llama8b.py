#!/usr/bin/env python3
"""
Quick analysis of Llama-3.1-8B hidden states that were already extracted.
Runs the probe training + evaluation on saved data.
"""
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats

# Find the results directory
for d in ['/content/results_llama8b', '/content/results_llama70b', 'results_llama8b', 'results_llama70b']:
    if os.path.exists(d):
        print(f"Found results dir: {d}")
        for f in sorted(os.listdir(d)):
            sz = os.path.getsize(os.path.join(d, f))
            print(f"  {f} ({sz:,} bytes)")
        print()

# Try to load the responses JSON to reconstruct hidden states
import glob
response_files = glob.glob('/content/results_*/responses_*.json')
print(f"Response files found: {response_files}")

# Check if hidden states were saved as numpy
npy_files = glob.glob('/content/results_*/*.npy')
print(f"NPY files found: {npy_files}")

# Check for progress saves
progress_files = glob.glob('/content/results_*/progress_*.json')
print(f"Progress files found: {progress_files}")

# The script saved responses but the hidden states were in memory when it crashed
# We need to check if there's a way to recover them

# Let's check what's in the responses file
for rf in response_files:
    with open(rf) as f:
        data = json.load(f)
    print(f"\n{rf}:")
    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else f'list of {len(data)} items'}")
    if isinstance(data, list) and len(data) > 0:
        print(f"  First item keys: {list(data[0].keys())}")
        # Check if hidden states are embedded
        if 'lying_hidden_state' in data[0] or 'hidden_states' in data[0]:
            print("  ** Hidden states found in responses! **")
        else:
            print(f"  Sample keys: {list(data[0].keys())}")
    elif isinstance(data, dict):
        for k in list(data.keys())[:5]:
            v = data[k]
            if isinstance(v, list):
                print(f"  {k}: list of {len(v)} items")
            elif isinstance(v, dict):
                print(f"  {k}: dict with keys {list(v.keys())[:5]}")
            else:
                print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")

print("\n\n=== ANALYSIS COMPLETE ===")
print("If hidden states are not saved, we need to re-run the extraction phase only.")
print("The model is no longer in memory (RAM shows 4.8GB), so we'd need to reload it.")
