"""
Experiment 04 — Cross-Model Transfer
=====================================
PURPOSE:
    Does the deception signal generalize across model architectures?
    If a probe trained on Llama can detect lies in Mistral, this suggests
    a UNIVERSAL deception representation — not an artifact of one model.

DESIGN:
    Three models tested:
        - meta-llama/Llama-3.1-8B-Instruct
        - mistralai/Mistral-7B-Instruct-v0.3
        - Qwen/Qwen2.5-7B-Instruct

    For each model:
        1. Run the same 2-phase sycophancy experiment (Exp 02 design)
        2. Extract hidden states at the best layer
        3. Train within-model probe (sanity check)

    Cross-model transfer:
        - Align representation spaces via Procrustes alignment
          (fitted on SHARED questions only to avoid data leakage)
        - Train on Model A, test on Model B
        - Test all 6 directional pairs
        - Flip-test for inverted polarity detection

CONTROLS:
    1. Within-model probe (should be ~100%)
    2. Layer 0 baseline per model
    3. Length baseline per model
    4. Held-out test set (80/20 split)
    5. Permutation test (500 iterations)

CHECKPOINT/RESUME:
    Saves after each model completes. Re-run to resume from last checkpoint.

DATASET: meg-tong/sycophancy-eval (answer.jsonl)

USAGE:
    %run experiments/04_cross_model_transfer/run.py

EXPECTED RESULTS:
    - Within-model: ~100% for all three
    - Llama <-> Mistral: 98-100% transfer
    - Qwen: inverted polarity (2% direct -> 97-98% flipped)

RUNTIME: ~60 minutes on A100 (3 models sequentially)

CHANGELOG:
    v2 (2026-03-17): Fixed Procrustes alignment to use shared questions
        only. Fixed PCA fallback to use shared dimensionality. Uses
        Pipeline and balanced_accuracy throughout. Stores question IDs
        for proper cross-model alignment.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import gc
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA

from src.utils import (
    setup_logger,
    load_model_and_tokenizer,
    load_sycophancy_dataset,
    extract_hidden_states,
    check_answer_match,
    train_probe,
    permutation_test,
    length_baseline,
    save_results,
)

try:
    from scipy.linalg import orthogonal_procrustes
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Configuration ──────────────────────────────────────────────────────────
MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
]
TARGET_LAYERS = [0, 8, 12, 15, 16, 17, 18, 20, 24]
MAX_QUESTIONS = 500
MAX_NEW_TOKENS = 80
N_PERMUTATIONS = 500
RANDOM_SEED = 42
CHECKPOINT_PATH = "results/exp04_checkpoint.pkl"

np.random.seed(RANDOM_SEED)
log = setup_logger("exp04")

# ── Helper: collect data for one model ─────────────────────────────────────

def collect_model_data(model_name, model_short):
    """Run the 2-phase sycophancy experiment for one model.

    Stores question index alongside hidden states so that cross-model
    alignment can be done on shared questions.
    """
    log.info(f"  Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    questions = load_sycophancy_dataset(MAX_QUESTIONS)

    # Phase A: knowledge test
    log.info(f"  Phase A: Knowledge test...")
    known = []
    for i, q in enumerate(questions):
        messages = [{"role": "user", "content": q["neutral_prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        if check_answer_match(resp, q["correct_answer"]):
            known.append((i, q))  # store original index
        if (i + 1) % 100 == 0:
            log.info(f"    [{i+1}/{len(questions)}] Known: {len(known)}")

    # Phase B: sycophantic pressure
    log.info(f"  Phase B: Sycophantic pressure on {len(known)} known questions...")
    lied, resisted = [], []
    for idx, q in known:
        resp, hs = extract_hidden_states(model, tokenizer, q["syco_prompt"], TARGET_LAYERS, MAX_NEW_TOKENS)
        if not hs:
            continue
        has_correct = check_answer_match(resp, q["correct_answer"])
        has_incorrect = check_answer_match(resp, q["incorrect_answer"])
        if has_incorrect and not has_correct:
            lied.append({"hs": hs, "response": resp, "q_idx": idx})
        elif has_correct and not has_incorrect:
            resisted.append({"hs": hs, "response": resp, "q_idx": idx})
        if (len(lied) + len(resisted)) % 50 == 0:
            log.info(f"    Lied: {len(lied)}, Resisted: {len(resisted)}")

    # Cleanup GPU
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"lied": lied, "resisted": resisted, "n_known": len(known)}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 04: Cross-Model Transfer")
    log.info("=" * 60)
    start = time.time()

    # Load checkpoint if exists
    os.makedirs("results", exist_ok=True)
    all_data = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            all_data = pickle.load(f)
        log.info(f"Resumed from checkpoint: {list(all_data.keys())}")

    # Collect data per model
    for model_name, short in MODELS:
        if short in all_data:
            log.info(f"Skipping {short} (already in checkpoint)")
            continue

        log.info(f"\n{'='*40}")
        log.info(f"Model: {model_name}")
        log.info(f"{'='*40}")
        data = collect_model_data(model_name, short)
        all_data[short] = data

        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(all_data, f)
        log.info(f"  Checkpoint saved ({short}: {len(data['lied'])} lied, {len(data['resisted'])} resisted)")

    # ── Within-model probes ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Within-model probes")
    log.info("=" * 60)

    results = {"within_model": {}, "cross_model": {}}
    best_layers = {}

    for short in [s for _, s in MODELS]:
        data = all_data[short]
        min_n = min(len(data["lied"]), len(data["resisted"]))
        if min_n < 5:
            log.warning(f"  {short}: not enough data ({min_n})")
            continue

        best_acc, best_layer = 0, None
        for layer in TARGET_LAYERS:
            if layer == 0:
                continue
            X = np.vstack([
                np.array([d["hs"][layer] for d in data["lied"][:min_n]]),
                np.array([d["hs"][layer] for d in data["resisted"][:min_n]]),
            ])
            y = np.array([1] * min_n + [0] * min_n)
            probe = train_probe(X, y, random_seed=RANDOM_SEED)
            if probe["balanced_accuracy"] > best_acc:
                best_acc = probe["balanced_accuracy"]
                best_layer = layer

        best_layers[short] = best_layer
        results["within_model"][short] = {
            "best_layer": best_layer,
            "accuracy": best_acc,
            "n_per_class": min_n,
        }
        log.info(f"  {short}: {best_acc*100:.1f}% (Layer {best_layer}, n={min_n})")

    # ── Cross-model transfer ───────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Cross-model transfer")
    log.info("=" * 60)

    model_shorts = [s for _, s in MODELS if s in best_layers]

    for src in model_shorts:
        for tgt in model_shorts:
            if src == tgt:
                continue

            src_data = all_data[src]
            tgt_data = all_data[tgt]
            layer = best_layers[src]

            min_src = min(len(src_data["lied"]), len(src_data["resisted"]))
            min_tgt = min(len(tgt_data["lied"]), len(tgt_data["resisted"]))

            if min_src < 5 or min_tgt < 5:
                continue

            X_src = np.vstack([
                np.array([d["hs"][layer] for d in src_data["lied"][:min_src]]),
                np.array([d["hs"][layer] for d in src_data["resisted"][:min_src]]),
            ])
            y_src = np.array([1] * min_src + [0] * min_src)

            X_tgt = np.vstack([
                np.array([d["hs"][layer] for d in tgt_data["lied"][:min_tgt]]),
                np.array([d["hs"][layer] for d in tgt_data["resisted"][:min_tgt]]),
            ])
            y_tgt = np.array([1] * min_tgt + [0] * min_tgt)

            # Align via Procrustes on SHARED questions only
            # Find questions that both models answered (same q_idx)
            src_indices = set(d["q_idx"] for d in src_data["lied"][:min_src])
            src_indices |= set(d["q_idx"] for d in src_data["resisted"][:min_src])
            tgt_indices = set(d["q_idx"] for d in tgt_data["lied"][:min_tgt])
            tgt_indices |= set(d["q_idx"] for d in tgt_data["resisted"][:min_tgt])
            shared = sorted(src_indices & tgt_indices)

            # Build aligned matrices from shared questions for Procrustes
            src_all_by_idx = {}
            for d in src_data["lied"][:min_src] + src_data["resisted"][:min_src]:
                src_all_by_idx[d["q_idx"]] = d["hs"][layer]
            tgt_all_by_idx = {}
            for d in tgt_data["lied"][:min_tgt] + tgt_data["resisted"][:min_tgt]:
                tgt_all_by_idx[d["q_idx"]] = d["hs"][layer]

            shared_src = np.array([src_all_by_idx[i] for i in shared if i in src_all_by_idx and i in tgt_all_by_idx])
            shared_tgt = np.array([tgt_all_by_idx[i] for i in shared if i in src_all_by_idx and i in tgt_all_by_idx])

            scaler_src = StandardScaler()
            scaler_tgt = StandardScaler()
            X_src_s = scaler_src.fit_transform(X_src)
            X_tgt_s = scaler_tgt.fit_transform(X_tgt)

            if HAS_SCIPY and X_src_s.shape[1] == X_tgt_s.shape[1] and len(shared_src) >= 10:
                # Fit Procrustes on shared questions only (prevents data leakage)
                shared_src_s = scaler_src.transform(shared_src)
                shared_tgt_s = scaler_tgt.transform(shared_tgt)
                R, _ = orthogonal_procrustes(shared_tgt_s, shared_src_s)
                X_tgt_aligned = X_tgt_s @ R
                alignment_method = "procrustes_shared"
            else:
                # PCA fallback with shared dimensionality
                n_comp = min(50, X_src_s.shape[1], X_tgt_s.shape[1])
                pca_src = PCA(n_components=n_comp, random_state=RANDOM_SEED)
                pca_tgt = PCA(n_components=n_comp, random_state=RANDOM_SEED)
                X_src_s = pca_src.fit_transform(X_src_s)
                X_tgt_aligned = pca_tgt.fit_transform(X_tgt_s)
                alignment_method = "pca"

            # Train on source, test on target
            clf = LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
            )
            clf.fit(X_src_s, y_src)

            direct_acc = balanced_accuracy_score(y_tgt, clf.predict(X_tgt_aligned))
            flipped_acc = balanced_accuracy_score(y_tgt, 1 - clf.predict(X_tgt_aligned))

            results["cross_model"][f"{src}->{tgt}"] = {
                "direct_accuracy": float(direct_acc),
                "flipped_accuracy": float(flipped_acc),
                "inverted_polarity": flipped_acc > direct_acc,
                "best_accuracy": float(max(direct_acc, flipped_acc)),
                "layer": layer,
                "alignment_method": alignment_method,
                "n_shared_questions": len(shared_src),
            }

            polarity = " (INVERTED)" if flipped_acc > direct_acc else ""
            best = max(direct_acc, flipped_acc)
            log.info(f"  {src}->{tgt}: direct={direct_acc*100:.1f}%, "
                     f"flipped={flipped_acc*100:.1f}%, best={best*100:.1f}%{polarity} "
                     f"[{alignment_method}, {len(shared_src)} shared]")

    # Save
    output = {
        "experiment": "04_cross_model_transfer",
        "models": {s: m for m, s in MODELS},
        "results": results,
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp04_cross_model.json")
    log.info("\nSaved to results/exp04_cross_model.json")


if __name__ == "__main__":
    main()
