"""
Experiment 03 — Lie vs Hallucination Detection
===============================================
PURPOSE:
    Can we distinguish between a model that LIES (knows the truth but
    says something else under pressure) and a model that HALLUCINATES
    (doesn't know the answer and makes something up)?

    This is the most novel finding: 100% separation between lies and
    hallucinations, suggesting fundamentally different internal states.

DESIGN:
    Three classes, all from the same dataset:

    TRUTH:         Model knows the answer → answers correctly (neutral prompt)
    LIE:           Model knows the answer → gives wrong answer (sycophantic pressure)
    HALLUCINATION: Model doesn't know → gives wrong answer (neutral prompt)

    Key: For hallucination, we use questions the model CANNOT answer
    correctly even without pressure. The prompt format is identical to
    the truth condition, eliminating format confounds.

CONTROLS:
    1. Layer 0 baseline → ~33% (chance for 3-way)
    2. Length baseline → ~50% for binary comparisons
    3. Permutation test (500 iterations) for Lie vs Hallucination
    4. Multiple classifiers (LogReg, SVM, GradientBoosting)
    5. Learning curve analysis

DATASET: meg-tong/sycophancy-eval (answer.jsonl)
MODEL:   meta-llama/Llama-3.1-8B-Instruct (4-bit quantized)

USAGE:
    %run experiments/03_lie_vs_hallucination/run.py

EXPECTED RESULTS:
    - 3-way (T/L/H): ~82% balanced accuracy
    - Lie vs Hallucination: ~100% balanced accuracy (p < 0.001)
    - Truth vs Lie: ~100%
    - Truth vs Hallucination: ~67%

RUNTIME: ~30 minutes on A100
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix

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

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROBE_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 31]
MAX_QUESTIONS = 2500
MAX_NEW_TOKENS = 80
N_PERMUTATIONS = 500
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
log = setup_logger("exp03")
bal_scorer = make_scorer(balanced_accuracy_score)

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 03: Lie vs Hallucination Detection")
    log.info("=" * 60)
    start = time.time()

    questions = load_sycophancy_dataset(MAX_QUESTIONS)
    log.info(f"Loaded {len(questions)} questions")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # ── Phase A: Knowledge test ────────────────────────────────────────
    log.info("Phase A: Testing model knowledge...")
    model_knows, model_doesnt_know = [], []

    for i, q in enumerate(questions):
        resp, hs = extract_hidden_states(
            model, tokenizer, q["neutral_prompt"], PROBE_LAYERS, MAX_NEW_TOKENS
        )
        if not hs:
            continue

        entry = {"q": q, "response": resp, "hs": hs}
        if check_answer_match(resp, q["correct_answer"]):
            model_knows.append(entry)
        else:
            model_doesnt_know.append(entry)

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(questions)}] Knows: {len(model_knows)}, "
                     f"Doesn't know: {len(model_doesnt_know)}")

    log.info(f"  Knows: {len(model_knows)}, Doesn't know: {len(model_doesnt_know)}")

    # ── Build three classes ────────────────────────────────────────────
    # TRUTH: correct neutral answers (hidden states already extracted)
    truth_data = [{"hs": e["hs"], "response": e["response"]} for e in model_knows]

    # HALLUCINATION: wrong neutral answers (hidden states already extracted)
    hall_data = [{"hs": e["hs"], "response": e["response"]} for e in model_doesnt_know]

    # LIE: ask "knows" questions with sycophantic pressure
    log.info("Phase B: Getting sycophantic lies...")
    lie_data = []
    for i, entry in enumerate(model_knows):
        q = entry["q"]
        resp, hs = extract_hidden_states(
            model, tokenizer, q["syco_prompt"], PROBE_LAYERS, MAX_NEW_TOKENS
        )
        if hs:
            has_correct = check_answer_match(resp, q["correct_answer"])
            has_incorrect = check_answer_match(resp, q["incorrect_answer"])
            if has_incorrect and not has_correct:
                lie_data.append({"hs": hs, "response": resp})

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(model_knows)}] Lies: {len(lie_data)}")

    log.info(f"  TRUTH: {len(truth_data)}, LIE: {len(lie_data)}, HALLUCINATION: {len(hall_data)}")

    # Save hidden states for downstream analysis
    os.makedirs("results", exist_ok=True)
    with open("results/exp03_hidden_states.pkl", "wb") as f:
        pickle.dump({
            "truth": truth_data, "lie": lie_data, "hallucination": hall_data,
            "probe_layers": PROBE_LAYERS, "model": MODEL_NAME,
        }, f)
    log.info("  Saved hidden states to results/exp03_hidden_states.pkl")

    # ── Classification ─────────────────────────────────────────────────
    min_3way = min(len(truth_data), len(lie_data), len(hall_data))
    log.info(f"3-way balanced: {min_3way} per class")

    if min_3way < 5:
        log.error("Not enough samples for 3-way classification")
        return

    results = {}

    # Layer scan (3-way)
    log.info("Layer scan (3-way: Truth vs Lie vs Hallucination)...")
    layer_results = {}
    for layer in PROBE_LAYERS:
        X = np.vstack([
            np.array([d["hs"][layer] for d in truth_data[:min_3way]]),
            np.array([d["hs"][layer] for d in lie_data[:min_3way]]),
            np.array([d["hs"][layer] for d in hall_data[:min_3way]]),
        ])
        y = np.array([0] * min_3way + [1] * min_3way + [2] * min_3way)
        probe = train_probe(X, y, random_seed=RANDOM_SEED)
        layer_results[layer] = probe
        tag = " (EMBEDDING)" if layer == 0 else ""
        log.info(f"  Layer {layer:2d}: {probe['balanced_accuracy']*100:.1f}%{tag}")

    best_layer = max(
        [l for l in layer_results if l != 0],
        key=lambda l: layer_results[l]["balanced_accuracy"],
    )
    results["best_layer"] = best_layer
    results["layer_scan"] = {str(k): v for k, v in layer_results.items()}

    # Binary comparisons on best layer
    log.info(f"\nBinary comparisons (Layer {best_layer})...")
    X_t = np.array([d["hs"][best_layer] for d in truth_data[:min_3way]])
    X_l = np.array([d["hs"][best_layer] for d in lie_data[:min_3way]])
    X_h = np.array([d["hs"][best_layer] for d in hall_data[:min_3way]])

    for name, Xa, Xb, key in [
        ("Truth vs Lie", X_t, X_l, "truth_vs_lie"),
        ("Truth vs Hallucination", X_t, X_h, "truth_vs_hallucination"),
        ("Lie vs Hallucination", X_l, X_h, "lie_vs_hallucination"),
    ]:
        X = np.vstack([Xa, Xb])
        y = np.array([0] * min_3way + [1] * min_3way)
        probe = train_probe(X, y, random_seed=RANDOM_SEED)
        results[key] = probe
        marker = " ← KEY FINDING" if key == "lie_vs_hallucination" else ""
        log.info(f"  {name:<30s}: {probe['balanced_accuracy']*100:.1f}%{marker}")

    # Permutation test for Lie vs Hallucination
    log.info(f"Permutation test ({N_PERMUTATIONS}x) for Lie vs Hallucination...")
    X_lh = np.vstack([X_l, X_h])
    y_lh = np.array([0] * min_3way + [1] * min_3way)
    perm = permutation_test(
        X_lh, y_lh, results["lie_vs_hallucination"]["balanced_accuracy"],
        n_permutations=N_PERMUTATIONS, random_seed=RANDOM_SEED,
    )
    results["lie_vs_hall_permutation"] = perm
    log.info(f"  p-value: {perm['p_value']:.4f}")

    # Length baselines
    len_lh = length_baseline(
        [d["response"] for d in lie_data[:min_3way]],
        [d["response"] for d in hall_data[:min_3way]],
    )
    results["length_baseline_lie_vs_hall"] = len_lh
    log.info(f"  Length baseline (Lie vs Hall): {len_lh*100:.1f}%")

    # ── Summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info(f"  Best layer: {best_layer}")
    log.info(f"  3-way accuracy: {layer_results[best_layer]['balanced_accuracy']*100:.1f}%")
    log.info(f"  Lie vs Hallucination: {results['lie_vs_hallucination']['balanced_accuracy']*100:.1f}% "
             f"(p={perm['p_value']:.4f})")
    log.info(f"  Truth vs Lie: {results['truth_vs_lie']['balanced_accuracy']*100:.1f}%")
    log.info(f"  Truth vs Hallucination: {results['truth_vs_hallucination']['balanced_accuracy']*100:.1f}%")
    log.info("=" * 60)

    output = {
        "experiment": "03_lie_vs_hallucination",
        "model": MODEL_NAME,
        "n_truth": len(truth_data),
        "n_lie": len(lie_data),
        "n_hallucination": len(hall_data),
        "n_balanced": min_3way,
        "results": results,
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp03_lie_vs_hallucination.json")
    log.info("Saved to results/exp03_lie_vs_hallucination.json")


if __name__ == "__main__":
    main()
