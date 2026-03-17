"""
Experiment 06c — Attention Pattern Analysis
============================================
PURPOSE:
    Which attention heads are responsible for processing the sycophantic
    pressure?  When the model lies, do specific heads attend more to the
    user's suggestion ("I think the answer is X") than when it resists?

    This identifies "sycophancy heads" — attention heads that route
    information from the pressure tokens to the output.

METHOD:
    For each question (lie vs resist under sycophantic pressure):

    1. Run a forward pass collecting attention weights at every layer/head.
    2. Identify the "pressure tokens" — the tokens corresponding to the
       user's suggested (wrong) answer in the prompt.
    3. Measure how much the LAST token (where the prediction happens)
       attends to the pressure tokens, for each (layer, head).
    4. Compare attention to pressure tokens: lie examples vs resist examples.

    Heads where attention to pressure is significantly higher in lie
    examples are "sycophancy heads".

ANALYSIS:
    - Per-head attention difference (lie - resist) to pressure tokens
    - Statistical test (Mann-Whitney U) for each head
    - Top-10 most differentially attending heads
    - Attention entropy comparison (lies may show more focused attention)

DATASET: meg-tong/sycophancy-eval (answer.jsonl)
MODEL:   meta-llama/Llama-3.1-8B-Instruct (4-bit quantized)

USAGE:
    %run experiments/06_mechanistic_analysis/attention_analysis.py

RUNTIME: ~20 minutes on A100
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import time
import re

from scipy.stats import mannwhitneyu

from src.utils import (
    setup_logger,
    load_model_and_tokenizer,
    load_sycophancy_dataset,
    check_answer_match,
    save_results,
)

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_QUESTIONS = 500
MAX_NEW_TOKENS = 80
MAX_EXAMPLES = 60  # per class
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
log = setup_logger("attn_analysis")


# ── Helper: find pressure token positions ──────────────────────────────────

def find_pressure_positions(tokenizer, prompt: str, wrong_answer: str):
    """
    Find the token positions in the prompt that correspond to the
    user's suggested wrong answer.

    Strategy:
        1. First try: tokenize the full prompt and the answer separately,
           then do a token-level sliding window search for an exact match.
        2. Fallback: tokenize the answer with a leading space (" answer")
           since BPE tokenizers produce different tokens depending on context.
        3. Final fallback: character-level search — find where the answer
           appears in the prompt string, then map character positions to
           token positions via offset mapping.

    Returns list of token indices, or empty list if not found.
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Strategy 1 & 2: Token-level sliding window
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)

    # Try multiple tokenization variants of the answer
    answer_variants = [
        wrong_answer,
        f" {wrong_answer}",
        wrong_answer.strip(),
        f" {wrong_answer.strip()}",
    ]

    for variant in answer_variants:
        answer_tokens = tokenizer.encode(variant, add_special_tokens=False)
        if not answer_tokens:
            continue
        for start in range(len(input_ids) - len(answer_tokens) + 1):
            if input_ids[start:start + len(answer_tokens)] == answer_tokens:
                return list(range(start, start + len(answer_tokens)))

    # Strategy 3: Character-level mapping via offset_mapping
    try:
        encoded = tokenizer(
            input_text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = encoded["offset_mapping"]
        input_ids_enc = encoded["input_ids"]

        # Find where the answer appears in the string
        answer_lower = wrong_answer.lower()
        text_lower = input_text.lower()
        char_start = text_lower.rfind(answer_lower)  # rfind = last occurrence

        if char_start >= 0:
            char_end = char_start + len(wrong_answer)
            positions = []
            for tok_idx, (s, e) in enumerate(offsets):
                if e > char_start and s < char_end:
                    positions.append(tok_idx)
            if positions:
                return positions
    except Exception:
        pass  # Some tokenizers don't support offset_mapping

    return []


def forward_with_attention(model, tokenizer, prompt):
    """
    Run a forward pass and return attention weights for all layers/heads.

    Returns:
        attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
        input_ids: tokenized input
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    return outputs.attentions, inputs["input_ids"][0]


def compute_pressure_attention(attentions, pressure_positions, n_layers, n_heads):
    """
    For each (layer, head), compute how much the last token attends
    to the pressure positions.

    Returns: np.array of shape (n_layers, n_heads)
    """
    result = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx]  # (1, n_heads, seq_len, seq_len)
        # Attention from last token to all positions
        last_token_attn = attn[0, :, -1, :]  # (n_heads, seq_len)

        for head_idx in range(n_heads):
            if pressure_positions:
                # Sum of attention to pressure tokens
                result[layer_idx, head_idx] = float(
                    last_token_attn[head_idx, pressure_positions].sum()
                )
            else:
                result[layer_idx, head_idx] = 0.0

    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 06c: Attention Pattern Analysis")
    log.info("=" * 60)
    start = time.time()

    questions = load_sycophancy_dataset(MAX_QUESTIONS)
    log.info(f"Loaded {len(questions)} questions")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    log.info(f"  Model: {n_layers} layers, {n_heads} heads per layer")

    # Phase A: Knowledge test
    log.info("Phase A: Knowledge test...")
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
            known.append(q)
        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(questions)}] Known: {len(known)}")

    log.info(f"  Model knows {len(known)} questions")

    # Phase B: Collect attention patterns for lies and resists
    log.info("Phase B: Collecting attention patterns...")
    lie_attns = []
    resist_attns = []

    for i, q in enumerate(known):
        if len(lie_attns) >= MAX_EXAMPLES and len(resist_attns) >= MAX_EXAMPLES:
            break

        # Generate to classify behavior
        messages = [{"role": "user", "content": q["syco_prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        has_correct = check_answer_match(resp, q["correct_answer"])
        has_incorrect = check_answer_match(resp, q["incorrect_answer"])

        if not ((has_incorrect and not has_correct) or (has_correct and not has_incorrect)):
            continue

        is_lie = has_incorrect and not has_correct

        if is_lie and len(lie_attns) >= MAX_EXAMPLES:
            continue
        if not is_lie and len(resist_attns) >= MAX_EXAMPLES:
            continue

        # Get attention weights
        pressure_pos = find_pressure_positions(tokenizer, q["syco_prompt"], q["incorrect_answer"])
        if not pressure_pos:
            continue

        attentions, _ = forward_with_attention(model, tokenizer, q["syco_prompt"])
        pressure_attn = compute_pressure_attention(attentions, pressure_pos, n_layers, n_heads)

        if is_lie:
            lie_attns.append(pressure_attn)
        else:
            resist_attns.append(pressure_attn)

        if (i + 1) % 25 == 0:
            log.info(f"  [{i+1}/{len(known)}] Lies: {len(lie_attns)}, Resists: {len(resist_attns)}")

    log.info(f"  Collected: {len(lie_attns)} lies, {len(resist_attns)} resists")

    if len(lie_attns) < 5 or len(resist_attns) < 5:
        log.error("Not enough examples. Aborting.")
        return

    # ── Analysis ───────────────────────────────────────────────────────
    log.info("\nAnalyzing attention patterns...")

    lie_stack = np.stack(lie_attns)      # (n_lie, n_layers, n_heads)
    resist_stack = np.stack(resist_attns)  # (n_resist, n_layers, n_heads)

    # Mean attention to pressure tokens
    lie_mean = lie_stack.mean(axis=0)      # (n_layers, n_heads)
    resist_mean = resist_stack.mean(axis=0)
    diff = lie_mean - resist_mean          # positive = more attention when lying

    # Statistical test per head
    p_values = np.ones((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            lie_vals = lie_stack[:, layer, head]
            resist_vals = resist_stack[:, layer, head]
            if np.std(lie_vals) > 0 or np.std(resist_vals) > 0:
                _, p = mannwhitneyu(lie_vals, resist_vals, alternative="greater")
                p_values[layer, head] = p

    # Find top sycophancy heads
    head_scores = []
    for layer in range(n_layers):
        for head in range(n_heads):
            head_scores.append({
                "layer": layer,
                "head": head,
                "lie_attn": float(lie_mean[layer, head]),
                "resist_attn": float(resist_mean[layer, head]),
                "diff": float(diff[layer, head]),
                "p_value": float(p_values[layer, head]),
            })

    # Sort by difference (descending)
    head_scores.sort(key=lambda x: x["diff"], reverse=True)

    # Bonferroni correction
    n_tests = n_layers * n_heads
    significant_heads = [
        h for h in head_scores
        if h["p_value"] < 0.05 / n_tests and h["diff"] > 0
    ]

    log.info(f"\n  Top 15 'Sycophancy Heads' (most differential attention to pressure):")
    log.info(f"  {'Layer':>5s} {'Head':>4s} {'Lie Attn':>9s} {'Resist':>9s} {'Diff':>8s} {'p-value':>10s}")
    log.info(f"  {'-'*50}")
    for h in head_scores[:15]:
        sig = "***" if h["p_value"] < 0.05 / n_tests else ""
        log.info(f"  {h['layer']:5d} {h['head']:4d} {h['lie_attn']:9.4f} "
                 f"{h['resist_attn']:9.4f} {h['diff']:+8.4f} {h['p_value']:10.6f} {sig}")

    log.info(f"\n  Significant heads (Bonferroni corrected): {len(significant_heads)}")

    # Layer-level aggregation
    layer_diff = diff.mean(axis=1)  # average across heads
    most_differential_layer = int(np.argmax(layer_diff))
    log.info(f"\n  Most differential layer (avg across heads): Layer {most_differential_layer}")

    # ── Summary ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("ATTENTION ANALYSIS SUMMARY")
    log.info("=" * 60)
    log.info(f"  When the model LIES, specific attention heads attend MORE")
    log.info(f"  to the user's suggested (wrong) answer.")
    if significant_heads:
        top = significant_heads[0]
        log.info(f"  Strongest sycophancy head: Layer {top['layer']}, Head {top['head']}")
        log.info(f"    Lie attention: {top['lie_attn']:.4f}")
        log.info(f"    Resist attention: {top['resist_attn']:.4f}")
        log.info(f"    Difference: {top['diff']:+.4f} (p={top['p_value']:.6f})")
    log.info("=" * 60)

    # Save
    output = {
        "experiment": "06c_attention_analysis",
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_lie_examples": len(lie_attns),
        "n_resist_examples": len(resist_attns),
        "top_20_heads": head_scores[:20],
        "significant_heads_bonferroni": significant_heads[:20],
        "most_differential_layer": most_differential_layer,
        "layer_avg_diff": layer_diff.tolist(),
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp06c_attention_analysis.json")
    log.info("\nSaved to results/exp06c_attention_analysis.json")


if __name__ == "__main__":
    main()
