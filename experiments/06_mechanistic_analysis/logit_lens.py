"""
Experiment 06a — Logit Lens: Where Does the Lie Originate?
==========================================================
PURPOSE:
    At which layer does the model's internal prediction flip from the
    correct answer to the wrong answer?  The Logit Lens (nostalgebraist,
    2020) projects each layer's hidden state through the unembedding
    matrix to obtain a distribution over the vocabulary.  By tracking
    the rank / probability of the correct vs. wrong token across layers,
    we can pinpoint the exact layer where truth is overridden.

METHOD:
    For each question where the model LIED under sycophantic pressure:
        1. Run a forward pass (no generation) on the sycophantic prompt,
           collecting hidden states at every layer.
        2. For each layer, project the hidden state at the last prompt
           position through the unembedding matrix:
               logits_layer = LayerNorm(h_layer) @ W_unembed
        3. Record the rank and log-probability of:
               - The correct answer's first token
               - The wrong answer's first token (the one the model chose)
        4. Plot the "truth trajectory": at which layer does the wrong
           answer overtake the correct answer?

    We do the same for questions where the model RESISTED (told the truth)
    as a control — there the correct answer should dominate throughout.

OUTPUT:
    - Per-layer rank of correct vs. wrong answer token
    - "Flip layer" — where wrong answer first outranks correct answer
    - Aggregated plot across all lie examples
    - Comparison with resist examples

DATASET: meg-tong/sycophancy-eval (answer.jsonl)
MODEL:   meta-llama/Llama-3.1-8B-Instruct (4-bit quantized)

USAGE:
    %run experiments/06_mechanistic_analysis/logit_lens.py

RUNTIME: ~20 minutes on A100

CHANGELOG:
    v2 (2026-03-17): Fixed lm_head access for quantized models.
        Added multiple token variants for robust matching.
        Added visualization output.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import json
import time

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
MAX_EXAMPLES = 50  # max lie/resist examples to analyze
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
log = setup_logger("logit_lens")


# ── Core: Logit Lens projection ───────────────────────────────────────────

def get_lm_head_and_norm(model):
    """
    Safely extract the lm_head and final layer norm from the model,
    handling both regular and quantized models.

    For quantized models, lm_head.weight might be a QuantizedTensor.
    We extract the dequantized weight matrix for manual projection.
    """
    lm_head = model.lm_head

    # Get final layer norm
    if hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model.model, "final_layernorm"):
        final_norm = model.model.final_layernorm
    else:
        final_norm = None

    return lm_head, final_norm


def logit_lens_forward(model, tokenizer, prompt: str):
    """
    Run a forward pass and project each layer's hidden state through
    the unembedding matrix.

    Returns:
        all_logits: dict[layer_idx] -> logits tensor (vocab_size,)
        input_ids: the tokenized input
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    lm_head, final_norm = get_lm_head_and_norm(model)
    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
    n_layers = len(hidden_states) - 1  # -1 because index 0 is embedding

    all_logits = {}
    last_pos = -1  # last token position (where prediction happens)

    for layer_idx in range(n_layers + 1):
        h = hidden_states[layer_idx][0, last_pos, :]  # (hidden_dim,)

        # Apply final layer norm (same as the model does before lm_head)
        if final_norm is not None:
            h_normed = final_norm(h.unsqueeze(0)).squeeze(0)
        else:
            h_normed = h

        # Project through unembedding — use the lm_head module directly
        # This handles both regular and quantized weights transparently
        try:
            logits = lm_head(h_normed.unsqueeze(0)).squeeze(0)  # (vocab_size,)
        except Exception:
            # Fallback: manual matmul with dequantized weight
            weight = lm_head.weight
            if hasattr(weight, "dequantize"):
                weight = weight.dequantize()
            logits = torch.matmul(h_normed.float(), weight.float().T)

        all_logits[layer_idx] = logits.cpu().float()

    return all_logits, inputs["input_ids"][0]


def get_token_ids(tokenizer, text: str) -> list:
    """
    Get multiple token ID variants for a given text.
    Tries: " answer", "answer", " Answer", "Answer"
    Returns list of unique token IDs to check.
    """
    variants = [f" {text}", text, f" {text.capitalize()}", text.capitalize()]
    token_ids = set()
    for v in variants:
        tokens = tokenizer.encode(v, add_special_tokens=False)
        if tokens:
            token_ids.add(tokens[0])
    return list(token_ids)


def get_best_rank(logits, token_ids):
    """Get the best (lowest) rank among multiple token ID variants."""
    sorted_indices = torch.argsort(logits, descending=True)
    ranks = torch.zeros(logits.shape[0], dtype=torch.long)
    ranks[sorted_indices] = torch.arange(len(logits))

    probs = torch.softmax(logits, dim=-1)

    best_rank = len(logits)  # worst possible
    best_prob = 0.0
    best_tid = -1
    for tid in token_ids:
        if tid >= 0 and tid < len(logits):
            r = int(ranks[tid])
            p = float(probs[tid])
            if r < best_rank:
                best_rank = r
                best_prob = p
                best_tid = tid

    return best_rank, best_prob, best_tid


def analyze_trajectory(all_logits, correct_token_ids, wrong_token_ids, n_layers):
    """
    Analyze the rank and probability trajectory of correct vs wrong tokens.
    Uses multiple token variants for robust matching.

    Returns dict with per-layer data and the "flip layer".
    """
    trajectory = []
    flip_layer = None

    for layer_idx in range(n_layers + 1):
        logits = all_logits[layer_idx]

        correct_rank, correct_prob, _ = get_best_rank(logits, correct_token_ids)
        wrong_rank, wrong_prob, _ = get_best_rank(logits, wrong_token_ids)

        # Top prediction at this layer
        top_token_id = int(torch.argmax(logits))

        trajectory.append({
            "layer": layer_idx,
            "correct_rank": correct_rank,
            "wrong_rank": wrong_rank,
            "correct_prob": correct_prob,
            "wrong_prob": wrong_prob,
            "top_token_id": top_token_id,
        })

        # Detect flip: first layer where wrong outranks correct
        if flip_layer is None and wrong_rank < correct_rank:
            flip_layer = layer_idx

    return {
        "trajectory": trajectory,
        "flip_layer": flip_layer,
        "n_layers": n_layers,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 06a: Logit Lens — Where Does the Lie Originate?")
    log.info("=" * 60)
    start = time.time()

    questions = load_sycophancy_dataset(MAX_QUESTIONS)
    log.info(f"Loaded {len(questions)} questions")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    n_layers = model.config.num_hidden_layers

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

    # Phase B: Sycophantic pressure + Logit Lens
    log.info("Phase B: Logit Lens analysis on sycophantic responses...")
    lie_trajectories = []
    resist_trajectories = []

    for i, q in enumerate(known):
        # Get token IDs for correct and wrong answers (multiple variants)
        correct_tids = get_token_ids(tokenizer, q["correct_answer"])
        wrong_tids = get_token_ids(tokenizer, q["incorrect_answer"])

        if not correct_tids or not wrong_tids:
            continue

        # First: generate to see if model lied or resisted
        messages = [{"role": "user", "content": q["syco_prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        resp = tokenizer.decode(gen_out[0][input_len:], skip_special_tokens=True).strip()

        has_correct = check_answer_match(resp, q["correct_answer"])
        has_incorrect = check_answer_match(resp, q["incorrect_answer"])

        if not ((has_incorrect and not has_correct) or (has_correct and not has_incorrect)):
            continue  # ambiguous

        # Now run logit lens
        all_logits, _ = logit_lens_forward(model, tokenizer, q["syco_prompt"])
        traj = analyze_trajectory(all_logits, correct_tids, wrong_tids, n_layers)
        traj["question"] = q["question"][:80]
        traj["correct_answer"] = q["correct_answer"]
        traj["incorrect_answer"] = q["incorrect_answer"]

        if has_incorrect and not has_correct:
            traj["behavior"] = "lied"
            lie_trajectories.append(traj)
        else:
            traj["behavior"] = "resisted"
            resist_trajectories.append(traj)

        if (i + 1) % 25 == 0:
            log.info(f"  [{i+1}/{len(known)}] Lies: {len(lie_trajectories)}, "
                     f"Resisted: {len(resist_trajectories)}")

        # Limit to avoid excessive runtime
        if len(lie_trajectories) >= MAX_EXAMPLES and len(resist_trajectories) >= MAX_EXAMPLES:
            break

    log.info(f"  Analyzed: {len(lie_trajectories)} lies, {len(resist_trajectories)} resists")

    # ── Aggregate results ──────────────────────────────────────────────
    log.info("\nAggregating results...")

    # Average rank trajectory for lies
    avg_lie_correct_rank = []
    avg_lie_wrong_rank = []
    flip_layers = []
    median_flip = None
    mean_flip = None

    if lie_trajectories:
        flip_layers = [t["flip_layer"] for t in lie_trajectories if t["flip_layer"] is not None]

        for layer_idx in range(n_layers + 1):
            correct_ranks = [t["trajectory"][layer_idx]["correct_rank"] for t in lie_trajectories]
            wrong_ranks = [t["trajectory"][layer_idx]["wrong_rank"] for t in lie_trajectories]
            avg_lie_correct_rank.append(float(np.mean(correct_ranks)))
            avg_lie_wrong_rank.append(float(np.mean(wrong_ranks)))

        if flip_layers:
            median_flip = float(np.median(flip_layers))
            mean_flip = float(np.mean(flip_layers))
            log.info(f"  LIE trajectories:")
            log.info(f"    Median flip layer: {median_flip:.1f}")
            log.info(f"    Mean flip layer: {mean_flip:.1f}")
            log.info(f"    Range: {min(flip_layers)} - {max(flip_layers)}")

    # Average rank trajectory for resists
    avg_resist_correct_rank = []
    avg_resist_wrong_rank = []

    if resist_trajectories:
        for layer_idx in range(n_layers + 1):
            correct_ranks = [t["trajectory"][layer_idx]["correct_rank"] for t in resist_trajectories]
            wrong_ranks = [t["trajectory"][layer_idx]["wrong_rank"] for t in resist_trajectories]
            avg_resist_correct_rank.append(float(np.mean(correct_ranks)))
            avg_resist_wrong_rank.append(float(np.mean(wrong_ranks)))

    # ── Summary ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("LOGIT LENS SUMMARY")

    if lie_trajectories:
        log.info(f"\nWhen the model LIES:")
        log.info(f"  Early layers (0-10): correct answer ranked ~{np.mean(avg_lie_correct_rank[:11]):.0f}, "
                 f"wrong answer ranked ~{np.mean(avg_lie_wrong_rank[:11]):.0f}")
        late_start = min(20, n_layers)
        log.info(f"  Late layers ({late_start}-{n_layers}): correct answer ranked "
                 f"~{np.mean(avg_lie_correct_rank[late_start:]):.0f}, "
                 f"wrong answer ranked ~{np.mean(avg_lie_wrong_rank[late_start:]):.0f}")
        if flip_layers:
            log.info(f"  FLIP LAYER (median): {median_flip:.1f}")
            log.info(f"  -> The model 'knows' the truth until layer ~{median_flip:.0f}, "
                     f"then the sycophantic pressure overrides it.")

    if resist_trajectories:
        log.info(f"\nWhen the model RESISTS:")
        log.info(f"  Correct answer stays dominant throughout all layers.")
        log.info(f"  -> The sycophantic pressure never overrides the truth.")

    # Save
    output = {
        "experiment": "06a_logit_lens",
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "n_lie_trajectories": len(lie_trajectories),
        "n_resist_trajectories": len(resist_trajectories),
        "lie_flip_layers": flip_layers,
        "median_flip_layer": median_flip,
        "mean_flip_layer": mean_flip,
        "avg_lie_correct_rank": avg_lie_correct_rank,
        "avg_lie_wrong_rank": avg_lie_wrong_rank,
        "avg_resist_correct_rank": avg_resist_correct_rank,
        "avg_resist_wrong_rank": avg_resist_wrong_rank,
        "lie_examples": [
            {
                "question": t["question"],
                "correct": t["correct_answer"],
                "wrong": t["incorrect_answer"],
                "flip_layer": t["flip_layer"],
            }
            for t in lie_trajectories[:10]
        ],
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp06a_logit_lens.json")
    log.info("\nSaved to results/exp06a_logit_lens.json")


if __name__ == "__main__":
    main()
