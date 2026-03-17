"""
Experiment 06b — Activation Patching: Causal Evidence for Deception
===================================================================
PURPOSE:
    Provide CAUSAL (not just correlational) evidence for where deception
    originates.  Activation patching (Meng et al., 2022; Vig et al., 2020)
    replaces the hidden state at a specific layer from one run with the
    hidden state from another run, and measures the effect on the output.

    If patching layer L from a "resist" run into a "lie" run causes the
    model to tell the truth → layer L is causally responsible for the lie.

METHOD:
    For each question where the model lies under sycophantic pressure:

    1. CLEAN RUN (resist):
       Run the neutral prompt → model answers correctly.
       Save hidden states at every layer.

    2. CORRUPTED RUN (lie):
       Run the sycophantic prompt → model lies.
       Save hidden states at every layer.

    3. PATCHING:
       For each layer L:
           - Start with the corrupted (lie) run
           - Replace h_L with the clean (truth) hidden state
           - Continue the forward pass from layer L+1
           - Check: did the output flip from lie → truth?

    4. METRIC:
       "Recovery rate" = fraction of examples where patching layer L
       caused the output to flip from wrong to correct.

    The layer with the highest recovery rate is the one most causally
    responsible for the deception.

IMPORTANT NOTE:
    This requires running the model WITHOUT quantization for clean
    activation patching.  If GPU memory is insufficient, we fall back
    to a "logit-based" patching approximation.

DATASET: meg-tong/sycophancy-eval (answer.jsonl)
MODEL:   meta-llama/Llama-3.1-8B-Instruct

USAGE:
    %run experiments/06_mechanistic_analysis/activation_patching.py

RUNTIME: ~30-45 minutes on A100 80GB
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import time
import gc

from src.utils import (
    setup_logger,
    load_sycophancy_dataset,
    check_answer_match,
    save_results,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_QUESTIONS = 500
MAX_NEW_TOKENS = 80
MAX_PATCH_EXAMPLES = 30  # Patching is expensive — limit examples
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
log = setup_logger("act_patch")


# ── Helper: forward pass with hidden state collection ──────────────────────

def forward_with_hidden_states(model, tokenizer, prompt):
    """
    Run a forward pass and return all hidden states + the predicted logits.
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

    return {
        "hidden_states": outputs.hidden_states,  # tuple of (1, seq_len, hidden_dim)
        "logits": outputs.logits,                 # (1, seq_len, vocab_size)
        "input_ids": inputs["input_ids"],
    }


def patch_and_predict(model, tokenizer, prompt, clean_hidden_states, patch_layer):
    """
    Run a forward pass on `prompt`, but replace the hidden state at
    `patch_layer` with the corresponding state from `clean_hidden_states`.

    Uses a forward hook to inject the clean activation at the LAST
    sequence position. This handles the sequence length mismatch between
    the neutral prompt (shorter) and the sycophantic prompt (longer):
    we always patch at position -1 (the prediction position), which is
    semantically equivalent regardless of sequence length.

    Returns the top-1 predicted token and the logits.
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # The clean hidden state to inject (last position of the clean run)
    # Note: clean_hidden_states may have a different seq_len than `prompt`,
    # because the neutral prompt is shorter than the sycophantic prompt.
    # We always take position -1 (the prediction position) from the clean run
    # and inject it at position -1 of the corrupted run.
    clean_h = clean_hidden_states[patch_layer][0, -1, :].clone()

    # Register hook on the target layer
    hook_handle = None
    target_module = model.model.layers[patch_layer - 1] if patch_layer > 0 else None

    if target_module is not None:
        def hook_fn(module, input, output):
            # output is typically a tuple; first element is the hidden state
            # Shape: (batch, seq_len, hidden_dim)
            if isinstance(output, tuple):
                h = output[0]
                # Only patch the last position (prediction position)
                # This avoids the seq_len mismatch issue entirely
                h[0, -1, :] = clean_h.to(h.device, dtype=h.dtype)
                return (h,) + output[1:]
            else:
                output[0, -1, :] = clean_h.to(output.device, dtype=output.dtype)
                return output

        hook_handle = target_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    if hook_handle is not None:
        hook_handle.remove()

    # Get top prediction at last position
    logits = outputs.logits[0, -1, :]
    top_token = int(torch.argmax(logits))
    return top_token, logits


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 06b: Activation Patching")
    log.info("=" * 60)
    start = time.time()

    questions = load_sycophancy_dataset(MAX_QUESTIONS)
    log.info(f"Loaded {len(questions)} questions")

    # Try loading without quantization for clean patching
    hf_token = os.environ.get("HF_TOKEN", "")
    log.info(f"Loading {MODEL_NAME}...")

    try:
        # Try full precision first
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True,
            token=hf_token,
        )
        log.info("  Loaded in float16 (no quantization)")
    except Exception:
        log.info("  float16 failed, falling back to 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            output_hidden_states=True,
            token=hf_token,
        )
        log.info("  Loaded in 4-bit (patching results may be approximate)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers

    # Phase A: Find questions where model knows and lies
    log.info("Phase A: Finding lie examples...")
    lie_examples = []

    for i, q in enumerate(questions):
        if len(lie_examples) >= MAX_PATCH_EXAMPLES:
            break

        # Check if model knows (neutral)
        messages = [{"role": "user", "content": q["neutral_prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        resp_neutral = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        if not check_answer_match(resp_neutral, q["correct_answer"]):
            continue

        # Check if model lies (sycophantic)
        messages = [{"role": "user", "content": q["syco_prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        resp_syco = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        has_correct = check_answer_match(resp_syco, q["correct_answer"])
        has_incorrect = check_answer_match(resp_syco, q["incorrect_answer"])

        if has_incorrect and not has_correct:
            lie_examples.append(q)

        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(questions)}] Lie examples: {len(lie_examples)}")

    log.info(f"  Found {len(lie_examples)} lie examples")

    # Phase B: Activation patching
    log.info(f"\nPhase B: Patching {len(lie_examples)} examples across {n_layers} layers...")

    # Track recovery per layer
    recovery_counts = np.zeros(n_layers + 1)  # +1 for embedding
    total_examples = 0

    for ex_i, q in enumerate(lie_examples):
        # Get clean hidden states (neutral prompt)
        clean_result = forward_with_hidden_states(model, tokenizer, q["neutral_prompt"])
        clean_hs = clean_result["hidden_states"]

        # Get correct answer token
        correct_tid = tokenizer.encode(" " + q["correct_answer"], add_special_tokens=False)
        if not correct_tid:
            continue
        correct_tid = correct_tid[0]

        total_examples += 1

        # Patch each layer
        for layer_idx in range(1, n_layers + 1):  # skip 0 (embedding)
            try:
                top_token, _ = patch_and_predict(
                    model, tokenizer, q["syco_prompt"], clean_hs, layer_idx
                )
                if top_token == correct_tid:
                    recovery_counts[layer_idx] += 1
            except Exception as e:
                log.warning(f"  Patching failed at layer {layer_idx}: {e}")
                continue

        if (ex_i + 1) % 5 == 0:
            log.info(f"  [{ex_i+1}/{len(lie_examples)}] done")

    # ── Results ────────────────────────────────────────────────────────
    if total_examples == 0:
        log.error("No examples processed. Aborting.")
        return

    recovery_rates = recovery_counts / total_examples

    log.info("\n" + "=" * 60)
    log.info("ACTIVATION PATCHING RESULTS")
    log.info("=" * 60)
    log.info(f"  Total examples: {total_examples}")
    log.info(f"\n  Recovery rate per layer (fraction of lies → truth after patching):")

    best_layer = int(np.argmax(recovery_rates[1:])) + 1
    for layer_idx in range(1, n_layers + 1):
        bar = "█" * int(recovery_rates[layer_idx] * 40)
        marker = " ← BEST" if layer_idx == best_layer else ""
        log.info(f"    Layer {layer_idx:2d}: {recovery_rates[layer_idx]*100:5.1f}% {bar}{marker}")

    log.info(f"\n  MOST CAUSAL LAYER: {best_layer} "
             f"({recovery_rates[best_layer]*100:.1f}% recovery)")
    log.info(f"  → Patching layer {best_layer} from truth into lie "
             f"recovers the correct answer {recovery_rates[best_layer]*100:.0f}% of the time.")

    # Save
    output = {
        "experiment": "06b_activation_patching",
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "total_examples": total_examples,
        "recovery_rates": recovery_rates.tolist(),
        "best_layer": best_layer,
        "best_recovery_rate": float(recovery_rates[best_layer]),
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp06b_activation_patching.json")
    log.info("\nSaved to results/exp06b_activation_patching.json")


if __name__ == "__main__":
    main()
