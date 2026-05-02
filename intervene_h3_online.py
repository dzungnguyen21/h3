"""
intervene_h3_online.py
======================
Single-pass H3 Activation Steering.

Key differences from intervene_h3_calibrated.py (output-level suppression):
  - Acts on the hidden state at `best_layer` BEFORE the LM head sees it
  - Only steers when a COCO object token is in top-k of the unsteered logits
  - Avoids the substitution problem: if "handbag" is blocked at the context
    level, the model is no longer in a "I am about to name a carried object"
    state — it does not substitute "backpack" instead.

Algorithm per decoding step t:
  1. Full forward pass (with KV cache) — hook captures h_t at best_layer
  2. Check: is any COCO token id in top-k of logits?   [one-line check]
  3. If yes: compute H3 projection from captured h_t
  4. If prob < threshold: steer h_t, re-run layers[best_layer+1 .. N] + norm
     + lm_head with the steered state, using the pre-step KV cache for those
     layers (saved cheaply as a reference slice before the full forward).
  5. Sample (greedy) from corrected logits.

KV cache note: the full forward (step 1) updates KV cache for ALL layers
including best_layer+1..N using the unsteered h_t. When we re-run those
layers with steered h_t we pass the pre-step KV cache slice so the current
position attends correctly to its own (steered) Q while past positions use
the previously cached (unsteered) K/V. This is the mild inconsistency the
user accepted as negligible.
"""

import torch
import numpy as np
from collections import defaultdict
from config import MAX_TOKENS, PROMPT


# ---------------------------------------------------------------------------
# Helper: build the set of COCO object token IDs
# ---------------------------------------------------------------------------

def _build_coco_token_ids(tokenizer, evaluator) -> set:
    """
    Collect all token IDs that can appear when the model writes a COCO object
    word (or its synonyms).  We index every surface form that CHAIR tracks.
    """
    token_ids = set()
    # evaluator.inverse_synonym_dict  maps  synonym -> base_category
    all_surface_forms = list(evaluator.inverse_synonym_dict.keys())
    for word in all_surface_forms:
        for prefix in [word, " " + word]:
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            for tid in ids:
                if tokenizer.decode([tid]).strip():   # skip whitespace-only tokens
                    token_ids.add(tid)
    return token_ids


# ---------------------------------------------------------------------------
# Helper: partial forward — layers[start..end-1] → hidden state
# ---------------------------------------------------------------------------

def _partial_forward(language_model, h, layer_start, layer_end,
                     attention_mask, position_ids, pkv_slice):
    """
    Run `h` (shape [1, 1, hidden]) through layers[layer_start : layer_end]
    using `pkv_slice` as past_key_values (tuple indexed 0..len(pkv_slice)-1).

    Returns the new hidden state [1, 1, hidden].
    """
    present_kvs = []
    for rel_idx, layer_idx in enumerate(range(layer_start, layer_end)):
        layer = language_model.layers[layer_idx]
        past_kv = pkv_slice[rel_idx] if pkv_slice is not None else None
        layer_out = layer(
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_kv,
            use_cache=True,
            output_attentions=False,
        )
        h = layer_out[0]
        present_kvs.append(layer_out[1])   # (k, v) for this layer+step
    return h, present_kvs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_with_h3_steering(
    model,
    processor,
    image,
    h3_results,
    h3_calibrated_results,
    evaluator,
    steering_strength: float = 3.0,
    top_k_coco: int = 50,
    max_new_tokens: int = MAX_TOKENS,
):
    """
    Generate a caption for `image` using single-pass H3 activation steering.

    Returns
    -------
    caption       : str   — the generated text
    step_metadata : list  — one dict per token with keys:
                            step, token_id, token_str,
                            coco_in_topk, steered, h3_proj, h3_prob
    """
    # ── Unpack trained artefacts ──────────────────────────────────────────
    best_layer  = h3_results["best_layer"]
    scaler      = h3_results["scaler"]           # StandardScaler fitted on hidden vecs
    direction   = h3_results["direction"]         # shape [hidden_dim]
    clf         = h3_calibrated_results["clf"]
    feat_scaler = h3_calibrated_results["feat_scaler"]
    prob_thresh = h3_calibrated_results["prob_threshold"]

    # Normalised steering vector (float32 numpy → torch)
    d_norm    = direction / (np.linalg.norm(direction) + 1e-12)
    d_tensor  = torch.tensor(d_norm, dtype=torch.float32)   # [hidden_dim]

    # ── COCO token set ────────────────────────────────────────────────────
    coco_ids_set = _build_coco_token_ids(processor.tokenizer, evaluator)

    # ── Encode prompt ─────────────────────────────────────────────────────
    inputs      = processor(images=image, text=PROMPT, return_tensors="pt")
    inputs      = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len  = inputs["input_ids"].shape[1]

    # References into the language model trunk
    lang_model  = model.model.language_model      # LlamaModel
    n_layers    = len(lang_model.layers)
    lm_head     = model.lm_head
    final_norm  = lang_model.norm

    # ── Hook: capture h_t at best_layer without modifying it ─────────────
    captured_h = [None]   # mutable 1-element list so the closure can write

    def _capture_hook(module, inp, output):
        # Safely handle both tuple and raw tensor outputs from the layer
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        if hidden_states.dim() == 3:
            captured_h[0] = hidden_states[0, -1, :].detach().float().cpu()
        elif hidden_states.dim() == 2:
            captured_h[0] = hidden_states[-1, :].detach().float().cpu()
        else:
            captured_h[0] = hidden_states.detach().float().cpu()
        return output

    hook_handle = lang_model.layers[best_layer].register_forward_hook(_capture_hook)

    # ── Manual decoding loop ──────────────────────────────────────────────
    generated_ids  = []
    step_metadata  = []
    past_key_values = None

    # Build initial input: use the pixel_values from processor for the first step
    # then only input_ids=[[next_token]] for subsequent steps.
    first_step = True

    try:
        for step in range(max_new_tokens):

            # Save the PRE-STEP KV cache for layers above best_layer.
            # These are plain tuple references — no clone needed because
            # HuggingFace Llama returns NEW concatenated tensors each step.
            if past_key_values is not None:
                # Safely handle both legacy tuples and modern DynamicCache objects
                if hasattr(past_key_values, "key_cache"):
                    old_upper_pkv = [
                        (past_key_values.key_cache[i], past_key_values.value_cache[i])
                        for i in range(best_layer + 1, len(past_key_values.key_cache))
                    ]
                else:
                    old_upper_pkv = past_key_values[best_layer + 1:]
            else:
                old_upper_pkv = None

            # ── Step 1: full forward pass ─────────────────────────────────
            with torch.inference_mode():
                if first_step:
                    fwd_out = model(
                        **inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    first_step = False
                else:
                    fwd_out = model(
                        input_ids=current_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

            logits          = fwd_out.logits[0, -1, :]     # [vocab]
            past_key_values = fwd_out.past_key_values       # updated (all layers)

            # ── Step 2: COCO check (one line) ────────────────────────────
            top_k_ids     = torch.topk(logits, top_k_coco).indices
            coco_in_top_k = bool(
                any(tid.item() in coco_ids_set for tid in top_k_ids)
            )

            did_steer = False
            h3_proj   = None
            h3_prob   = None

            if coco_in_top_k:
                # ── Step 3: H3 score from captured hidden state ───────────
                h_np    = captured_h[0].numpy()                    # [hidden]
                h_std   = (h_np - scaler.mean_) / scaler.scale_   # standardise
                h3_proj = float(h_std @ direction)
                feat    = feat_scaler.transform(
                    np.array([[h3_proj]], dtype=np.float32)
                )
                h3_prob = float(clf.predict_proba(feat)[0, 1])

                if h3_prob < prob_thresh:
                    # ── Step 4: steer and re-run upper layers ─────────────
                    # Steer: push h_t toward the grounded subspace
                    h_cpu   = captured_h[0].clone()                # [hidden]
                    h_steer = h_cpu + steering_strength * torch.tensor(d_norm, dtype=torch.float32)

                    # Move steered vector to model device as [1, 1, hidden]
                    h_steer_dev = h_steer.to(model.device).to(
                        next(lang_model.parameters()).dtype
                    ).unsqueeze(0).unsqueeze(0)

                    # Build position_ids for the current (single) token
                    seq_len_so_far = prompt_len + len(generated_ids)
                    position_ids = torch.tensor(
                        [[seq_len_so_far - 1]], device=model.device
                    )

                    # attention_mask: all ones over full past sequence
                    full_len     = seq_len_so_far
                    attn_mask    = torch.ones(
                        1, full_len, device=model.device,
                        dtype=torch.long
                    )

                    with torch.inference_mode():
                        h_out, _ = _partial_forward(
                            lang_model,
                            h_steer_dev,
                            layer_start = best_layer + 1,
                            layer_end   = n_layers,
                            attention_mask = attn_mask,
                            position_ids   = position_ids,
                            pkv_slice      = old_upper_pkv,
                        )
                        h_normed        = final_norm(h_out)         # [1, 1, hidden]
                        corrected_logits = lm_head(h_normed)[0, -1, :]  # [vocab]

                    logits    = corrected_logits
                    did_steer = True

            # ── Step 5: greedy sample ─────────────────────────────────────
            next_token_id = int(logits.argmax().item())

            if next_token_id == processor.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            current_ids = torch.tensor(
                [[next_token_id]], device=model.device
            )

            step_metadata.append({
                "step"        : step,
                "token_id"    : next_token_id,
                "token_str"   : processor.tokenizer.decode([next_token_id]),
                "coco_in_topk": coco_in_top_k,
                "steered"     : did_steer,
                "h3_proj"     : h3_proj,
                "h3_prob"     : h3_prob,
            })

    finally:
        hook_handle.remove()

    caption = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return caption, step_metadata


# ---------------------------------------------------------------------------
# Convenience wrapper: mirrors the signature of intervene_h3_calibrated
# so it can be swapped in with minimal changes to the calling code.
# ---------------------------------------------------------------------------

def intervene_h3_online(
    model,
    processor,
    image,
    h3_results,
    h3_calibrated_results,
    evaluator,
    steering_strength: float = 3.0,
    top_k_coco: int = 50,
    max_new_tokens: int = MAX_TOKENS,
):
    """
    Drop-in replacement for intervene_h3_calibrated().

    Returns
    -------
    caption       : str
    step_metadata : list[dict]
    n_steered     : int   — number of steps where steering was applied
    """
    caption, metadata = generate_with_h3_steering(
        model, processor, image,
        h3_results, h3_calibrated_results, evaluator,
        steering_strength=steering_strength,
        top_k_coco=top_k_coco,
        max_new_tokens=max_new_tokens,
    )
    n_steered = sum(1 for m in metadata if m["steered"])
    return caption, metadata, n_steered
