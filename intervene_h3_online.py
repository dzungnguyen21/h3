"""
intervene_h3_online.py
======================
Single-pass H3 Activation Steering (Surgical Trigger Edition).
"""

import torch
import numpy as np
from collections import defaultdict
from config import MAX_TOKENS, PROMPT
import inspect


# ---------------------------------------------------------------------------
# Helper: build the set of COCO object token IDs with Stop-Word Filtering
# ---------------------------------------------------------------------------

def _build_coco_token_ids(tokenizer, evaluator) -> set:
    token_ids = set()
    all_surface_forms = list(evaluator.inverse_synonym_dict.keys())
    
    # Common English stop-words and prefixes that should NEVER trigger a noun-state probe
    stop_words = {"a", "an", "the", "in", "on", "of", "and", "is", "to", "with", "for", "it", "at", "by", "as"}
    
    for word in all_surface_forms:
        for prefix in [word, " " + word]:
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            for tid in ids:
                decoded = tokenizer.decode([tid]).strip().lower()
                # 1. Must not be empty
                # 2. Must be longer than 1 character (prevents triggering on "s" or "O")
                # 3. Must not be a common stop word
                if decoded and len(decoded) > 1 and decoded not in stop_words:
                    token_ids.add(tid)
    return token_ids


# ---------------------------------------------------------------------------
# Helper: partial forward — layers[start..end-1] → hidden state
# ---------------------------------------------------------------------------

def _partial_forward(language_model, h, layer_start, layer_end,
                     attention_mask, position_ids, pkv_slice):
    present_kvs = []
    
    # Calculate position embeddings required by newer Transformers versions
    kwargs = {}
    if hasattr(language_model, "rotary_emb"):
        sig = inspect.signature(language_model.layers[0].forward)
        if "position_embeddings" in sig.parameters:
            kwargs["position_embeddings"] = language_model.rotary_emb(h, position_ids)

    for rel_idx, layer_idx in enumerate(range(layer_start, layer_end)):
        layer = language_model.layers[layer_idx]
        past_kv = pkv_slice[rel_idx] if pkv_slice is not None else None
        
        layer_out = layer(
            h, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_kv, use_cache=True, output_attentions=False,
            **kwargs
        )
        # Safely extract hidden state regardless of tuple format
        if isinstance(layer_out, tuple):
            h = layer_out[0]
            if len(layer_out) > 1:
                present_kvs.append(layer_out[1])
        else:
            h = layer_out
            
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
    top_k_coco: int = 1,  # SURGICAL TRIGGER: Only steer if model is fully committed to a noun
    max_new_tokens: int = MAX_TOKENS,
):
    best_layer  = h3_results["best_layer"]
    scaler      = h3_results["scaler"]
    direction   = h3_results["direction"]
    clf         = h3_calibrated_results["clf"]
    feat_scaler = h3_calibrated_results["feat_scaler"]
    prob_thresh = h3_calibrated_results["prob_threshold"]

    d_norm    = direction / (np.linalg.norm(direction) + 1e-12)
    coco_ids_set = _build_coco_token_ids(processor.tokenizer, evaluator)

    inputs      = processor(images=image, text=PROMPT, return_tensors="pt")
    inputs      = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len  = inputs["input_ids"].shape[1]

    lang_model  = model.model.language_model
    n_layers    = len(lang_model.layers)
    lm_head     = model.lm_head
    final_norm  = lang_model.norm

    captured_h = [None]
    def _capture_hook(module, inp, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.dim() == 3:
            captured_h[0] = hidden_states[0, -1, :].detach().float().cpu()
        elif hidden_states.dim() == 2:
            captured_h[0] = hidden_states[-1, :].detach().float().cpu()
        else:
            captured_h[0] = hidden_states.detach().float().cpu()
        return output

    hook_handle = lang_model.layers[best_layer].register_forward_hook(_capture_hook)

    generated_ids  = []
    step_metadata  = []
    past_key_values = None
    first_step = True

    try:
        for step in range(max_new_tokens):
            if past_key_values is not None:
                # Robust integer indexing for Cache compatibility
                old_upper_pkv = [
                    past_key_values[i] for i in range(best_layer + 1, n_layers)
                ]
            else:
                old_upper_pkv = None

            with torch.inference_mode():
                if first_step:
                    fwd_out = model(**inputs, use_cache=True, return_dict=True)
                    first_step = False
                else:
                    current_ids = torch.tensor([[generated_ids[-1]]], device=model.device)
                    fwd_out = model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True, return_dict=True)

            logits          = fwd_out.logits[0, -1, :]
            past_key_values = fwd_out.past_key_values

            # SURGICAL TRIGGER: Check top_k=1 instead of 50
            top_k_ids     = torch.topk(logits, top_k_coco).indices
            coco_in_top_k = bool(any(tid.item() in coco_ids_set for tid in top_k_ids))

            did_steer = False
            h3_proj, h3_prob = None, None

            if coco_in_top_k:
                h_np    = captured_h[0].numpy()
                h_std   = (h_np - scaler.mean_) / scaler.scale_
                h3_proj = float(h_std @ direction)
                feat    = feat_scaler.transform(np.array([[h3_proj]], dtype=np.float32))
                h3_prob = float(clf.predict_proba(feat)[0, 1])

                if h3_prob < prob_thresh:
                    h_cpu   = captured_h[0].clone()
                    h_steer = h_cpu + steering_strength * torch.tensor(d_norm, dtype=torch.float32)
                    h_steer_dev = h_steer.to(model.device).to(next(lang_model.parameters()).dtype).unsqueeze(0).unsqueeze(0)

                    seq_len_so_far = prompt_len + len(generated_ids)
                    position_ids = torch.tensor([[seq_len_so_far - 1]], device=model.device)
                    
                    # No attention mask needed for single-token decoding
                    attn_mask = None

                    with torch.inference_mode():
                        h_out, _ = _partial_forward(lang_model, h_steer_dev, best_layer + 1, n_layers, attn_mask, position_ids, old_upper_pkv)
                        corrected_logits = lm_head(final_norm(h_out))[0, -1, :]

                    logits    = corrected_logits
                    did_steer = True

            next_token_id = int(logits.argmax().item())
            if next_token_id == processor.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            step_metadata.append({
                "step": step, "token_id": next_token_id, "token_str": processor.tokenizer.decode([next_token_id]),
                "coco_in_topk": coco_in_top_k, "steered": did_steer, "h3_proj": h3_proj, "h3_prob": h3_prob,
            })

    finally:
        hook_handle.remove()

    caption = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return caption, step_metadata


def intervene_h3_online(model, processor, image, h3_results, h3_calibrated_results, evaluator, steering_strength=3.0, top_k_coco=1, max_new_tokens=MAX_TOKENS):
    # Notice we pass top_k_coco=1 down to the steering function now
    caption, metadata = generate_with_h3_steering(model, processor, image, h3_results, h3_calibrated_results, evaluator, steering_strength, top_k_coco, max_new_tokens)
    return caption, metadata, sum(1 for m in metadata if m["steered"])