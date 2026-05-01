import torch
import numpy as np
from collections import defaultdict
from config import MAX_TOKENS, PROMPT

def _build_coco_token_ids(tokenizer, evaluator):
    """
    Collect all token IDs for COCO object words and synonyms.
    Includes the subword and capitalization fixes.
    """
    token_ids = set()
    all_surface_forms = list(evaluator.inverse_synonym_dict.keys())
    for word in all_surface_forms:
        variants_to_check = [word.lower(), word.capitalize(), word.upper()]
        for v in variants_to_check:
            for prefix in [v, " " + v]:
                ids = tokenizer.encode(prefix, add_special_tokens=False)
                # Prevent subword collateral damage: only ban single-token words
                if len(ids) == 1:
                    tid = ids[0]
                    if tokenizer.decode([tid]).strip():
                        token_ids.add(tid)
    return token_ids


def intervene_h3_local(
    model,
    processor,
    image,
    h3_results,
    h3_calibrated_results,
    evaluator,
    suppression_strength: float = 5.0,
    top_k_coco: int = 50,
    max_new_tokens: int = MAX_TOKENS,
):
    """
    Single-pass LOCAL Logit Suppression.
    Instead of suppressing a word for the entire generation (which causes 
    semantic substitutions like 'handbag' -> 'backpack'), this intercepts 
    generation at the exact step the model tries to hallucinate. 
    It applies the penalty for *that step only*, forcing a grammatical pivot.
    """
    best_layer  = h3_results["best_layer"]
    scaler      = h3_results["scaler"]
    direction   = h3_results["direction"]
    clf         = h3_calibrated_results["clf"]
    feat_scaler = h3_calibrated_results["feat_scaler"]
    prob_thresh = h3_calibrated_results["prob_threshold"]

    coco_ids_set = _build_coco_token_ids(processor.tokenizer, evaluator)

    inputs = processor(images=image, text=PROMPT, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    lang_model = model.model.language_model
    
    # Hook to capture hidden state BEFORE the LM head sees it
    captured_h = [None]
    def _capture_hook(module, inp, output):
        # Safely handle both tuple and raw tensor outputs from the layer
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        if hidden_states.dim() == 3:
            captured_h[0] = hidden_states[0, -1, :].detach().float().cpu()
        elif hidden_states.dim() == 2:
            captured_h[0] = hidden_states[-1, :].detach().float().cpu()
        return output

    hook_handle = lang_model.layers[best_layer].register_forward_hook(_capture_hook)

    generated_ids = []
    past_key_values = None
    first_step = True

    try:
        for step in range(max_new_tokens):
            with torch.inference_mode():
                if first_step:
                    fwd_out = model(
                        **inputs,
                        use_cache=True,
                        return_dict=True,
                    )
                    first_step = False
                else:
                    current_ids = torch.tensor([[generated_ids[-1]]], device=model.device)
                    fwd_out = model(
                        input_ids=current_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

            # We clone the logits so we can safely mutate them for THIS step only
            logits = fwd_out.logits[0, -1, :].clone()
            past_key_values = fwd_out.past_key_values

            # Check if the model is highly prioritizing a COCO object noun right now
            top_k_ids = torch.topk(logits, top_k_coco).indices
            coco_in_top_k = any(tid.item() in coco_ids_set for tid in top_k_ids)

            if coco_in_top_k:
                # The model wants to output a noun. Run the H3 Probe.
                h_np    = captured_h[0].numpy()
                h_std   = (h_np - scaler.mean_) / scaler.scale_
                h3_proj = float(h_std @ direction)
                feat    = feat_scaler.transform(np.array([[h3_proj]], dtype=np.float32))
                h3_prob = float(clf.predict_proba(feat)[0, 1])

                if h3_prob < prob_thresh:
                    # HALLUCINATION DETECTED!
                    # Apply local suppression: Penalize ALL COCO object tokens 
                    # for THIS STEP ONLY. 
                    # Because we penalize "backpack" alongside "handbag", the model 
                    # cannot substitute. It is forced to pivot to a non-object token
                    # (like an adjective, or a preposition).
                    for tid in coco_ids_set:
                        if tid < logits.shape[-1]:
                            logits[tid] -= suppression_strength

            # Sample from the (potentially suppressed) logits
            next_token_id = int(logits.argmax().item())

            if next_token_id == processor.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)

    finally:
        hook_handle.remove()

    caption = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return caption