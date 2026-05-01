from config import MAX_TOKENS, PROMPT
from collections import defaultdict
from transformers import LogitsProcessor
import torch
import numpy as np


def intervene_h3_calibrated(
    model,
    processor,
    image,
    h3_results,
    h3_calibrated_results,
    evaluator,
    suppression_strength=5.0,
    max_new_tokens=MAX_TOKENS,
):
    """
    Two-pass surgical suppression with calibrated H3 threshold.
    Returns both pass 1 and pass 2 captions so they can be
    evaluated and compared without running twice.
    """
    from transformers import LogitsProcessor
    from collections import defaultdict

    best_layer  = h3_results["best_layer"]
    scaler      = h3_results["scaler"]
    direction   = h3_results["direction"]
    clf         = h3_calibrated_results["clf"]
    feat_scaler = h3_calibrated_results["feat_scaler"]
    prob_thresh = h3_calibrated_results["prob_threshold"]

    base_to_syns = defaultdict(list)
    for syn, base in evaluator.inverse_synonym_dict.items():
        base_to_syns[base].append(syn)

    inputs = processor(images=image, text=PROMPT, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # ── Pass 1: dry-run generation ────────────────────────────────────
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    generated_ids  = generated[0][prompt_len:].tolist()
    pass1_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    pass1_text_lower = pass1_text.lower()

    # Detect mentioned objects in pass 1 output
    chair_result    = evaluator.compute_hallucinations(0, pass1_text_lower)
    mentioned_bases = set(chair_result["recall_words"]) | \
                      set(chair_result["mscoco_hallucinated_words"])

    # If nothing mentioned, pass 2 = pass 1
    if not mentioned_bases:
        return pass1_text, pass1_text, [], {}

    # ── Per-object H3 probe ───────────────────────────────────────────
    object_probs   = {}
    object_h3_proj = {}

    for base_obj in mentioned_bases:
        syns = sorted(
            base_to_syns.get(base_obj, [base_obj]), key=len, reverse=True
        )

        target_pos = None
        for i in range(len(generated_ids)):
            prefix = processor.tokenizer.decode(
                generated_ids[:i+1], skip_special_tokens=True
            ).lower()
            prev = processor.tokenizer.decode(
                generated_ids[:i], skip_special_tokens=True
            ).lower()
            for syn in syns:
                if syn in prefix and syn not in prev:
                    target_pos = i
                    break
            if target_pos is not None:
                break

        if target_pos is None:
            continue

        context_tensor = torch.tensor(
            [inputs["input_ids"][0].tolist() + generated_ids[:target_pos]],
            device=model.device
        )
        with torch.inference_mode():
            out = model.model.language_model(
                input_ids=context_tensor,
                output_hidden_states=True,
                return_dict=True,
            )
        h_vec   = out.hidden_states[best_layer + 1][0, -1, :]\
                     .float().cpu().numpy()
        h_std   = (h_vec - scaler.mean_) / scaler.scale_
        h3_proj = float(h_std @ direction)
        object_h3_proj[base_obj] = h3_proj

        proj_scaled = feat_scaler.transform(
            np.array([[h3_proj]], dtype=np.float32)
        )
        prob = float(clf.predict_proba(proj_scaled)[0, 1])
        object_probs[base_obj] = prob

    absent_bases = {
        obj for obj, prob in object_probs.items()
        if prob < prob_thresh
    }

    # If nothing to suppress, pass 2 = pass 1
    suppress_ids = set()
    for base_obj in absent_bases:
        for variant in base_to_syns.get(base_obj, [base_obj]):
            variants_to_check = [
                variant.lower(),
                variant.capitalize(),
                variant.upper()
            ]
            
            for v in variants_to_check:
                for prefix_str in [v, " " + v]:
                    encoded_tids = processor.tokenizer.encode(
                        prefix_str, add_special_tokens=False
                    )
                    
                    if len(encoded_tids) == 1:
                        tid = encoded_tids[0]
                        if processor.tokenizer.decode([tid]).strip():
                            suppress_ids.add(tid)

    if not suppress_ids:
        return pass1_text, pass1_text, [], object_h3_proj

    # ── Pass 2: suppressed generation ────────────────────────────────
    class SurgicalSuppressor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for tid in suppress_ids:
                if tid < scores.shape[-1]:
                    scores[:, tid] -= suppression_strength
            return scores

    with torch.inference_mode():
        out_int = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[SurgicalSuppressor()],
        )
    pass2_text = processor.tokenizer.decode(
        out_int[0][prompt_len:].tolist(), skip_special_tokens=True
    )

    # Returns: (pass1_caption, pass2_caption, absent_objects, projections)
    return pass1_text, pass2_text, list(absent_bases), object_h3_proj
