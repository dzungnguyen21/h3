"""
intervene_h3_single_suppress.py
===============================
Single-Pass Logit Suppression (The Ultimate Architecture).
Intercepts hidden states in real-time via Layer 0 hook, runs the H3 probe, 
and suppresses hallucination logits dynamically before sampling.
"""

import torch
import numpy as np
from transformers import LogitsProcessor
from collections import defaultdict
from config import MAX_TOKENS, PROMPT

def build_token_maps(processor, evaluator):
    """
    Precomputes the first tokens (for triggering the probe) and all tokens 
    (for suppressing the hallucination) for every COCO object.
    """
    first_token_to_bases = defaultdict(list)
    base_to_all_tids = defaultdict(set)
    
    base_to_syns = defaultdict(list)
    for syn, base in evaluator.inverse_synonym_dict.items():
        base_to_syns[base].append(syn)
        
    for base_obj, syns in base_to_syns.items():
        expanded_variants = set()
        for variant in syns + [base_obj]:
            variant = variant.strip()
            expanded_variants.add(variant)
            if not variant.endswith('s'):
                expanded_variants.add(variant + "s")
                if variant.endswith('ch') or variant.endswith('sh') or variant.endswith('x'):
                    expanded_variants.add(variant + "es")
                    
        for variant in expanded_variants:
            casing_options = [variant.lower(), variant.capitalize(), variant.upper()]
            for cased in casing_options:
                for prefix_str in [" " + cased, cased]:
                    tids = processor.tokenizer.encode(prefix_str, add_special_tokens=False)
                    real_tids = [t for t in tids if len(processor.tokenizer.decode([t]).strip()) > 0]
                    
                    if not real_tids:
                        continue
                        
                    # 1. Trigger map: Only trigger on the FIRST token of the word
                    first_token = real_tids[0]
                    if base_obj not in first_token_to_bases[first_token]:
                        first_token_to_bases[first_token].append(base_obj)
                        
                    # 2. Suppression map: Ban ALL subwords associated with the object
                    for tid in real_tids:
                        base_to_all_tids[base_obj].add(tid)
                        
    return dict(first_token_to_bases), dict(base_to_all_tids)

class H3SinglePassSuppressor(LogitsProcessor):
    def __init__(self, captured_h, h3_results, h3_calibrated_results, first_token_map, suppression_map, top_k=1, strength=10.0):
        self.captured_h = captured_h
        self.scaler = h3_results["scaler"]
        self.direction = h3_results["direction"]
        self.clf = h3_calibrated_results["clf"]
        self.feat_scaler = h3_calibrated_results["feat_scaler"]
        self.prob_thresh = h3_calibrated_results["prob_threshold"]
        
        self.first_token_map = first_token_map
        self.suppression_map = suppression_map
        self.top_k = top_k
        self.strength = strength
        self.n_steered = 0
        
    def __call__(self, input_ids, scores):
        logits = scores[0]
        
        # Check if the model is about to generate the start of a COCO object
        top_k_ids = torch.topk(logits, self.top_k).indices.tolist()
        
        bases_to_probe = set()
        for tid in top_k_ids:
            if tid in self.first_token_map:
                bases_to_probe.update(self.first_token_map[tid])
                
        if not bases_to_probe or self.captured_h[0] is None:
            return scores
            
        # Model is committed to a noun. Run the perfectly aligned H3 Probe!
        h_np = self.captured_h[0].numpy()
        h_std = (h_np - self.scaler.mean_) / self.scaler.scale_
        h3_proj = float(h_std @ self.direction)
        feat = self.feat_scaler.transform(np.array([[h3_proj]], dtype=np.float32))
        h3_prob = float(self.clf.predict_proba(feat)[0, 1])
        
        if h3_prob < self.prob_thresh:
            # HALLUCINATION DETECTED!
            self.n_steered += 1
            for base_obj in bases_to_probe:
                for suppress_tid in self.suppression_map.get(base_obj, set()):
                    if suppress_tid < scores.shape[-1]:
                        scores[0, suppress_tid] -= self.strength
                        
        return scores

def intervene_h3_single_suppress(
    model, processor, image, h3_results, h3_calibrated_results, evaluator, 
    suppression_strength=10.0, top_k_coco=1, max_new_tokens=MAX_TOKENS
):
    best_layer = h3_results["best_layer"]
    
    first_token_map, suppression_map = build_token_maps(processor, evaluator)
    
    inputs = processor(images=image, text=PROMPT, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Hook Layer 0 to capture the state continuously
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

    lang_model = model.model.language_model
    hook_handle = lang_model.layers[best_layer].register_forward_hook(_capture_hook)
    
    # Inject our probe directly into the HF generation pipeline
    suppressor = H3SinglePassSuppressor(
        captured_h, h3_results, h3_calibrated_results, 
        first_token_map, suppression_map, 
        top_k=top_k_coco, strength=suppression_strength
    )
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            logits_processor=[suppressor]
        )
        
    hook_handle.remove()
    
    prompt_len = inputs["input_ids"].shape[1]
    caption = processor.tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
    
    # Return dummy metadata to match old signature
    dummy_metadata = [{"steered": True}] * suppressor.n_steered
    return caption, dummy_metadata, suppressor.n_steered