import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from pycocotools.coco import COCO

# Add main directory to path to import CHAIR and other utilities
sys.path.append(".")
from chair import CHAIR

# ── Configuration ─────────────────────────────────────────────────────
MAX_TOKENS    = 256
DATA_DIR      = "coco"
IMAGE_DIR     = "coco/images/val2014"
MAX_IMAGES    = 500
BEST_LAYER    = 0
TOP_K         = 10
SAVE_PATH     = "joint_probe_data.pkl"

def get_image_positions(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns the indices of image token positions in the input sequence.
    LLaVA uses IMAGE_TOKEN_ID = 32000 as the placeholder for image tokens.
    """
    IMAGE_TOKEN_ID = 32000
    positions = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    return positions

def collect_joint_probe_data(
    model,
    processor,
    annotations: dict,
    evaluator,
    image_dir: str = "coco/images/val2014",
    max_images: int = 500,
    best_layer: int = 0,
    layers_for_ic: list = None,
    save_path: str = "joint_probe_data.pkl"
):
    """
    Collect joint features [h_vec, ic_max, ic_mean_topk, ic_spatial]
    for each object mention across all images.

    Returns:
        X_joint : list of (h_vec, ic_max, ic_mean_topk, ic_spatial) tuples
        y_joint : numpy array of labels (1=grounded, 0=hallucinated)
    """
    import os
    import pickle
    from collections import defaultdict

    if os.path.exists(save_path):
        print(f"Loading joint probe data from {save_path}...")
        with open(save_path, "rb") as f:
            X_joint, y_joint = pickle.load(f)
        return X_joint, y_joint

    lm_head, final_norm = get_llava_lm_head(model)

    base_to_syns = defaultdict(list)
    for syn, base in evaluator.inverse_synonym_dict.items():
        base_to_syns[base].append(syn)

    prompt    = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    n_layers  = len(model.model.language_model.layers)

    if layers_for_ic is None:
        layers_for_ic = list(range(0, n_layers, 4))

    X_joint    = []
    y_joint    = []
    skipped    = 0
    n_grounded = 0
    n_hall     = 0

    sample_anns = [
        a for a in list(annotations.values())[:max_images]
        if a.get("generated_caption")
    ]

    for ann in tqdm(sample_anns, desc="Collecting joint probe data"):
        try:
            image = Image.open(
                os.path.join(image_dir, ann["file_name"])
            ).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        image_positions = get_image_positions(inputs["input_ids"])

        if len(image_positions) == 0:
            continue

        img_start = image_positions[0].item()
        img_end   = image_positions[-1].item() + 1
        n_img     = img_end - img_start

        # ── Generate caption ──────────────────────────────────────────
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS, do_sample=False
            )

        prompt_len    = inputs["input_ids"].shape[1]
        generated_ids = generated[0][prompt_len:].tolist()
        generated_text = processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).lower()

        chair_result      = evaluator.compute_hallucinations(
            ann["image_id"], generated_text
        )
        this_grounded     = chair_result["recall_words"]
        this_hallucinated = chair_result["mscoco_hallucinated_words"]

        if not this_grounded and not this_hallucinated:
            continue

        # ── Prefill forward pass — reused for all objects in this image ──
        with torch.inference_mode():
            out_prefill = model.model.language_model(
                input_ids=inputs["input_ids"],
                output_hidden_states=True,
                return_dict=True,
            )

        # Precompute logit lens for all IC layers
        # ── DTYPE FIX: keep float16 through norm+lm_head, cast at softmax ──
        all_layer_probs = []
        for layer_idx in layers_for_ic:
            probs = _logit_lens_patch_probs(
                out_prefill.hidden_states[layer_idx + 1],
                img_start, img_end, final_norm, lm_head
            )
            all_layer_probs.append(probs)   # (n_img, vocab) float32 numpy

        # ── Per-object features ───────────────────────────────────────
        for obj, label in (
            [(w, 1) for w in this_grounded] +
            [(w, 0) for w in this_hallucinated]
        ):
            syns = sorted(
                base_to_syns.get(obj, [obj]), key=len, reverse=True
            )

            # Find token position for H3
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
                        end_pos = i
                        # Step backward to find the first token of the synonym
                        for j in range(end_pos, -1, -1):
                            chunk = processor.tokenizer.decode(
                                generated_ids[j:end_pos+1], skip_special_tokens=True
                            ).strip().lower()
                            if syn in chunk:
                                target_pos = j
                                break
                        break
                if target_pos is not None:
                    break

            if target_pos is None:
                skipped += 1
                continue

            # H3: hidden state at last token position before object word
            context_tensor = torch.tensor(
                [inputs["input_ids"][0].tolist() + generated_ids[:target_pos]],
                device=model.device
            )
            with torch.inference_mode():
                out_ctx = model.model.language_model(
                    input_ids=context_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
            h_vec = out_ctx.hidden_states[best_layer + 1][0, -1, :]\
                           .float().cpu().numpy()   # (4096,) float32

            # IC: internal confidence from logit lens
            object_token_ids = list({
                tid
                for word in syns + [obj]
                for variant in [word, " " + word]
                for tid in processor.tokenizer.encode(
                    variant, add_special_tokens=False
                )
            })

            ic_result = _compute_ic_features(
                all_layer_probs, object_token_ids, n_img, top_k=10
            )
            if ic_result is None:
                skipped += 1
                continue

            ic_max, ic_mean_topk, ic_spatial = ic_result

            X_joint.append((h_vec, ic_max, ic_mean_topk, ic_spatial))
            y_joint.append(label)

            if label == 1:
                n_grounded += 1
            else:
                n_hall += 1

    print(f"\n  Joint data collected : {len(X_joint)} samples")
    print(f"  Grounded             : {n_grounded}")
    print(f"  Hallucinated         : {n_hall}")
    print(f"  Skipped              : {skipped}")

    y_joint_arr = np.array(y_joint)
    with open(save_path, "wb") as f:
        pickle.dump((X_joint, y_joint_arr), f)

    return X_joint, y_joint_arr