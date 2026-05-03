import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from chair import CHAIR
from PIL import Image
import os

MAX_TOKENS = 128
PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def extract_hidden_states_before_object(
    model,
    processor,
    image: Image.Image,
    object_words: list, # Now accepts a list of synonyms
    generated_ids: list,
    prompt: str = "USER: <image>\nDescribe this image in detail.\nASSISTANT:",
    layers_to_probe: list = None,
):
    """
    Run a forward pass and extract hidden states at the token position
    JUST BEFORE the object word is first completed in generation.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]

    # Find position where any of the object words first completes
    target_pos = None
    for i in range(len(generated_ids)):
        prefix = processor.tokenizer.decode(
            generated_ids[:i+1], skip_special_tokens=True
        ).lower()
        prev = processor.tokenizer.decode(
            generated_ids[:i], skip_special_tokens=True
        ).lower()

        for obj_word in object_words:
            if obj_word.lower() in prefix and obj_word.lower() not in prev:
                end_pos = i
                # Step backward to find the first token of the object word
                for j in range(end_pos, -1, -1):
                    chunk = processor.tokenizer.decode(
                        generated_ids[j:end_pos+1], skip_special_tokens=True
                    ).strip().lower()
                    if obj_word.lower() in chunk:
                        target_pos = j
                        break
                break
        if target_pos is not None:
            break

    if target_pos is None:
        return None

    # Build full input including generated tokens up to target position
    full_ids = (
        inputs["input_ids"][0].tolist()
        + generated_ids[:target_pos]
    )
    full_ids_tensor = torch.tensor([full_ids], device=model.device)

    n_layers = len(model.model.language_model.layers)
    if layers_to_probe is None:
        layers_to_probe = list(range(0, n_layers // 2, 4)) + \
                          list(range(n_layers // 2, int(n_layers * 0.75), 2))

    with torch.inference_mode():
        outputs = model.model.language_model(
            input_ids=full_ids_tensor,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_by_layer = {}
    for layer_idx in layers_to_probe:
        h = outputs.hidden_states[layer_idx + 1]
        vec = h[0, -1, :].float().cpu().numpy()
        hidden_by_layer[layer_idx] = vec

    return hidden_by_layer, target_pos


def collect_existence_probe_data(
    model,
    processor,
    annotations: dict,
    evaluator: CHAIR,
    image_dir: str = "coco/images/val2014",
    max_images: int = 500,
    layers_to_probe: list = None,
    cache_path: str = "existence_probe_data.pkl"
):
    import os
    import pickle
    
    if os.path.exists(cache_path):
        print(f"\n  [CACHE] Loading cached probe data from {cache_path}...")
        with open(cache_path, "rb") as f:
            data_by_layer, layers_to_probe = pickle.load(f)
        return data_by_layer, layers_to_probe

    n_layers = len(model.model.language_model.layers)
    if layers_to_probe is None:
        layers_to_probe = list(range(0, n_layers // 2, 4)) + \
                          list(range(n_layers // 2, int(n_layers * 0.75), 2))

    data_by_layer = {l: {"X": [], "y": []} for l in layers_to_probe}
    skipped = 0
    n_grounded_total = 0
    n_hall_total = 0

    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

    # Build a reverse mapping from base COCO category to all its synonyms
    base_to_synonyms = defaultdict(list)
    for syn, base in evaluator.inverse_synonym_dict.items():
        base_to_synonyms[base].append(syn)

    with tqdm(list(annotations.values())[:max_images], desc="Collecting H3 data") as pbar:
        for ann in pbar:
            if not ann.get("generated_caption"):
                continue

            grounded_words     = ann.get("grounded_words", [])
            hallucinated_words = ann.get("hallucinated_words", [])

            if not grounded_words and not hallucinated_words:
                continue

            file_name = ann["file_name"]
            try:
                image = Image.open(
                    os.path.join(image_dir, file_name)
                ).convert("RGB")
            except Exception:
                continue

            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=MAX_TOKENS, do_sample=False
                )

            prompt_len    = inputs["input_ids"].shape[1]
            generated_ids = generated[0][prompt_len:].tolist()
            generated_text = processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).lower()

            chair_result = evaluator.compute_hallucinations(
                ann["image_id"], generated_text
            )
            this_grounded     = chair_result["recall_words"]
            this_hallucinated = chair_result["mscoco_hallucinated_words"]

            for obj, label in (
                [(w, 1) for w in this_grounded] +
                [(w, 0) for w in this_hallucinated]
            ):
                # Get all synonyms for this object to check in the text
                valid_synonyms = base_to_synonyms.get(obj, [obj])
                # Sort by length so longer synonyms ("hot dog") match before shorter ones ("dog")
                valid_synonyms = sorted(valid_synonyms, key=len, reverse=True)

                result = extract_hidden_states_before_object(
                    model, processor, image, valid_synonyms,
                    generated_ids, prompt, layers_to_probe
                )

                if result is None:
                    skipped += 1
                    continue

                hidden_by_layer, target_pos = result

                for layer_idx, vec in hidden_by_layer.items():
                    data_by_layer[layer_idx]["X"].append(vec)
                    data_by_layer[layer_idx]["y"].append(label)

                if label == 1:
                    n_grounded_total += 1
                else:
                    n_hall_total += 1

            pbar.set_postfix(
                grounded=n_grounded_total,
                hallucinated=n_hall_total,
                skipped=skipped
            )

    
    import pickle
    print(f"\n  [CACHE] Saving probe data to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump((data_by_layer, layers_to_probe), f)

    return data_by_layer, layers_to_probe

def run_existence_probe(data_by_layer: dict, layers_to_probe: list):
    results = {}

    print("\n  Probing layers for existence direction:")
    print("  " + "-" * 55)
    print(f"  {'Layer':<8} {'n_samples':<12} {'AUC (CV)':<12} {'Interpretation'}")
    print("  " + "-" * 55)

    for layer_idx in layers_to_probe:
        X = np.array(data_by_layer[layer_idx]["X"])
        y = np.array(data_by_layer[layer_idx]["y"])

        if len(X) < 10 or len(np.unique(y)) < 2:
            print(f"  {layer_idx:<8} {'insufficient data'}")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for train_idx, val_idx in cv.split(X_scaled, y):
            clf = LogisticRegression(max_iter=5000, C=1.0)
            clf.fit(X_scaled[train_idx], y[train_idx])
            probs = clf.predict_proba(X_scaled[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], probs))

        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)

        if mean_auc > 0.70:
            interp = "STRONG direction"
        elif mean_auc > 0.60:
            interp = "moderate direction"
        elif mean_auc > 0.55:
            interp = "weak direction"
        else:
            interp = "no direction"

        print(f"  {layer_idx:<8} {len(X):<12} {mean_auc:.3f}±{std_auc:.3f}  {interp}")

        results[layer_idx] = {
            "auc": mean_auc,
            "auc_std": std_auc,
            "n_samples": len(X),
            "n_grounded": int(y.sum()),
            "n_hallucinated": int((1 - y).sum()),
        }

    return results


def extract_existence_direction(data_by_layer: dict, best_layer: int):
    X = np.array(data_by_layer[best_layer]["X"])
    y = np.array(data_by_layer[best_layer]["y"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(X_scaled, y)

    direction = clf.coef_[0]
    projections = X_scaled @ direction
    proj_grounded     = projections[y == 1]
    proj_hallucinated = projections[y == 0]

    return direction, scaler, clf, proj_grounded, proj_hallucinated

def plot_h3_results(probe_results: dict, proj_grounded, proj_hallucinated, best_layer: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    layers = sorted(probe_results.keys())
    aucs   = [probe_results[l]["auc"]     for l in layers]
    stds   = [probe_results[l]["auc_std"] for l in layers]

    ax.errorbar(layers, aucs, yerr=stds,
                fmt="o-", color="#534AB7", linewidth=1.5,
                markersize=5, capsize=3)
    ax.axhline(0.5,  color="#888780", linewidth=0.8,
               linestyle="--", label="chance (0.5)")
    ax.axhline(0.7,  color="#1D9E75", linewidth=0.8,
               linestyle="--", label="strong (0.7)")
    ax.axvline(best_layer, color="#D85A30", linewidth=1,
               linestyle=":", label=f"best layer ({best_layer})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Existence direction strength per layer")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)

    ax2 = axes[1]
    ax2.hist(proj_grounded,     bins=30, alpha=0.6,
             color="#1D9E75", label=f"grounded (n={len(proj_grounded)})",
             density=True)
    ax2.hist(proj_hallucinated, bins=30, alpha=0.6,
             color="#D85A30", label=f"hallucinated (n={len(proj_hallucinated)})",
             density=True)
    ax2.axvline(0, color="#888780", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Projection onto existence direction")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Hidden state separation at layer {best_layer}")
    ax2.legend(fontsize=8)

    plt.suptitle(
        "H3: Does a linear existence direction appear in early layers?",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("exp_h3_existence_direction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: exp_h3_existence_direction.png")


def run_h3(model, processor, annotations, evaluator, image_dir="coco/images/val2014", max_images=300):
    print("=" * 60)
    print("EXPERIMENT H3: Existence Direction Probe")
    print("=" * 60)

    # Step 1: collect hidden states
    data_by_layer, layers_to_probe = collect_existence_probe_data(
        model, processor, annotations,
        evaluator=evaluator,        # pass evaluator down
        image_dir=image_dir,
        max_images=max_images,
    )

    total_samples = len(data_by_layer[layers_to_probe[0]]["X"])
    print(f"\n  Total samples collected: {total_samples}")
    if total_samples < 20:
        print("  Too few samples — increase max_images")
        return None

    # Step 2: probe each layer
    probe_results = run_existence_probe(data_by_layer, layers_to_probe)

    if not probe_results:
        print("  No valid probe results")
        return None

    # Step 3: find best layer
    best_layer = max(probe_results, key=lambda l: probe_results[l]["auc"])
    best_auc   = probe_results[best_layer]["auc"]
    print(f"\n  Best layer: {best_layer}  AUC={best_auc:.3f}")

    if best_auc > 0.60:
        print("  SUPPORTS H3 — existence direction found")
    else:
        print("  NOT significant — no clear existence direction")

    # Step 4: extract direction and plot
    direction, scaler, clf, proj_g, proj_h = extract_existence_direction(
        data_by_layer, best_layer
    )

    plot_h3_results(probe_results, proj_g, proj_h, best_layer)

    h3 = {
        "probe_results"  : probe_results,
        "best_layer"     : best_layer,
        "best_auc"       : best_auc,
        "direction"      : direction,
        "scaler"         : scaler,
        "clf"            : clf,
        "data_by_layer"  : data_by_layer,
        "layers_to_probe": layers_to_probe,
    }

    return h3
