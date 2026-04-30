import os
import json
import torch
import pickle
from tqdm import tqdm
from PIL import Image

from load_model import model, processor
from load_data import coco, img_ids
from chair import CHAIR, compute_chair_score
from intervene_h3_calibrated import intervene_h3_calibrated
from intervene_h3_online import intervene_h3_online
from config import MAX_IMAGES

def main():
    print("Initializing CHAIR Evaluator...")
    evaluator = CHAIR("coco/annotations/annotations")

    # 1. Load baseline annotations
    if not os.path.exists("chair_annotations.json"):
        print("Generating annotations (Pass 1)...")
        # make_annotation handles generating the baseline annotations
        import make_annotation
        annotations = make_annotation.annotations
    else:
        print("Loading existing annotations...")
        with open("chair_annotations.json", "r") as f:
            annotations = json.load(f)

    # 2. Load H3 results
    if not os.path.exists("h3_results.pkl"):
        print("Generating H3 results...")
        import make_h3
        h3_results = make_h3.h3_results
    else:
        print("Loading existing H3 results...")
        with open("h3_results.pkl", "rb") as f:
            h3_results = pickle.load(f)

    # 3. Load Calibrated results
    if not os.path.exists("h3_calibrated_results.pkl"):
        print("Generating H3 calibrated results...")
        from collect_joint_data import collect_joint_probe_data
        from train_h3_calibrate import train_h3_only_calibrated
        
        X_joint, y_joint = collect_joint_probe_data(
            model, processor, annotations, evaluator, max_images=MAX_IMAGES
        )
        h3_calibrated_results = train_h3_only_calibrated(
            X_joint, y_joint, h3_results["direction"], h3_results["scaler"]
        )
    else:
        print("Loading existing H3 calibrated results...")
        with open("h3_calibrated_results.pkl", "rb") as f:
            h3_calibrated_results = pickle.load(f)

    print("\nAll required data loaded. Starting benchmark comparison...")
    
    pass1_anns = {}
    pass2_anns = {}
    pass3_anns = {}

    sample_keys = list(annotations.keys())[:MAX_IMAGES]

    for key in tqdm(sample_keys, desc="Benchmarking Passes"):
        ann = annotations[key]
        img_id = ann["image_id"]
        image_path = os.path.join("coco/images/val2014", ann["file_name"])
        
        if not os.path.exists(image_path):
            continue
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Pass 1 and 2 from intervene_h3_calibrated
        pass1_text, pass2_text, _, _ = intervene_h3_calibrated(
            model, processor, image, h3_results, h3_calibrated_results, evaluator
        )

        # Pass 3 from intervene_h3_online
        pass3_text, step_meta, n_steered = intervene_h3_online(
            model, processor, image, h3_results, h3_calibrated_results, evaluator
        )

        # Evaluate Pass 1
        res1 = evaluator.compute_hallucinations(img_id, pass1_text.lower())
        pass1_anns[key] = {
            "generated_caption": pass1_text,
            "objects_in_image": ann["objects_in_image"],
            "hallucinated_words": list(set(res1['mscoco_hallucinated_words'])),
            "grounded_words": list(set(res1['recall_words']))
        }

        # Evaluate Pass 2
        res2 = evaluator.compute_hallucinations(img_id, pass2_text.lower())
        pass2_anns[key] = {
            "generated_caption": pass2_text,
            "objects_in_image": ann["objects_in_image"],
            "hallucinated_words": list(set(res2['mscoco_hallucinated_words'])),
            "grounded_words": list(set(res2['recall_words']))
        }

        # Evaluate Pass 3
        res3 = evaluator.compute_hallucinations(img_id, pass3_text.lower())
        pass3_anns[key] = {
            "generated_caption": pass3_text,
            "objects_in_image": ann["objects_in_image"],
            "hallucinated_words": list(set(res3['mscoco_hallucinated_words'])),
            "grounded_words": list(set(res3['recall_words']))
        }

    print("\n\n" + "="*75)
    print(" BASELINE (Original from annotations)")
    print("="*75)
    compute_chair_score(annotations)

    print("\n\n" + "="*75)
    print(" PASS 1 (No Intervention)")
    print("="*75)
    compute_chair_score(pass1_anns)

    print("\n\n" + "="*75)
    print(" PASS 2 (Output-Level Suppression)")
    print("="*75)
    compute_chair_score(pass2_anns)

    print("\n\n" + "="*75)
    print(" PASS 3 (Context-Level Activation Steering)")
    print("="*75)
    compute_chair_score(pass3_anns)

if __name__ == "__main__":
    main()
