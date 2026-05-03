import os
import glob
import pickle
import json
import torch
from PIL import Image
from tqdm.auto import tqdm

# Import project modules
from load_model import model, processor
from chair import CHAIR
from intervene_h3_online import intervene_h3_online

def print_summary(name, n_images, n_hall_imgs, n_hall_words, n_recall_words, n_gt_words):
    chair_s = (n_hall_imgs / n_images) * 100 if n_images > 0 else 0
    total_generated = n_hall_words + n_recall_words
    chair_i = (n_hall_words / total_generated) * 100 if total_generated > 0 else 0
    coverage = (n_recall_words / n_gt_words) * 100 if n_gt_words > 0 else 0
    precision = (n_recall_words / total_generated) * 100 if total_generated > 0 else 0
    
    if precision + coverage > 0:
        f1 = 2 * (precision * coverage) / (precision + coverage)
    else:
        f1 = 0

    print(f"\\n--- Evaluating {name} Captions ---")
    print("CHAIR + Coverage Summary:")
    print(f"  Images processed         : {n_images}")
    print(f"  Images with hallucination: {n_hall_imgs} ({chair_s:.1f}%)")
    print(f"  Total hallucinated words : {n_hall_words}")
    print(f"  Total grounded words     : {n_recall_words}")
    print(f"  CHAIR_S                  : {chair_s:.1f}%")
    print(f"  CHAIR_I                  : {chair_i:.1f}%")
    print(f"  Coverage (Recall)        : {coverage:.1f}%")
    print(f"  Precision                : {precision:.1f}%")
    print(f"  F1                       : {f1:.1f}%")


def run_online_benchmark(num_images=500, steering_strength=3.0, top_k_coco=1):
    print('Loading CHAIR Evaluator...')
    evaluator = CHAIR("coco/annotations/annotations")

    print('Loading H3 Probes...')
    try:
        with open('h3_results.pkl', 'rb') as f:
            h3_results = pickle.load(f)
        with open('h3_calibrated_results.pkl', 'rb') as f:
            h3_calibrated_results = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: Calibration files not found! Run make_h3.py and calibration first.")
        return

    # Load baseline annotations
    ann_file = "chair_annotations.json"
    if not os.path.exists(ann_file):
        print(f"ERROR: Baseline file {ann_file} not found.")
        return
        
    print(f"Loading baseline from {ann_file}...")
    with open(ann_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
        
    # Get values and limit to num_images
    if isinstance(annotations, dict):
        ann_list = list(annotations.values())
    else:
        ann_list = list(annotations)
        
    sample_anns = [a for a in ann_list if a.get("generated_caption")][:num_images]

    base_hall_imgs = base_hall_words = base_recall_words = base_gt_words = 0
    steer_hall_imgs = steer_hall_words = steer_recall_words = steer_gt_words = 0
    skipped = 0

    print(f"\\nStarting Benchmark on {len(sample_anns)} images (Strength={steering_strength}, Top-K={top_k_coco})...")
    
    for ann in tqdm(sample_anns):
        img_id = ann["image_id"]
        img_path = os.path.join("coco/images/val2014", ann["file_name"])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            skipped += 1
            continue

        # --- 1. Evaluate Baseline (From JSON) ---
        baseline_caption = ann["generated_caption"]
        b_res = evaluator.compute_hallucinations(img_id, baseline_caption)
        
        b_gt = len(b_res.get("mscoco_gt_words", []))
        b_hall = len(b_res.get("mscoco_hallucinated_words", []))
        b_recall = len(b_res.get("recall_words", []))

        base_gt_words += b_gt
        base_hall_words += b_hall
        base_recall_words += b_recall
        if b_hall > 0:
            base_hall_imgs += 1

        # --- 2. Steered Generation (Online Activation Steering) ---
        steered_caption, _, _ = intervene_h3_online(
            model=model, 
            processor=processor, 
            image=image,
            h3_results=h3_results, 
            h3_calibrated_results=h3_calibrated_results, 
            evaluator=evaluator,
            steering_strength=steering_strength,
            top_k_coco=top_k_coco
        )

        # Evaluate Steered
        s_res = evaluator.compute_hallucinations(img_id, steered_caption)
        s_gt = len(s_res.get("mscoco_gt_words", []))
        s_hall = len(s_res.get("mscoco_hallucinated_words", []))
        s_recall = len(s_res.get("recall_words", []))

        steer_gt_words += s_gt
        steer_hall_words += s_hall
        steer_recall_words += s_recall
        if s_hall > 0:
            steer_hall_imgs += 1

    # --- Print Summaries ---
    actual_processed = len(sample_anns) - skipped
    print_summary("Baseline", actual_processed, base_hall_imgs, base_hall_words, base_recall_words, base_gt_words)
    print_summary("Online Steered", actual_processed, steer_hall_imgs, steer_hall_words, steer_recall_words, steer_gt_words)

if __name__ == "__main__":
    # Test on all 500 images
    run_online_benchmark(num_images=500, steering_strength=3.0, top_k_coco=1)