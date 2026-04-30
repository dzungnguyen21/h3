import json
import os
from chair import CHAIR
from chair import build_chair_annotations, generate_and_label_hallucinations
from load_model import model, processor
from load_data import coco, img_ids

if not os.path.exists("chair_annotations.json"):
    # Initialize the evaluator
    print("Initializing CHAIR Evaluator...")
    evaluator = CHAIR("coco/annotations/annotations")

    # Force generation for a small subset to ensure we see results
    subset_img_ids = img_ids[:500]
    annotations = build_chair_annotations(coco, subset_img_ids, evaluator)

    annotations = generate_and_label_hallucinations(
        model, processor, annotations, evaluator
    )
else:
    with open("chair_annotations.json", "r") as f:
        annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations")