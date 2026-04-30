import pickle
import os
from chair import CHAIR
from chair import build_chair_annotations, generate_and_label_hallucinations
from load_model import model, processor
from load_data import coco, img_ids


PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

evaluator = CHAIR("coco/annotations/annotations")


if not os.path.exists("h3_results.pkl"):
    evaluator = CHAIR("coco/annotations/annotations")

    h3_results = run_h3(
        model, processor, annotations,
        evaluator=evaluator,        # pass evaluator here
        max_images=max_images,
    )
    with open("h3_results.pkl", "wb") as f:
        pickle.dump(h3_results, f)

    print(f"Saved. Size: {os.path.getsize('h3_results.pkl') / 1e6:.1f} MB")
else:
    with open("h3_results.pkl", "rb") as f:
      h3_results = pickle.load(f)

    print(f"best_layer : {h3_results['best_layer']}")
    print(f"best_auc   : {h3_results['best_auc']:.3f}")
    print(f"direction  : shape={h3_results['direction'].shape}, dtype={h3_results['direction'].dtype}")
    print(f"scaler mean: shape={h3_results['scaler'].mean_.shape}")