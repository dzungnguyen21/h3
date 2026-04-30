from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pickle
import os

def train_h3_only_calibrated(X_joint, y, h3_direction, h3_scaler, save_path="h3_calibrated_results.pkl"):
    """
    Train H3-only logistic regression with calibrated threshold.
    Uses the same X_joint data as train_joint_probe but only
    the H3 projection feature (1 dimension instead of 4).
    """
    if os.path.exists(save_path):
        print(f"\n  Loading H3-only calibrated results from {save_path}...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    # Extract H3 projection only — ignore IC features
    h3_projections = np.array([
        float(((h_vec - h3_scaler.mean_) / h3_scaler.scale_) @ h3_direction)
        for h_vec, ic_max, ic_mean_topk, ic_spatial in X_joint
    ], dtype=np.float32).reshape(-1, 1)   # (n, 1)

    print(f"\n  H3-only data:")
    print(f"  Samples          : {len(h3_projections)}")
    print(f"  Grounded         : {y.sum()}")
    print(f"  Hallucinated     : {len(y) - y.sum()}")
    print(f"  H3 proj mean     : {h3_projections.mean():.3f}")
    print(f"  H3 proj std      : {h3_projections.std():.3f}")

    # 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        h3_projections, y, test_size=0.2, stratify=y, random_state=42
    )

    feat_scaler = StandardScaler()
    X_train_s   = feat_scaler.fit_transform(X_train)
    X_val_s     = feat_scaler.transform(X_val)

    # Train logistic regression
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='saga')
    clf.fit(X_train_s, y_train)

    # Calibrate on val set
    calibrated = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    calibrated.fit(X_val_s, y_val)

    # Find optimal threshold on val set
    val_probs  = calibrated.predict_proba(X_val_s)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 80)
    f1_scores  = [
        f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_threshold = float(thresholds[np.argmax(f1_scores)])
    best_f1        = float(np.max(f1_scores))
    val_auc        = roc_auc_score(y_val, val_probs)

    print(f"\n  H3-only calibrated classifier:")
    print(f"    Val AUC                : {val_auc:.3f}")
    print(f"    Optimal prob threshold : {best_threshold:.3f}")
    print(f"    Val F1 at threshold    : {best_f1:.3f}")

    # Also find the equivalent projection-space threshold for reference
    # (where calibrated P(grounded) = best_threshold)
    all_probs = calibrated.predict_proba(X_val_s)[:, 1]
    all_projs = X_val[:, 0]
    sorted_idx = np.argsort(all_projs)
    sorted_projs = all_projs[sorted_idx]
    sorted_probs = all_probs[sorted_idx]
    crossing = np.argmin(np.abs(sorted_probs - best_threshold))
    equiv_proj_threshold = float(sorted_projs[crossing])
    print(f"    Equivalent proj threshold: {equiv_proj_threshold:.3f}  "
          f"(vs fixed -2.0 used previously)")

    results = {
        "clf"                   : calibrated,
        "feat_scaler"           : feat_scaler,
        "prob_threshold"        : best_threshold,
        "equiv_proj_threshold"  : equiv_proj_threshold,
        "val_auc"               : val_auc,
    }

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {save_path}")

    return results