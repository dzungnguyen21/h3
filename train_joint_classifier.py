import numpy as np

def train_joint_probe(X_joint, y, h3_direction, h3_scaler):
    """
    Train a 4-feature joint classifier:
      [H3_projection, ic_max, ic_mean_topk, ic_spatial_consistency]

    Compares H3-only, IC-only, and joint AUC via 5-fold CV.
    Returns calibrated classifier and optimal probability threshold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.calibration import CalibratedClassifierCV

    # Build 4-dim feature matrix
    features = []
    for h_vec, ic_max, ic_mean_topk, ic_spatial in X_joint:
        h_std   = (h_vec - h3_scaler.mean_) / h3_scaler.scale_
        h3_proj = float(h_std @ h3_direction)
        features.append([h3_proj, ic_max, ic_mean_topk, ic_spatial])

    X_feat = np.array(features, dtype=np.float32)

    print(f"\n  Feature matrix shape : {X_feat.shape}")
    print(f"  Feature names        : [H3_proj, ic_max, ic_mean_topk, ic_spatial]")
    print(f"  Feature means        : {X_feat.mean(axis=0).round(4)}")
    print(f"  Feature stds         : {X_feat.std(axis=0).round(4)}")
    print(f"  Class balance        : grounded={y.sum()}, "
          f"hallucinated={len(y)-y.sum()}")

    # 5-fold AUC comparison
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n  AUC comparison (5-fold CV):")
    print("  " + "-" * 50)

    for feat_name, feat_cols in [
        ("H3 only                  ", [0]),
        ("IC only (max+topk+spatial)", [1, 2, 3]),
        ("H3 + IC (joint)          ", [0, 1, 2, 3]),
    ]:
        X_sub = X_feat[:, feat_cols]
        aucs  = []
        for train_idx, val_idx in cv.split(X_sub, y):
            sc    = StandardScaler()
            X_tr  = sc.fit_transform(X_sub[train_idx])
            X_va  = sc.transform(X_sub[val_idx])
            clf_c = LogisticRegression(
                max_iter=2000, C=1.0, solver='saga'
            )
            clf_c.fit(X_tr, y[train_idx])
            probs = clf_c.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y[val_idx], probs))
        print(f"  {feat_name}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # Train final classifier on 80% / calibrate on 20%
    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=42
    )

    feat_scaler = StandardScaler()
    X_train_s   = feat_scaler.fit_transform(X_train)
    X_val_s     = feat_scaler.transform(X_val)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver='saga')
    clf.fit(X_train_s, y_train)

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

    print(f"\n  Final joint classifier:")
    print(f"    Val AUC                : {val_auc:.3f}")
    print(f"    Optimal prob threshold : {best_threshold:.3f}")
    print(f"    Val F1 at threshold    : {best_f1:.3f}")
    print(f"    Feature weights        : {clf.coef_[0].round(4)}")
    print(f"      [H3_proj, ic_max, ic_mean_topk, ic_spatial]")

    return {
        "clf"           : calibrated,
        "feat_scaler"   : feat_scaler,
        "prob_threshold": best_threshold,
        "val_auc"       : val_auc,
    }