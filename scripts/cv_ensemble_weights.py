"""Tune LambdaMART ensemble weights with query-grouped out-of-fold predictions.

The three members are, in order: 277-feature raw+rank+agreement, 272-feature
raw+rank, and 136-feature raw.  Every model is trained without its validation
fold, so the weight search does not evaluate models on queries they saw.

Example:
    python3 scripts/cv_ensemble_weights.py \
        --train data/train --folds 5 --threads 4 \
        --trees 600 --num-leaves 31 --learning-rate 0.03 \
        --min-data-in-leaf 200 --lambda-l2 10 \
        --feature-fraction 0.6 --bagging-fraction 0.7
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from harness.cv import grouped_kfold
from harness.feature_io import read_svmlight_ltr, stack
from harness.metrics import mean_ndcg_from_scores
from scripts.lambdamart_utils import default_params, fit
from submission.feature_engineering import augment_dataset


def _weights_near_current():
    # Order: 277-feature / 272-feature / raw.  All are near the current
    # 0.50 / 0.30 / 0.20 leaderboard setting.
    return [
        (0.40, 0.30, 0.30),
        (0.45, 0.25, 0.30),
        (0.45, 0.30, 0.25),
        (0.45, 0.35, 0.20),
        (0.50, 0.25, 0.25),
        (0.50, 0.30, 0.20),
        (0.50, 0.35, 0.15),
        (0.55, 0.20, 0.25),
        (0.55, 0.25, 0.20),
        (0.55, 0.30, 0.15),
        (0.60, 0.20, 0.20),
        (0.60, 0.25, 0.15),
    ]


def _query_zscore(values, qids):
    values = np.asarray(values, dtype=np.float64).copy()
    qids = np.asarray(qids)
    start = 0
    for i in range(1, len(qids) + 1):
        if i == len(qids) or qids[i] != qids[start]:
            block = values[start:i]
            block -= block.mean()
            std = block.std()
            if std > 1e-12:
                block /= std
            values[start:i] = block
            start = i
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--min-data-in-leaf", type=int, default=200)
    ap.add_argument("--lambda-l2", type=float, default=10.0)
    ap.add_argument("--feature-fraction", type=float, default=0.6)
    ap.add_argument("--bagging-fraction", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = os.path.join(args.train, "train.txt") if os.path.isdir(args.train) else args.train
    X_raw, y, qids = stack(read_svmlight_ltr(path))
    X_rank = augment_dataset(X_raw, qids, "rank_all").astype(np.float32, copy=False)
    X_agreement = augment_dataset(X_raw, qids, "rank_all_agreement").astype(np.float32, copy=False)
    X_raw = np.asarray(X_raw, dtype=np.float32)

    folds = grouped_kfold(qids, k=args.folds, seed=args.seed)
    params = default_params(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        min_data_in_leaf=args.min_data_in_leaf,
        lambda_l2=args.lambda_l2,
        seed=args.seed,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=1 if args.bagging_fraction < 1.0 else 0,
        num_threads=args.threads,
    )
    matrices = (X_agreement, X_rank, X_raw)
    print(f"data={path} rows={len(y)} folds={args.folds} threads={args.threads}")
    print("models=277-feature-agreement,272-feature-rank,136-feature-raw")
    print("weights are ordered 277 / 272 / raw")

    fold_predictions = []
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"training fold {fold_no}/{len(folds)}", flush=True)
        val_preds = []
        for X in matrices:
            model = fit(X[train_idx], y[train_idx], qids[train_idx], params, args.trees)
            val_preds.append(_query_zscore(model.predict(X[val_idx]), qids[val_idx]))
        fold_predictions.append((val_idx, val_preds))

    results = []
    for weights in _weights_near_current():
        fold_scores = []
        for val_idx, predictions in fold_predictions:
            combined = sum(weight * prediction for weight, prediction in zip(weights, predictions))
            fold_scores.append(mean_ndcg_from_scores(combined, y[val_idx], qids[val_idx], k=10))
        results.append((float(np.mean(fold_scores)), float(np.std(fold_scores)), weights, fold_scores))

    print("\nweights                         mean       std   fold scores")
    for mean, std, weights, fold_scores in sorted(results, reverse=True):
        print(f"{weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}                  "
              f"{mean:.6f}  {std:.6f}  "
              f"{' '.join(f'{s:.4f}' for s in fold_scores)}")
    best = max(results, key=lambda item: item[0])
    print(f"\nBEST weights 277/272/raw = {best[2][0]:.2f}/{best[2][1]:.2f}/{best[2][2]:.2f}")
    print(f"BEST mean nDCG@10 = {best[0]:.6f} +/- {best[1]:.6f}")


if __name__ == "__main__":
    main()
