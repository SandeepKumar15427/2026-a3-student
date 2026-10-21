"""Run a targeted grouped-CV sweep for the Track B LambdaMART model.

This script trains no model when imported; run it on the stronger machine:

    python3 scripts/sweep_lambdamart.py \
        --train data/train \
        --folds 5 \
        --threads 4 \
        --log-out runs/lambdamart_sweep.txt

Each configuration is evaluated on the same query-grouped folds. Results are
printed immediately and appended to the log after every configuration, so an
interrupted sweep still leaves a usable report.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from harness.cv import grouped_kfold
from harness.feature_io import read_svmlight_ltr, stack
from scripts.lambdamart_utils import default_params, fit, ndcg10


def _resolve(path):
    return os.path.join(path, "train.txt") if os.path.isdir(path) else path


def _configurations():
    """Return a focused sweep covering capacity, regularization and learning rate."""
    configs = []

    # Tree-count sweep.
    for trees in (400, 600, 800):
        configs.append((trees, 15, 50, 0.05, 1.0, 0.8, 0.8))

    # Tree complexity sweep.
    for leaves in (15, 31, 63):
        configs.append((600, leaves, 50, 0.05, 1.0, 0.8, 0.8))

    # Leaf-size regularization sweep.
    for min_leaf in (50, 100, 200):
        configs.append((600, 31, min_leaf, 0.05, 1.0, 0.8, 0.8))

    # Learning-rate sweep with a moderately expressive model.
    for learning_rate in (0.02, 0.03, 0.05):
        configs.append((800, 31, 100, learning_rate, 1.0, 0.8, 0.8))

    # Regularization and feature/bagging fraction checks.
    for lambda_l2 in (1.0, 10.0):
        configs.append((600, 31, 100, 0.03, lambda_l2, 0.8, 0.8))
    configs.append((600, 31, 100, 0.03, 1.0, 1.0, 1.0))

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(configs))


def _format_config(config):
    trees, leaves, min_leaf, lr, l2, feature_fraction, bagging_fraction = config
    return (
        f"trees={trees} leaves={leaves} min_leaf={min_leaf} lr={lr:g} "
        f"lambda_l2={l2:g} feature_fraction={feature_fraction:g} "
        f"bagging_fraction={bagging_fraction:g}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train", help="train file or directory")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-out", default="runs/lambdamart_sweep.txt")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="optionally stop after this many configurations")
    args = parser.parse_args()

    path = _resolve(args.train)
    X, y, qids = stack(read_svmlight_ltr(path))
    folds = grouped_kfold(qids, k=args.folds, seed=args.seed)
    configs = _configurations()
    if args.max_configs is not None:
        configs = configs[:args.max_configs]

    os.makedirs(os.path.dirname(args.log_out) or ".", exist_ok=True)
    log = open(args.log_out, "w", encoding="utf-8")

    def report(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    report("LambdaMART grouped-CV sweep")
    report(f"data={path} rows={len(y)} features={X.shape[1]} folds={args.folds} threads={args.threads}")
    report("columns: index trees leaves min_leaf lr lambda_l2 feature_fraction bagging_fraction train val val/train seconds")

    results = []
    try:
        for index, config in enumerate(configs, 1):
            trees, leaves, min_leaf, lr, l2, feature_fraction, bagging_fraction = config
            description = _format_config(config)
            report(f"START {index}/{len(configs)} {description}")
            started = time.perf_counter()
            train_scores, val_scores = [], []

            params = default_params(
                num_leaves=leaves,
                learning_rate=lr,
                min_data_in_leaf=min_leaf,
                lambda_l2=l2,
                seed=args.seed,
                num_threads=args.threads,
                feature_fraction=feature_fraction,
                bagging_fraction=bagging_fraction,
                bagging_freq=1 if bagging_fraction < 1.0 else 0,
            )
            for train_idx, val_idx in folds:
                model = fit(X[train_idx], y[train_idx], qids[train_idx], params, trees)
                train_scores.append(ndcg10(model, X[train_idx], y[train_idx], qids[train_idx]))
                val_scores.append(ndcg10(model, X[val_idx], y[val_idx], qids[val_idx]))

            train_mean = float(np.mean(train_scores))
            val_mean = float(np.mean(val_scores))
            ratio = min(1.0, val_mean / train_mean) if train_mean > 0 else 0.0
            seconds = time.perf_counter() - started
            result = (val_mean, train_mean, ratio, config, seconds)
            results.append(result)
            report(
                f"DONE  {index}/{len(configs)} {description} "
                f"train={train_mean:.6f} val={val_mean:.6f} "
                f"val/train={ratio:.6f} seconds={seconds:.1f}"
            )

        report("")
        report("FINAL RANKING BY MEAN VALIDATION nDCG@10")
        for rank, (val_mean, train_mean, ratio, config, seconds) in enumerate(
                sorted(results, reverse=True), 1):
            report(
                f"{rank:2d}. val={val_mean:.6f} train={train_mean:.6f} "
                f"val/train={ratio:.6f} {_format_config(config)}"
            )
    finally:
        log.close()


if __name__ == "__main__":
    main()
