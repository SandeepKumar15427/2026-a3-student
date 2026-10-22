"""Small, query-local feature augmentations for the LambdaMART ranker.

The original feature file has 136 LETOR features.  These features are derived
only from the candidates of the same query, so they do not use labels and do
not leak information between query groups.
"""
import numpy as np


RAW_FEATURES = 136
CHANNELS = (109, 107, 119, 124)  # zero-based: whole BM25, title BM25, LMIR-D, LMIR-JM


def _rank_percentile(values):
    n = len(values)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return 1.0 - ranks / max(1, n - 1)


def _query_features(X):
    """Return derived columns for one query's document rows."""
    n = X.shape[0]
    channel_values = X[:, CHANNELS]
    ranks = np.column_stack([_rank_percentile(channel_values[:, i]) for i in range(4)])
    all_ranks = np.column_stack([_rank_percentile(X[:, i]) for i in range(X.shape[1])])

    differences = np.column_stack([
        X[:, 109] - X[:, 107],
        X[:, 119] - X[:, 124],
        X[:, 109] - X[:, 119],
    ])

    # The five field-specific BM25 and LMIR columns are consecutive in the
    # assignment feature layout.  These summarize evidence across fields.
    aggregates = np.column_stack([
        X[:, 105:110].max(axis=1), X[:, 105:110].mean(axis=1),
        X[:, 115:120].max(axis=1), X[:, 115:120].mean(axis=1),
        X[:, 120:125].max(axis=1), X[:, 120:125].mean(axis=1),
    ])

    top10 = max(1, int(np.ceil(0.10 * n)))
    top20 = max(1, int(np.ceil(0.20 * n)))
    agreement = np.column_stack([
        (ranks >= np.partition(ranks, -top10, axis=0)[-top10, :]).sum(axis=1),
        (ranks >= np.partition(ranks, -top20, axis=0)[-top20, :]).sum(axis=1),
        ranks.mean(axis=1), ranks.max(axis=1), ranks.min(axis=1),
    ])
    return {
        "rank": ranks,
        "rank_all": all_ranks,
        "differences": differences,
        "aggregates": aggregates,
        "agreement": agreement,
    }


def augment_query_features(X, mode="derived"):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != RAW_FEATURES:
        raise ValueError(f"feature engineering expects shape (n, {RAW_FEATURES}), got {X.shape}")
    parts = _query_features(X)
    if mode == "none":
        return X
    if mode == "rank":
        names = ("rank",)
    elif mode == "rank_all":
        names = ("rank_all",)
    elif mode == "rank_all_agreement":
        names = ("rank_all", "agreement")
    elif mode in ("differences", "aggregates", "agreement"):
        names = (mode,)
    elif mode in ("derived", "all"):
        names = ("rank", "differences", "aggregates", "agreement")
    else:
        raise ValueError(f"unknown feature-engineering mode: {mode}")
    return np.column_stack([X] + [parts[name] for name in names])


def augment_dataset(X, qids, mode="derived"):
    """Apply query-local features to a grouped LETOR matrix."""
    if mode == "none":
        return np.asarray(X, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    qids = np.asarray(qids)
    if len(X) != len(qids):
        raise ValueError("X and qids must have the same number of rows")
    pieces = []
    start = 0
    for i in range(1, len(qids) + 1):
        if i == len(qids) or qids[i] != qids[start]:
            pieces.append(augment_query_features(X[start:i], mode))
            start = i
    return np.vstack(pieces) if pieces else X
