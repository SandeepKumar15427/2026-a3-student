"""
submission/ranking.py -- your competition entry (assignment Sections 4.2, 5).

    prepare(model_path)
    fuse_candidates(qid, candidate_doc_ids, base_scores, k)   -> Track A
    rerank_with_ltr(qid, candidate_doc_ids, features, k)      -> Tracks B and C

The shipped baseline fuses the channels with your RRF (a BM25-channel stand-in
until rrf_scores is written) and reranks with a tiny pre-trained LambdaMART model (submission/model.txt, produced by
scripts/train_model.py). Keep the shape; replace the internals. The harness
never trains anything: prepare() only LOADS the artifact, and must raise if it
is missing (no silent fallback to training or to an untrained model).

Both ranking functions must return at most k (doc_id, score) pairs, best
first, using ONLY doc_ids from `candidate_doc_ids`, with no duplicates.
`base_scores`: {doc_id: {channel: score}} (the 4 named channels).
`features`:    {doc_id: [full feature vector]}.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from submission.fusion import weighted_combmnz_scores
from submission.feature_engineering import augment_query_features

_MODEL = None
_MODEL_FEATURES = None

# Selected from the fusion sweep on the released DEV split. The order is
# explicit by channel name rather than relying on feature indices here.
_FUSION_WEIGHTS = {
    "bm25_whole_document": 0.50,
    "bm25_title": 0.25,
    "lmir_dir_whole_document": 0.25,
    "lmir_jm_whole_document": 0.00,
}


def prepare(model_path: str) -> None:
    global _MODEL, _MODEL_FEATURES
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"trained model artifact not found at {model_path!r}; run scripts/train_model.py and "
            f"include the artifact in your submission (grading never retrains)"
        )
    import lightgbm as lgb
    _MODEL = lgb.Booster(model_file=model_path)
    _MODEL_FEATURES = _MODEL.num_feature()


def _unique(ids: List[str]) -> List[str]:
    return list(dict.fromkeys(ids))


def fuse_candidates(qid: str, candidate_doc_ids: List[str], base_scores: Dict[str, Dict[str, float]],
                    k: int) -> List[Tuple[str, float]]:
    ids = _unique(candidate_doc_ids)
    fused = weighted_combmnz_scores(
        {d: base_scores[d] for d in ids},
        _FUSION_WEIGHTS,
    )
    ranked = sorted(ids, key=lambda d: -fused[d])  # stable: ties keep candidate order
    return [(d, fused[d]) for d in ranked[:k]]


def rerank_with_ltr(qid: str, candidate_doc_ids: List[str], features: Dict[str, List[float]],
                    k: int) -> List[Tuple[str, float]]:
    if _MODEL is None:
        raise RuntimeError("call prepare(model_path) first")
    ids = _unique(candidate_doc_ids)
    if not ids:
        return []
    X = np.asarray([features[d] for d in ids], dtype=np.float64)
    if X.shape[1] != _MODEL_FEATURES:
        if X.shape[1] == 136 and _MODEL_FEATURES == 272:
            X = augment_query_features(X, "rank_all")
        else:
            raise ValueError(f"model expects {_MODEL_FEATURES} features but received {X.shape[1]}")
    preds = _MODEL.predict(X)
    order = np.argsort(-preds, kind="stable")
    return [(ids[i], float(preds[i])) for i in order[:k]]
