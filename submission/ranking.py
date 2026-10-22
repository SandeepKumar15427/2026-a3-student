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
_ENSEMBLE_MODELS = []
_ENSEMBLE_WEIGHTS = []

# Fixed channel weights; the order is explicit by channel name rather than
# relying on feature indices here.
_FUSION_WEIGHTS = {
    "bm25_whole_document": 0.50,
    "bm25_title": 0.25,
    "lmir_dir_whole_document": 0.25,
    "lmir_jm_whole_document": 0.00,
}


def prepare(model_path: str) -> None:
    global _MODEL, _MODEL_FEATURES, _ENSEMBLE_MODELS, _ENSEMBLE_WEIGHTS
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"trained model artifact not found at {model_path!r}; run scripts/train_model.py and "
            f"include the artifact in your submission (grading never retrains)"
        )
    import lightgbm as lgb
    _MODEL = lgb.Booster(model_file=model_path)
    _MODEL_FEATURES = _MODEL.num_feature()

    # Optional approved ensemble.  Order: primary 277-feature model,
    # previous 272-feature model, raw 136-feature model.  The environment
    # variable is useful for local weight experiments without code edits.
    # Selected with grouped out-of-fold TRAIN predictions, not by repeatedly
    # tuning on DEV: 277-feature / 272-feature / raw.
    raw_weights = os.environ.get("LTR_ENSEMBLE_WEIGHTS", "0.45,0.25,0.30")
    try:
        weights = np.asarray([float(x) for x in raw_weights.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("LTR_ENSEMBLE_WEIGHTS must be three comma-separated numbers") from exc
    if weights.shape != (3,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("LTR_ENSEMBLE_WEIGHTS must contain three nonnegative values with positive sum")
    _ENSEMBLE_WEIGHTS = weights / weights.sum()
    model_paths = [
        model_path,
        os.path.join(os.path.dirname(model_path), "model_rank_all.txt"),
        os.path.join(os.path.dirname(model_path), "model_raw.txt"),
    ]
    _ENSEMBLE_MODELS = [_MODEL]
    missing_companions = [
        model_paths[index]
        for index in (1, 2)
        if _ENSEMBLE_WEIGHTS[index] > 0 and not os.path.isfile(model_paths[index])
    ]
    if missing_companions:
        # C1 runs recipe.py in an isolated directory containing only the
        # generated primary model.  Use that model by itself there; normal
        # MSLR submissions still use the ensemble when all artifacts exist.
        _ENSEMBLE_WEIGHTS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        _ENSEMBLE_MODELS.extend([None, None])
    else:
        for index in (1, 2):
            if _ENSEMBLE_WEIGHTS[index] == 0:
                _ENSEMBLE_MODELS.append(None)
            else:
                _ENSEMBLE_MODELS.append(lgb.Booster(model_file=model_paths[index]))


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
    def model_input(model):
        expected = model.num_feature()
        if X.shape[1] == expected:
            return X
        if X.shape[1] == 136 and expected == 272:
            return augment_query_features(X, "rank_all")
        if X.shape[1] == 136 and expected == 277:
            return augment_query_features(X, "rank_all_agreement")
        raise ValueError(f"model expects {expected} features but received {X.shape[1]}")

    predictions = []
    for model, weight in zip(_ENSEMBLE_MODELS, _ENSEMBLE_WEIGHTS):
        if weight == 0:
            predictions.append(None)
            continue
        score = np.asarray(model.predict(model_input(model)), dtype=np.float64)
        # Normalize per query because different feature spaces produce
        # incomparable LightGBM score scales.
        score = (score - score.mean()) / max(float(score.std()), 1e-12)
        predictions.append(score)
    preds = sum(weight * score for weight, score in zip(_ENSEMBLE_WEIGHTS, predictions) if score is not None)
    order = np.argsort(-preds, kind="stable")
    return [(ids[i], float(preds[i])) for i in order[:k]]
