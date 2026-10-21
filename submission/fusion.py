"""
submission/fusion.py -- rank fusion over the named base-ranker scores that
fuse_candidates() receives (assignment Sections 3.1 and 4.1).

Input convention for every function here: `base_scores` maps each candidate
doc_id to {channel_name: score}, HIGHER = BETTER for every channel (LMIR
scores are log-probabilities, i.e. negative -- larger is still better).
Output: {doc_id: fused_score}, higher = better. Ties in a channel are broken
by the dict's insertion order (the candidate order), so results are
deterministic.

What you must write (Section 4.1): RRF (TODO(fusion-0)) and at least ONE
score-based method, CombSUM or CombMNZ (TODO(fusion-1)). Hand-written: no
fusion library. Staff test these functions against small hand-checkable examples
(docs/GRADING.md).
"""
from typing import Dict, Iterable, List, Optional


def _channels(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]]) -> List[str]:
    if channels is not None:
        return list(channels)
    first = next(iter(base_scores.values()), {})
    return list(first.keys())


def rrf_scores(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]] = None,
               c: float = 60.0) -> Dict[str, float]:
    """Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
    score(d) = sum over channels of 1 / (c + rank_channel(d)), ranks 1-based,
    best score = rank 1. Needs only each channel's ORDER, never its scale."""
    ids = list(base_scores)
    selected = _channels(base_scores, channels)
    fused = {doc_id: 0.0 for doc_id in ids}
    for channel in selected:
        # Python's sorted is stable, so equal scores retain candidate order.
        ranked = sorted(ids, key=lambda doc_id: base_scores[doc_id][channel], reverse=True)
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (c + rank)
    return fused


def minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise {doc: score} to [0, 1] within one channel. A channel
    where every document has the same score normalises to all zeros."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {d: 0.0 for d in values}
    return {d: (v - lo) / (hi - lo) for d, v in values.items()}


def combsum_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """CombSUM (Fox & Shaw, TREC-2 1994): sum of the per-channel MIN-MAX-
    normalised scores."""
    ids = list(base_scores)
    selected = _channels(base_scores, channels)
    combined = {doc_id: 0.0 for doc_id in ids}
    for channel in selected:
        channel_values = {doc_id: base_scores[doc_id][channel] for doc_id in ids}
        normalized = minmax_normalize(channel_values)
        for doc_id in ids:
            combined[doc_id] += normalized[doc_id]
    return combined


def combmnz_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """CombMNZ: CombSUM multiplied by the number of channels in which the
    document has a normalised score > 0. (Every candidate here has a score
    from every channel, so 'appears in the list' is defined as 'normalised
    score > 0' -- see docs/SUBMISSION_INTERFACE.md.)"""
    ids = list(base_scores)
    selected = _channels(base_scores, channels)
    combined = {doc_id: 0.0 for doc_id in ids}
    appearances = {doc_id: 0 for doc_id in ids}
    for channel in selected:
        channel_values = {doc_id: base_scores[doc_id][channel] for doc_id in ids}
        normalized = minmax_normalize(channel_values)
        for doc_id in ids:
            combined[doc_id] += normalized[doc_id]
            if normalized[doc_id] > 0.0:
                appearances[doc_id] += 1
    return {doc_id: combined[doc_id] * appearances[doc_id] for doc_id in ids}


def weighted_combmnz_scores(base_scores: Dict[str, Dict[str, float]],
                            weights: Dict[str, float],
                            channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Weighted CombMNZ using per-channel min-max normalization.

    ``weights`` changes each channel's contribution to the normalized sum;
    the CombMNZ multiplier still counts channels whose normalized score is
    positive. Missing weights default to zero, which makes the selected
    channel set explicit and deterministic.
    """
    ids = list(base_scores)
    selected = _channels(base_scores, channels)
    combined = {doc_id: 0.0 for doc_id in ids}
    appearances = {doc_id: 0 for doc_id in ids}
    for channel in selected:
        channel_values = {doc_id: base_scores[doc_id][channel] for doc_id in ids}
        normalized = minmax_normalize(channel_values)
        weight = float(weights.get(channel, 0.0))
        for doc_id in ids:
            combined[doc_id] += weight * normalized[doc_id]
            if normalized[doc_id] > 0.0:
                appearances[doc_id] += 1
    return {doc_id: combined[doc_id] * appearances[doc_id] for doc_id in ids}
