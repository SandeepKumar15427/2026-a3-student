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
    # TODO(fusion-0): implement RRF. For each channel, sort the documents by that channel's score, best first
    # (use a STABLE sort so ties keep candidate order), give the best document rank 1, and add 1 / (c + rank)
    # to each document's fused score. Worked example (c = 60):
    #   base = {"a": {"x": 3.0, "y": 1.0}, "b": {"x": 2.0, "y": 3.0}, "c": {"x": 1.0, "y": 2.0}}
    #   a: 1/61 + 1/63   b: 1/62 + 1/61   c: 1/63 + 1/62
    # You may also weight channels or tune c on TRAIN (Section 4.1). Until you do, submission/ranking.py
    # falls back to sorting by the BM25 channel so the pipeline still runs.
    raise NotImplementedError("TODO(fusion-0): implement rrf_scores")


def minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise {doc: score} to [0, 1] within one channel. A channel
    where every document has the same score normalises to all zeros."""
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {d: 0.0 for d in values}
    return {d: (v - lo) / (hi - lo) for d, v in values.items()}


def combsum_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """CombSUM (Fox & Shaw, TREC-2 1994): sum of the per-channel MIN-MAX-
    normalised scores."""
    # TODO(fusion-1): implement CombSUM (or CombMNZ below; at least one of the
    # two is required). Normalise each channel with minmax_normalize() first,
    # then sum across channels. Example (2 channels, 3 docs):
    #   base = {"a": {"x": 1.0, "y": 10.0}, "b": {"x": 3.0, "y": 20.0}, "c": {"x": 2.0, "y": 30.0}}
    #   normalised x: a=0, b=1, c=0.5 ; y: a=0, b=0.5, c=1  ->  a=0.0, b=1.5, c=1.5
    raise NotImplementedError("TODO(fusion-1): implement combsum_scores (or combmnz_scores)")


def combmnz_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """CombMNZ: CombSUM multiplied by the number of channels in which the
    document has a normalised score > 0. (Every candidate here has a score
    from every channel, so 'appears in the list' is defined as 'normalised
    score > 0' -- see docs/SUBMISSION_INTERFACE.md.)"""
    # TODO(fusion-1, alternative): optional if you implemented combsum_scores.
    raise NotImplementedError("TODO(fusion-1): implement combmnz_scores (or combsum_scores)")
