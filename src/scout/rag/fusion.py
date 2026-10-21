"""Reciprocal Rank Fusion (RRF): merge ranked lists from different search methods.

Vector search returns cosine distances and full-text search returns ts_rank scores.
Those scales aren't comparable, so adding them needs hand-tuned weights. RRF ignores
the raw scores and only uses rank: each list gives an item 1 / (k + rank). Items found
by both methods rise to the top. k=60 is the value from the original paper
(Cormack et al., 2009) and damps the gap between rank 1 and rank 2.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion[T: Hashable](
    ranked_lists: Sequence[Sequence[T]], k: int = 60
) -> list[tuple[T, float]]:
    scores: dict[T, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
