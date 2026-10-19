"""Exact bad-good transition counts in O(N log N), including both kinds of ties."""
from __future__ import annotations

import numpy as np

STATES = ("wrong", "tie", "correct")
UTILITY = np.array([0.0, 0.5, 1.0])


def pair_transitions(y, before, after):
    """Return [bad, before_state, after_state] counts; order matches input bads.

    wrong: bad score < good score; correct: bad score > good score.
    Good points are swept in before-score order; a Fenwick tree counts the
    after-score coordinates. No bad x good matrix or pair sampling is used.
    """
    y, a, b = np.asarray(y), np.asarray(before, float), np.asarray(after, float)
    if not (len(y) == len(a) == len(b)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Нужны одинаковые длины и бинарный target")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("pair_transitions требует конечные парные скоры")
    ga, gb, ba, bb = a[y == 0], b[y == 0], a[y == 1], b[y == 1]
    nb, ng = len(ba), len(ga)
    result = np.zeros((nb, 3, 3), dtype=np.int64)
    if not nb or not ng:
        return result
    coordinates = np.unique(gb)
    order = np.argsort(ga, kind="stable")
    ga, gb = ga[order], gb[order]
    good_indices = np.searchsorted(coordinates, gb) + 1
    bad_order = np.argsort(ba, kind="stable")
    lower = np.searchsorted(coordinates, bb, side="left")
    upper = np.searchsorted(coordinates, bb, side="right")

    def sweep(inclusive):
        bit = np.zeros(len(coordinates) + 1, dtype=np.int64)
        counts = np.zeros((nb, 3), dtype=np.int64)
        cursor = 0

        def prefix(index):
            value = 0
            while index:
                value += bit[index]
                index -= index & -index
            return value

        for i in bad_order:
            while cursor < ng and (ga[cursor] < ba[i] or (inclusive and ga[cursor] == ba[i])):
                j = int(good_indices[cursor])
                while j < len(bit):
                    bit[j] += 1
                    j += j & -j
                cursor += 1
            lt, le = prefix(int(lower[i])), prefix(int(upper[i]))
            counts[i] = (cursor - le, le - lt, lt)
        return counts

    strict, inclusive = sweep(False), sweep(True)
    sorted_b = np.sort(gb)
    lt = np.searchsorted(sorted_b, bb, side="left")
    le = np.searchsorted(sorted_b, bb, side="right")
    all_counts = np.stack((ng - le, le - lt, lt), axis=1)
    result[:, 2] = strict
    result[:, 1] = inclusive - strict
    result[:, 0] = all_counts - inclusive
    return result


def transition_statistics(matrix):
    counts = np.asarray(matrix, dtype=np.int64).sum(axis=0)
    change = UTILITY[None, :] - UTILITY[:, None]
    gained = float((counts * np.maximum(change, 0)).sum())
    lost = float((counts * np.maximum(-change, 0)).sum())
    total = int(counts.sum())
    result = {
        "pairs_bad_good_total": total,
        "pairs_improved": int(counts[change > 0].sum()),
        "pairs_worsened": int(counts[change < 0].sum()),
        "pairs_unchanged": int(np.trace(counts)),
        "pairs_wrong_to_correct": int(counts[0, 2]),
        "pairs_correct_to_wrong": int(counts[2, 0]),
        "pair_gain_units": gained, "pair_loss_units": lost,
        "pair_net_units": gained - lost,
        "gini_delta_from_pairs": 2 * (gained - lost) / total if total else np.nan,
    }
    for i, old in enumerate(STATES):
        for j, new in enumerate(STATES):
            result[f"pairs_{old}_to_{new}"] = int(counts[i, j])
    return result


def per_default_pair_statistics(matrix):
    change = UTILITY[None, :] - UTILITY[:, None]
    return {
        "pairs_improved": matrix[:, change > 0].sum(axis=1),
        "pairs_worsened": matrix[:, change < 0].sum(axis=1),
        "pairs_wrong_to_correct": matrix[:, 0, 2],
        "pairs_correct_to_wrong": matrix[:, 2, 0],
        "pair_gain_units": (matrix * np.maximum(change, 0)).sum(axis=(1, 2)),
        "pair_loss_units": (matrix * np.maximum(-change, 0)).sum(axis=(1, 2)),
    }


class WeightedGini:
    """Cache score sorting for fast paired bootstrap with multiplicity weights."""
    def __init__(self, y, scores):
        self.y = np.asarray(y, dtype=int)
        score = np.asarray(scores, dtype=float)
        if not np.isfinite(score).all():
            raise ValueError("WeightedGini требует конечные скоры")
        self.order = np.argsort(score, kind="stable")
        ordered = score[self.order]
        self.starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]]) if len(score) else np.array([], dtype=int)

    def __call__(self, weights=None):
        if not len(self.order):
            return np.nan
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights, float)
        bad = np.add.reduceat((w * self.y)[self.order], self.starts)
        good = np.add.reduceat((w * (1 - self.y))[self.order], self.starts)
        if bad.sum() <= 0 or good.sum() <= 0:
            return np.nan
        auc = np.dot(bad, np.cumsum(good) - 0.5 * good) / (bad.sum() * good.sum())
        return float(2 * auc - 1)
