"""
topic_alignment.py
==================
Topic-label alignment for the Bootstrap refits in `recipe_recommender.train_lda`.

Why this exists
---------------
Each Bootstrap refit of an sklearn LDA returns topics in an arbitrary order: topic
"0" in one fit has nothing to do with topic "0" in another (topics are exchangeable
-- the classic LDA label-switching non-identifiability). Before we can treat the
Bootstrap fits as comparable samples of phi -- i.e. before `phi_samples[:, k, :]`
means "the same flavor topic k across all draws" -- every fit's topics must be
permuted to line up with a common reference (the main point estimate phi_hat).

We *fix* this explicitly by matching each fit's topics to phi_hat with the Hungarian
algorithm on cosine similarity, then reordering the rows.

The match is a linear assignment problem: maximize the total cosine similarity of the
ref<->new topic pairing under a one-to-one constraint. `scipy.optimize.
linear_sum_assignment` solves it exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def align_phi_perm(phi_ref: np.ndarray, phi_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the permutation that aligns ``phi_new``'s topics to ``phi_ref``.

    Parameters
    ----------
    phi_ref, phi_new : np.ndarray, shape (K, V)
        Topic->ingredient distributions (rows are topics). Must share K and V.

    Returns
    -------
    aligned : np.ndarray, shape (K, V)
        ``phi_new`` with its rows reordered so that ``aligned[k]`` is the topic of
        ``phi_new`` that best matches ``phi_ref[k]``.
    perm : np.ndarray, shape (K,)
        The column indices used, i.e. ``aligned == phi_new[perm]``. ``perm[k]`` is the
        index of the ``phi_new`` topic assigned to reference topic ``k``.
    """
    phi_ref = np.asarray(phi_ref, dtype=float)
    phi_new = np.asarray(phi_new, dtype=float)
    if phi_ref.shape != phi_new.shape:
        raise ValueError(f"phi_ref {phi_ref.shape} and phi_new {phi_new.shape} "
                         "must have the same shape (K, V)")
    if phi_ref.ndim != 2:
        raise ValueError(f"expected 2-D (K, V) arrays, got ndim={phi_ref.ndim}")

    # sim[i, j] = cosine similarity between ref topic i and new topic j.
    # cosine_similarity normalizes internally and returns 0 for zero rows (no NaN).
    sim = cosine_similarity(phi_ref, phi_new)                 # (K, K)
    # Hungarian minimizes cost; negate to MAXIMIZE total similarity.
    row_ind, col_ind = linear_sum_assignment(-sim)
    # row_ind is 0..K-1 ascending for a square matrix, so col_ind is already the
    # permutation indexed by reference topic.
    perm = col_ind[np.argsort(row_ind)]
    return phi_new[perm], perm


def align_phi(phi_ref: np.ndarray, phi_new: np.ndarray) -> np.ndarray:
    """Reorder ``phi_new``'s topics to align with ``phi_ref`` (rows are topics).

    Thin wrapper over :func:`align_phi_perm` that returns only the reordered array.
    """
    aligned, _ = align_phi_perm(phi_ref, phi_new)
    return aligned
