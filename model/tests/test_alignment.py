"""
test_alignment.py
=================
Unit tests for src/topic_alignment.align_phi: under a random permutation (with and
without noise) the Hungarian matcher must recover the original topic order.

Run from the model/ folder:
    pytest tests/ -q
    # or, without pytest:
    python tests/test_alignment.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from topic_alignment import align_phi, align_phi_perm  # noqa: E402


def _random_phi(K: int, V: int, seed: int) -> np.ndarray:
    """A random row-stochastic (K, V) matrix, like a topic->ingredient phi."""
    rng = np.random.default_rng(seed)
    phi = rng.dirichlet(np.ones(V) * 0.3, size=K)
    return phi


def test_recovers_exact_permutation():
    """phi_new is an exact row permutation of phi_ref -> align must invert it."""
    phi_ref = _random_phi(K=6, V=40, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(6)
    phi_new = phi_ref[perm]                       # phi_new[j] = phi_ref[perm[j]]

    aligned, recovered = align_phi_perm(phi_ref, phi_new)

    assert np.allclose(aligned, phi_ref), "aligned phi must equal the reference"
    # recovered is the inverse permutation: perm[recovered] == identity
    assert np.array_equal(perm[recovered], np.arange(6))


def test_recovers_permutation_under_noise():
    """Small noise must not break the matching (topics still nearest to their twin)."""
    phi_ref = _random_phi(K=8, V=60, seed=2)
    rng = np.random.default_rng(3)
    perm = rng.permutation(8)
    noise = rng.normal(scale=1e-3, size=phi_ref.shape)
    phi_new = np.clip(phi_ref[perm] + noise[perm], 1e-9, None)
    phi_new = phi_new / phi_new.sum(axis=1, keepdims=True)

    _, recovered = align_phi_perm(phi_ref, phi_new)
    assert np.array_equal(perm[recovered], np.arange(8))


def test_identity_when_already_aligned():
    """phi_new == phi_ref -> permutation is the identity."""
    phi_ref = _random_phi(K=5, V=30, seed=4)
    aligned, recovered = align_phi_perm(phi_ref, phi_ref.copy())
    assert np.array_equal(recovered, np.arange(5))
    assert np.allclose(aligned, phi_ref)


def test_many_random_permutations():
    """Stress: 25 random permutations of varying K must all be recovered."""
    rng = np.random.default_rng(5)
    for _ in range(25):
        K = int(rng.integers(2, 13))
        phi_ref = _random_phi(K=K, V=50, seed=int(rng.integers(0, 1_000_000)))
        perm = rng.permutation(K)
        _, recovered = align_phi_perm(phi_ref, phi_ref[perm])
        assert np.array_equal(perm[recovered], np.arange(K)), f"failed at K={K}"


def test_shape_mismatch_raises():
    phi_ref = _random_phi(K=4, V=20, seed=6)
    phi_new = _random_phi(K=4, V=21, seed=7)
    try:
        align_phi(phi_ref, phi_new)
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


if __name__ == "__main__":
    # Lightweight runner so the file works without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} alignment tests passed.")
