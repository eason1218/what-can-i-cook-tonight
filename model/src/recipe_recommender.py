"""
recipe_recommender.py
=====================
A Bayesian LDA recipe recommender over latent "flavor topics".

Pipeline ("What Can I Cook Tonight?")
-------------------------------------
    Step 1  train_lda()             -> LDA topic model: sklearn point estimate of phi on
                                       the full corpus + Bootstrap pseudo-posterior of phi
    Step 2  filter_candidates()     -> keep recipes the user can actually make (coverage)
    Step 3  infer_user_posterior()  -> Bayesian posterior over flavor topics for an ingredient set
    Step 4  score_recipes()         -> composite score using the full (Bootstrap) phi posterior
    Step 5  recommend()             -> Top-5 recipes as a list of dicts

The model (sklearn point estimate + Bootstrap pseudo-posterior)
--------------------------------------------------------------
A fully-Bayesian NUTS fit of phi does not scale past a few hundred recipes, so phi is
fit as a *point estimate* with sklearn's LatentDirichletAllocation on the FULL corpus
(phi_hat = row-normalized components_). phi's uncertainty is then approximated by
*Bootstrap*: refit B times on resampled recipes (warm-started from phi_hat) and keep
the B aligned fits as pseudo-samples `phi_samples`. K is chosen by held-out perplexity.

Where the Bayesian content lives
--------------------------------
Every downstream quantity (the user's flavor profile, each recipe's flavor profile,
the flavor-alignment similarity) is computed *per phi-sample and then averaged*, so
uncertainty is propagated to the final score and reported per recommendation. The
genuinely uncertain, fully-Bayesian quantity is the *user's* topic posterior (Step 3):
inferred from a handful of ingredients via Bayes' theorem applied to each phi-sample.
phi itself is a point estimate, so its Bootstrap spread is small on a large corpus --
we call it "bootstrap stability", not a Bayesian posterior (see README).

Label alignment (topic identifiability)
---------------------------------------
Topics are exchangeable, so each Bootstrap refit returns them in an arbitrary order.
Before they can be treated as comparable samples, every refit's phi is permuted back
onto phi_hat with the Hungarian algorithm on cosine similarity
(topic_alignment.align_phi), so phi_samples[:, k, :] is a coherent "topic k" across draws.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import softmax

# sklearn / joblib are imported lazily inside train_lda (and the Bootstrap worker),
# so importing this module stays light and the worker processes spawn cheaply.

# --------------------------------------------------------------------------- #
#  Hyper-priors (as specified in the task)                                     #
# --------------------------------------------------------------------------- #
ALPHA_PRIOR = 0.1    # symmetric Dirichlet on theta_m  -> sparse topic mix per recipe
BETA_PRIOR  = 0.01   # symmetric Dirichlet on phi_k    -> sparse ingredient set per topic

# Scoring (Step 4):  score = coverage^A * overlap_bonus * alignment^B * rating^C
SCORE_ALPHA = 2.0    # coverage dominates: a recipe you cannot make is useless
SCORE_BETA  = 1.0    # flavor alignment
SCORE_GAMMA = 1.0    # rating prior
SCORE_TAU   = 4.0    # saturation scale of the absolute-overlap bonus: demotes
                     # trivially short recipes that hit coverage=1.0 on 2-3 items
RATING_KAPPA = 5.0   # Beta-Binomial smoothing strength (pseudo-count toward global mean)

_EPS = 1e-12         # numerical floor for logs / KL

# Dietary filters (Step 5 option). Heuristic: a recipe is dropped if any of its
# normalized ingredient *words* equals a blocked token. Editable -- extend freely.
# Vegetarian is reliable; vegan is conservative (it also catches e.g. "coconut milk"
# via "milk"), so it errs toward over-excluding. Edit DIET_BLOCKLIST to taste.
_MEAT = {
    "chicken", "beef", "pork", "bacon", "ham", "sausage", "turkey", "lamb", "veal",
    "duck", "fish", "salmon", "tuna", "shrimp", "prawn", "crab", "lobster", "anchovy",
    "meat", "steak", "mince", "gelatin", "gelatine", "clam", "oyster", "mussel",
    "scallop", "cod", "tilapia", "sardine", "pepperoni", "prosciutto", "chorizo",
}
_NON_VEGAN = _MEAT | {
    "milk", "butter", "cheese", "cream", "egg", "yogurt", "yoghurt", "honey",
    "mayonnaise", "mayo", "ghee", "custard", "buttermilk", "whey", "casein", "lard",
}
DIET_BLOCKLIST = {"vegetarian": _MEAT, "vegan": _NON_VEGAN}


# =========================================================================== #
#  0 · Ingredient normalization (shared by data prep AND query time)          #
# =========================================================================== #
# The user types free-text ingredients ("Tomatoes", "olive oil"). Recipe
# ingredients come pre-tokenized from Food.com. To make set-intersection
# (coverage) meaningful, BOTH sides must pass through the *same* canonicalizer.

_WS = re.compile(r"\s+")
_NON_ALPHA = re.compile(r"[^a-z\s]")

# Generic, non-discriminating tokens that should not define a recipe's identity.
_GENERIC = {
    "fresh", "dried", "frozen", "raw", "cooked", "large", "small", "medium",
    "extra", "fine", "ground", "whole", "half", "ripe", "organic", "free",
    "range", "hot", "cold", "warm", "room", "temperature", "size", "boneless",
    "skinless", "low", "fat", "reduced", "lean", "virgin", "purpose", "all",
}

try:  # inflect gives high-quality singularization ("tomatoes" -> "tomato")
    import inflect as _inflect_mod
    _INFLECT = _inflect_mod.engine()
except Exception:  # pragma: no cover - inflect is optional
    _INFLECT = None


def _singularize(word: str) -> str:
    """Singularize a single word; inflect if available, else a tiny rule set."""
    if _INFLECT is not None:
        s = _INFLECT.singular_noun(word)
        return s if s else word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


@lru_cache(maxsize=None)
def normalize_token(raw: str) -> str:
    """Canonicalize one ingredient phrase: lowercase, de-punctuate, drop generic
    modifiers, singularize each remaining word. Returns "" if nothing survives.

    Cached: it is a pure function called ~half a million times over the catalogue
    but on only ~7k distinct strings (inflect singularization is the cost)."""
    if not isinstance(raw, str):
        return ""
    s = _NON_ALPHA.sub(" ", raw.lower())
    s = _WS.sub(" ", s).strip()
    words = [_singularize(w) for w in s.split() if w not in _GENERIC and len(w) >= 2]
    return " ".join(words)


def normalize_list(items: Sequence[str]) -> list[str]:
    """Canonicalize a list of ingredient phrases, dropping empties & duplicates
    while preserving order."""
    out, seen = [], set()
    for it in items:
        t = normalize_token(it)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _coerce_ingredients(cell) -> list[str]:
    """Accept an ingredients cell that may be a python list, a JSON string, or a
    python-literal string, and return a clean list of strings."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if isinstance(cell, str):
        for parser in (json.loads, _literal_eval):
            try:
                val = parser(cell)
                if isinstance(val, list):
                    return [str(x) for x in val]
            except Exception:
                continue
    return []


def _literal_eval(s):
    import ast
    return ast.literal_eval(s)


# =========================================================================== #
#  Container for the trained model                                            #
# =========================================================================== #
@dataclass
class LDAModel:
    """Everything downstream steps need from the fitted hybrid LDA.

    phi_samples : np.ndarray, shape (B, K, V)
        Bootstrap pseudo-samples of the topic->ingredient distributions, each
        aligned to phi_mean (so topic k is coherent across draws). Kept -- not
        collapsed to a mean -- so Steps 3-5 can propagate uncertainty.
    phi_mean : np.ndarray, shape (K, V)
        The sklearn point estimate phi_hat (the model's best guess of phi).
    topic_prior : np.ndarray, shape (K,)
        Empirical marginal topic frequency  P(topic)  = average recipe topic mix.
    vocab / ingr2idx : the modeled ingredient space (the LDA "words").
    topic_labels : human-readable tag per topic (its top ingredients).
    perplexity_table : DataFrame of held-out perplexity per candidate K (the model
        selection metric, lower=better); empty if K was fixed.
    bootstrap_stability : mean per-element std of phi across Bootstrap samples.
        NB: this is resampling *stability*, not a Bayesian posterior width.
    """
    best_k: int
    phi_samples: np.ndarray          # (B, K, V)
    phi_mean: np.ndarray             # (K, V)
    topic_prior: np.ndarray          # (K,)
    vocab: list[str]
    ingr2idx: dict[str, int]
    topic_labels: list[str]
    perplexity_table: pd.DataFrame
    bootstrap_stability: float = None

    @property
    def n_samples(self) -> int:
        return self.phi_samples.shape[0]


_MODEL_FIELDS = ("best_k", "phi_samples", "phi_mean", "topic_prior", "vocab",
                 "ingr2idx", "topic_labels", "perplexity_table", "bootstrap_stability")


def save_model(model: LDAModel, path: str = "models/lda_model.pkl") -> None:
    """Persist the fitted model."""
    import os
    import pickle
    payload = {k: getattr(model, k) for k in _MODEL_FIELDS}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_model(path: str = "models/lda_model.pkl") -> LDAModel:
    """Load a model saved by save_model (so recommend() needn't retrain)."""
    import pickle
    with open(path, "rb") as f:
        p = pickle.load(f)
    return LDAModel(**p)


# =========================================================================== #
#  Corpus construction                                                         #
# =========================================================================== #
_INGR_CACHE: dict = {}     # id(df) -> (n_rows, Series of normalized ingredient lists)


def _get_recipe_ingredients(df: pd.DataFrame) -> pd.Series:
    """Series of normalized ingredient-name lists for every recipe.

    Cached per DataFrame (keyed on id+length): the catalogue does not change
    between recommend() calls, so we normalize the ~50k recipes only once instead
    of on every call. (Consistent with the model cache, which also keys on id(df).)
    """
    key = id(df)
    hit = _INGR_CACHE.get(key)
    if hit is not None and hit[0] == len(df):
        return hit[1]
    series = df["ingredients"].apply(lambda c: normalize_list(_coerce_ingredients(c)))
    _INGR_CACHE[key] = (len(df), series)
    return series


def _label_topics(phi_mean: np.ndarray, vocab: list[str], top_n: int = 3) -> list[str]:
    """Human-readable label for each topic = its top ingredients (by posterior
    mean phi). Used for `flavor_tags`."""
    labels = []
    for k in range(phi_mean.shape[0]):
        top = np.argsort(phi_mean[k])[::-1][:top_n]
        labels.append(" / ".join(vocab[i] for i in top))
    return labels


# =========================================================================== #
#  STEP 1 · sklearn point-estimate phi + Bootstrap pseudo-posterior            #
# =========================================================================== #
# A fully-Bayesian NUTS fit of phi's posterior does not scale past a few hundred
# recipes, so we:
#   (1) fit ONE sklearn LDA on the FULL corpus -> point estimate phi_hat;
#   (2) approximate phi's uncertainty by BOOTSTRAP: refit on resampled recipes B
#       times, each warm-started from phi_hat, and keep the B fits as pseudo-samples.
# Each Bootstrap fit's topics are re-ordered to phi_hat with the Hungarian matcher
# (topic_alignment.align_phi), so phi_samples[:, k, :] is a coherent "topic k"
# across draws.
#
# Honest caveat (see README): on a large corpus phi is very well determined, so the
# Bootstrap spread is small *by design*. We therefore call it "bootstrap stability",
# NOT "posterior uncertainty": it measures how stable phi_hat is under resampling,
# not a Bayesian posterior. The genuinely uncertain quantity is the *user's* topic
# posterior (Step 3), which is inferred from a handful of ingredients -- and that
# stays fully Bayesian. We use the m-out-of-n Bootstrap (resample BOOTSTRAP_SIZE
# recipes per replicate, not all N): it is a recognized Bootstrap variant and keeps
# each refit's E-step cheap enough to run B of them in seconds.

# default size of each Bootstrap resample (m-out-of-n). Capped at the corpus size.
_HYBRID_BOOTSTRAP_SIZE = 10_000


def _build_doc_term_matrix(df: pd.DataFrame, vocab_top_n: int, min_df: int):
    """Full-corpus sparse document-term count matrix for sklearn LDA.

    Reuses the cached, normalized ingredient lists (`_get_recipe_ingredients`) so we
    canonicalize the catalogue only once. Vocabulary = the `vocab_top_n` most
    frequent ingredients (by document frequency, with a `min_df` floor). Row order
    matches `df`, but downstream never relies on that: recipe flavor profiles are
    recomputed from `phi_samples` + `ingr2idx`, not from this matrix.

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (n_recipes, V)   (counts; effectively 0/1)
    vocab : list[str]
    ingr2idx : dict[str, int]
    """
    from collections import Counter
    from scipy.sparse import csr_matrix

    lists = _get_recipe_ingredients(df)
    dfc = Counter()
    for lst in lists:
        dfc.update(set(lst))
    vocab = [w for w, c in dfc.most_common() if c >= min_df][:vocab_top_n]
    ingr2idx = {w: i for i, w in enumerate(vocab)}

    rows, cols = [], []
    for m, lst in enumerate(lists):
        for w in set(lst):                       # dedup -> binary presence counts
            j = ingr2idx.get(w)
            if j is not None:
                rows.append(m)
                cols.append(j)
    X = csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                   shape=(len(lists), len(vocab)))
    return X, vocab, ingr2idx


def _fit_sklearn_lda(X, K, alpha, beta, *, max_iter, random_state, n_jobs=1,
                     max_doc_update_iter=30, learning_method="batch"):
    """One sklearn LatentDirichletAllocation fit. Centralized so the K-sweep and the
    main fit share identical hyper-parameters."""
    from sklearn.decomposition import LatentDirichletAllocation
    return LatentDirichletAllocation(
        n_components=K, doc_topic_prior=alpha, topic_word_prior=beta,
        learning_method=learning_method, max_iter=max_iter,
        max_doc_update_iter=max_doc_update_iter, random_state=random_state,
        n_jobs=n_jobs).fit(X)


def _select_k_perplexity(X, K_candidates, alpha, beta, random_state,
                         k_select_size=12_000, holdout_frac=0.10, max_iter=8):
    """Choose K by held-out perplexity (lower = better).

    Runs on a random sub-sample of `k_select_size` recipes (K is a coarse structural
    choice that a sub-sample determines amply, and a full-corpus sweep would dominate
    the wall-time). For each K: fit on 90%, score perplexity on the held-out 10%.
    Returns (best_K, table) where table has columns K, holdout_perplexity.
    """
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    sel = rng.permutation(N)[:min(k_select_size, N)]
    Xs = X[sel]
    cut = max(1, int(Xs.shape[0] * (1.0 - holdout_frac)))
    Xtr, Xte = Xs[:cut], Xs[cut:]

    rows = []
    for K in K_candidates:
        m = _fit_sklearn_lda(Xtr, int(K), alpha, beta, max_iter=max_iter,
                             random_state=random_state)
        rows.append({"K": int(K), "holdout_perplexity": float(m.perplexity(Xte))})
        print(f"[train_lda]   K={int(K):>2d}  "
              f"holdout_perplexity={rows[-1]['holdout_perplexity']:.1f}")
    table = pd.DataFrame(rows)
    best_K = int(table.loc[table["holdout_perplexity"].idxmin(), "K"])
    return best_K, table


def _bootstrap_fit_one(args: tuple) -> np.ndarray:
    """One Bootstrap refit, warm-started from phi_hat and aligned back to it.

    Module-level and self-contained (imports inside) so it is picklable for joblib
    worker processes and cheap to spawn (no heavy top-level deps). `Xb` is the pre-resampled
    (m, V) matrix; the parent does the resampling so workers receive only what they
    need.

    Warm start: initialize the variational topic-word parameter `components_`
    (Dirichlet lambda) at `phi_hat * m + beta` so the fit starts at the point
    estimate, then run a few online `partial_fit` passes on the resample. We refresh
    sklearn's cached `exp(E[log beta])` from the injected components with scipy's
    digamma -- no fragile private sklearn internals beyond `_init_latent_vars`.
    """
    Xb, K, V, alpha, beta, phi_hat, n_warm_iter, max_doc_update_iter, seed = args
    from sklearn.decomposition import LatentDirichletAllocation
    from scipy.special import psi
    from topic_alignment import align_phi

    m = Xb.shape[0]
    lda = LatentDirichletAllocation(
        n_components=K, doc_topic_prior=alpha, topic_word_prior=beta,
        learning_method="online", max_iter=1,
        max_doc_update_iter=max_doc_update_iter, random_state=seed, n_jobs=1)
    lda._init_latent_vars(V)                                  # allocate components_
    lda.components_ = phi_hat * float(m) + beta               # warm start at phi_hat
    lda.exp_dirichlet_component_ = np.exp(
        psi(lda.components_) - psi(lda.components_.sum(axis=1))[:, None])
    lda.n_batch_iter_ = 0
    for _ in range(n_warm_iter):
        lda.partial_fit(Xb)
    phi_b = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    return align_phi(phi_hat, phi_b)                          # (K, V), topic-aligned


def train_lda(df: pd.DataFrame,
              K: int | None = None,
              K_candidates: Sequence[int] = (4, 6, 8, 10, 12),
              n_bootstrap: int = 50,
              vocab_top_n: int = 500,
              alpha: float = ALPHA_PRIOR,
              beta: float = BETA_PRIOR,
              random_state: int = 42,
              n_jobs: int = -1,
              *,
              min_df: int = 5,
              bootstrap_size: int | None = None,
              n_warm_iter: int = 2,
              max_doc_update_iter: int = 8,
              main_max_iter: int = 8) -> LDAModel:
    """Step 1: fit the LDA topic model and return an `LDAModel`.

    sklearn point estimate of phi on the full corpus + Bootstrap pseudo-posterior:

      1. Build the full-corpus doc-term matrix (vocab = top `vocab_top_n` ingredients).
      2. If `K` is None, choose it by held-out perplexity over `K_candidates`.
      3. Fit ONE sklearn LDA on the full corpus  -> phi_hat (the point estimate).
      4. Bootstrap `n_bootstrap` times (m-out-of-n resample, warm-started from
         phi_hat, parallel) and align each fit to phi_hat  -> phi_samples.
      5. Empirical topic prior P(topic) = mean doc-topic mixture (from transform).

    Returns an `LDAModel` (phi_samples, phi_mean, topic_prior, vocab, ingr2idx,
    topic_labels, perplexity_table, bootstrap_stability) that Steps 2-5 consume.

    `K` defaults to None (auto-select). `alpha`/`beta` are the sklearn
    doc_topic_prior / topic_word_prior. NB: `phi_samples`' spread is *bootstrap
    stability*, not a Bayesian posterior -- on a large corpus phi is well determined,
    so it is small by design; the genuinely uncertain, fully-Bayesian quantity is the
    user's topic posterior (Step 3). See module docstring / README.
    """
    X, vocab, ingr2idx = _build_doc_term_matrix(df, vocab_top_n, min_df)
    N, V = X.shape
    print(f"[train_lda] corpus: {N} recipes, vocab V={V}, "
          f"{int(X.nnz)} ingredient tokens")

    # ---- Step 2: choose K (held-out perplexity) unless the user fixed it --------
    if K is None:
        print(f"[train_lda] selecting K from {tuple(K_candidates)} "
              f"by held-out perplexity ...")
        best_K, perp_table = _select_k_perplexity(
            X, K_candidates, alpha, beta, random_state, max_iter=main_max_iter)
        print(f"[train_lda] selected K={best_K}")
    else:
        best_K = int(K)
        perp_table = pd.DataFrame(columns=["K", "holdout_perplexity"])
        print(f"[train_lda] K fixed at {best_K} (no sweep)")

    # ---- Step 3: main fit on the FULL corpus -> phi_hat ------------------------
    print(f"[train_lda] main fit (K={best_K}) on the full corpus ...")
    main = _fit_sklearn_lda(X, best_K, alpha, beta, max_iter=main_max_iter,
                            random_state=random_state, n_jobs=n_jobs)
    phi_hat = main.components_ / main.components_.sum(axis=1, keepdims=True)  # (K,V)

    # ---- Step 5 (prior): empirical marginal P(topic) from the doc-topic mix ----
    # = average recipe topic mixture, the sklearn analogue of the full-Bayes
    #   "mean theta". transform() is an E-step; do it on a sub-sample for speed.
    rng = np.random.default_rng(random_state)
    prior_idx = rng.permutation(N)[:min(N, 15_000)]
    theta = main.transform(X[prior_idx])                       # (n_sub, K)
    topic_prior = theta.mean(axis=0)
    topic_prior = topic_prior / topic_prior.sum()

    # ---- Step 4: Bootstrap pseudo-posterior of phi (parallel, m-out-of-n) ------
    m = min(bootstrap_size or _HYBRID_BOOTSTRAP_SIZE, N)
    print(f"[train_lda] bootstrap: {n_bootstrap} refits, m={m} recipes each, "
          f"warm-started (n_jobs={n_jobs}) ...")

    def _jobs():
        for b in range(n_bootstrap):
            idx = rng.integers(0, N, size=m)                  # resample WITH replace
            yield (X[idx], best_K, V, alpha, beta, phi_hat, n_warm_iter,
                   max_doc_update_iter, random_state + 1 + b)

    from joblib import Parallel, delayed
    samples = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_fit_one)(job) for job in _jobs())
    phi_samples = np.stack(samples, axis=0)                    # (B, K, V), aligned

    bootstrap_stability = float(phi_samples.std(axis=0).mean())
    print(f"[train_lda] bootstrap_stability (mean phi std) = "
          f"{bootstrap_stability:.5f}")

    topic_labels = _label_topics(phi_hat, vocab)
    return LDAModel(
        best_k=best_K, phi_samples=phi_samples, phi_mean=phi_hat,
        topic_prior=topic_prior, vocab=vocab, ingr2idx=ingr2idx,
        topic_labels=topic_labels, perplexity_table=perp_table,
        bootstrap_stability=bootstrap_stability)


# =========================================================================== #
#  STEP 2 · Filter candidates by ingredient coverage                          #
# =========================================================================== #
def filter_candidates(user_ingredients: list[str], df: pd.DataFrame,
                      threshold: float = 0.7, fallback: float = 0.5,
                      min_recipes: int = 20) -> pd.DataFrame:
    """Keep recipes the user can (mostly) make.

        coverage = |user ∩ recipe| / |recipe|

    Drop recipes below `threshold` (0.7). If that leaves fewer than `min_recipes`,
    relax to `fallback` (0.5). Returns a copy of df (filtered) augmented with
    coverage bookkeeping columns.
    """
    user_set = set(normalize_list(user_ingredients))
    recipe_lists = _get_recipe_ingredients(df)

    inter = recipe_lists.apply(lambda lst: len(user_set.intersection(lst)))
    size = recipe_lists.apply(len).clip(lower=1)
    coverage = (inter / size).astype(float)

    work = df.copy()
    work["_ingredients_norm"] = recipe_lists.values
    work["_inter"] = inter.values
    work["_size"] = size.values
    work["coverage"] = coverage.values

    kept = work[work["coverage"] >= threshold]
    used = threshold
    if len(kept) < min_recipes:
        # Relax the bar: better to recommend slightly-incomplete recipes than
        # to return an empty list.
        kept = work[work["coverage"] >= fallback]
        used = fallback
    print(f"[filter_candidates] threshold={used}: {len(kept)} candidate recipes")
    return kept.reset_index(drop=True)


# =========================================================================== #
#  STEP 3 · Bayesian posterior over flavor topics for an ingredient set        #
# =========================================================================== #
def infer_user_posterior(ingredients: Sequence[str], model: LDAModel,
                         phi_samples: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Posterior over latent topics given a set of observed ingredients.

    Treat the ingredients as evidence and apply Bayes' theorem *per posterior
    sample of phi*:

        P(topic=k | ingredients) ∝ P(ingredients | topic=k) · P(topic=k)
        P(ingredients | topic=k) = Π_i phi[k, i]      (ingredients independent | topic)
        P(topic=k)               = empirical marginal topic frequency

    We do this for every Bootstrap sample s of phi, giving a posterior over the topic
    weights that *carries the model's uncertainty* (rather than collapsing phi's
    Bootstrap spread to a single point estimate).

    Note: this same routine is reused to profile *recipes* (Step 4) -- a recipe's
    flavor profile is just the topic posterior of its own ingredient list.

    Returns
    -------
    mean : (K,)    posterior mean topic-weight vector  (the "flavor profile")
    post : (S, K)  per-sample topic posteriors         (uncertainty is retained)
    """
    phi = model.phi_samples if phi_samples is None else phi_samples   # (S, K, V)
    S, K, _ = phi.shape
    log_prior = np.log(model.topic_prior + _EPS)                     # (K,)

    idxs = [model.ingr2idx[t] for t in normalize_list(ingredients)
            if t in model.ingr2idx]

    if not idxs:
        # No in-vocab evidence -> posterior falls back to the prior (broadcast).
        post = np.tile(softmax(log_prior), (S, 1))
        return post.mean(axis=0), post

    # log P(ingredients | topic=k) summed over observed ingredients, per sample s.
    # Index the observed ingredients BEFORE taking the log (cheap: S*K*|idxs|),
    # since this is called once per candidate recipe.
    log_lik = np.log(phi[:, :, idxs] + _EPS).sum(axis=2)            # (S, K)
    log_post = log_lik + log_prior[None, :]                          # (S, K)
    post = softmax(log_post, axis=1)                                 # normalize over k
    return post.mean(axis=0), post


# =========================================================================== #
#  STEP 4 · Score & rank candidates with the full posterior                    #
# =========================================================================== #
def _kl_similarity(recipe_post: np.ndarray, user_post: np.ndarray
                   ) -> tuple[float, float]:
    """Flavor alignment = exp(-KL(recipe || user)), averaged over Bootstrap samples.

    recipe_post, user_post : (S, K)  -- paired sample-by-sample (same phi draw s),
    so the KL is evaluated within a single coherent draw and then averaged. We also
    return the std of the per-sample similarity, which becomes the recipe's
    `posterior_uncertainty` (bootstrap stability; small as phi is well determined).
    """
    kl = np.sum(recipe_post * (np.log(recipe_post + _EPS) - np.log(user_post + _EPS)),
                axis=1)                                              # (S,)
    sim = np.exp(-kl)                                               # (S,)
    return float(sim.mean()), float(sim.std())


def _bayesian_rating(avg_rating: float, n_ratings: float, mu_global: float,
                     kappa: float = RATING_KAPPA) -> float:
    """Beta-Binomial / shrinkage smoothed rating.

        rating = (avg_rating * n_ratings + mu_global * kappa) / (n_ratings + kappa)

    A recipe with few ratings is pulled toward the global mean mu_global; a recipe
    with many ratings keeps its own average. kappa is the strength of the prior
    (equivalent pseudo-observations).
    """
    return (avg_rating * n_ratings + mu_global * kappa) / (n_ratings + kappa)


def score_recipes(user_ingredients: list[str], candidates: pd.DataFrame,
                  user_post: np.ndarray, model: LDAModel, df: pd.DataFrame,
                  return_internal: bool = False) -> list[dict]:
    """Composite Bayesian score for every candidate recipe:

        score = coverage^alpha · overlap_bonus · flavor_alignment^beta · rating^gamma
                (alpha=2, beta=1, gamma=1;  overlap_bonus = 1 - exp(-|inter|/tau))

    coverage is a ratio (|inter|/|recipe|), so tiny recipes hit 1.0 trivially; the
    saturating overlap_bonus in the absolute number of matched ingredients demotes
    those and rewards substantively bigger matches. Every flavor quantity is
    computed over the full posterior (per-sample, then averaged), so uncertainty
    is propagated end to end.
    """
    mu_global = float(df["avg_rating"].mean())     # dataset mean rating (prior mean)
    user_set = set(normalize_list(user_ingredients))

    results = []
    for _, row in candidates.iterrows():
        recipe_ingr = row["_ingredients_norm"]

        # --- Step 4a: recipe's own flavor profile (reuse the Step-3 machinery) --
        _, recipe_post = infer_user_posterior(recipe_ingr, model)   # (S, K)

        # --- Step 4b: flavor alignment (posterior KL -> similarity) ------------
        alignment, uncertainty = _kl_similarity(recipe_post, user_post)

        # --- Step 4c: Bayesian smoothed rating ---------------------------------
        rating = _bayesian_rating(float(row["avg_rating"]), float(row["n_ratings"]),
                                  mu_global)

        # --- Step 4d: composite score ------------------------------------------
        # Absolute-overlap bonus saturates in |inter| (the count of matched
        # ingredients), so a 3/3 tiny recipe no longer beats a substantive 6/7.
        coverage = float(row["coverage"])
        overlap_bonus = 1.0 - np.exp(-float(row["_inter"]) / SCORE_TAU)
        score = (coverage ** SCORE_ALPHA) * overlap_bonus * \
                (alignment ** SCORE_BETA) * (rating ** SCORE_GAMMA)

        # --- presentation -------------------------------------------------------
        missing = [w for w in recipe_ingr if w not in user_set]
        top2 = np.argsort(recipe_post.mean(axis=0))[::-1][:2]
        flavor_tags = [model.topic_labels[k] for k in top2]

        results.append({
            "recipe_name": row.get("recipe_name", row.get("name", "")),
            "score": round(float(score), 4),
            "coverage": f"{int(row['_inter'])}/{int(row['_size'])} ingredients",
            "missing_ingredients": missing,
            "flavor_tags": flavor_tags,
            "predicted_rating": round(float(rating), 3),
            "posterior_uncertainty": round(float(uncertainty), 4),
            "_score": float(score),
            "_ingr_set": set(recipe_ingr),       # for MMR diversity re-ranking
        })

    results.sort(key=lambda d: d["_score"], reverse=True)
    if not return_internal:                       # default: clean public dicts
        for d in results:
            d.pop("_score", None)
            d.pop("_ingr_set", None)
    return results


# =========================================================================== #
#  STEP 5 helpers · preference filters + MMR diversity re-ranking               #
# =========================================================================== #
def _apply_pref_filters(candidates: pd.DataFrame, exclude=None, must_use=None,
                        diet: str | None = None) -> pd.DataFrame:
    """Drop candidate recipes that violate user preferences:
        exclude  : recipe must contain NONE of these ingredients
        must_use : recipe must contain ALL of these ingredients
        diet     : 'vegetarian'/'vegan' -> no blocked ingredient (DIET_BLOCKLIST)
    All matching is on normalized ingredient tokens, so it lines up with coverage.
    """
    if candidates.empty:
        return candidates
    excl = set(normalize_list(exclude or []))
    must = set(normalize_list(must_use or []))
    block = DIET_BLOCKLIST.get((diet or "").lower(), set())

    def ok(ingr_list) -> bool:
        s = set(ingr_list)
        if excl and (s & excl):
            return False
        if must and not must.issubset(s):
            return False
        if block and any(set(w.split()) & block for w in ingr_list):
            return False
        return True

    kept = candidates[candidates["_ingredients_norm"].map(ok)].reset_index(drop=True)
    if len(kept) != len(candidates):
        bits = []
        if diet:     bits.append(f"diet={diet}")
        if exclude:  bits.append(f"exclude={list(excl)}")
        if must_use: bits.append(f"must_use={list(must)}")
        print(f"[filters] {', '.join(bits)}: {len(candidates)} -> {len(kept)} recipes")
    return kept


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _mmr_select(results: list[dict], top_n: int, diversity: float) -> list[dict]:
    """Maximal-Marginal-Relevance re-ranking. Trades a recipe's score against its
    similarity (Jaccard on ingredient sets) to the already-picked recipes, so the
    Top-N are not near-duplicates. `diversity` in [0,1]: 0 = pure score (plain
    Top-N), higher = more variety. Each result must carry '_score' and '_ingr_set'.
    """
    if diversity <= 0 or len(results) <= 1:
        return results[:top_n]
    scores = np.array([r["_score"] for r in results], dtype=float)
    lo, hi = scores.min(), scores.max()
    rel = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)  # -> [0,1]
    sets = [r["_ingr_set"] for r in results]

    chosen = [int(np.argmax(rel))]
    remaining = [i for i in range(len(results)) if i != chosen[0]]
    while remaining and len(chosen) < top_n:
        best_i, best_mmr = remaining[0], -np.inf
        for i in remaining:
            sim = max(_jaccard(sets[i], sets[c]) for c in chosen)
            mmr = (1.0 - diversity) * rel[i] - diversity * sim
            if mmr > best_mmr:
                best_mmr, best_i = mmr, i
        chosen.append(best_i)
        remaining.remove(best_i)
    return [results[i] for i in chosen]


# =========================================================================== #
#  STEP 5 · recommend()  -- the public entry point                            #
# =========================================================================== #
# Module-level cache so repeated recommend() calls reuse the (expensive) fit.
_STATE: dict = {"model": None, "df_id": None}


def recommend(user_ingredients: list[str], df: pd.DataFrame,
              top_n: int = 5,
              exclude: list[str] | None = None,
              must_use: list[str] | None = None,
              diet: str | None = None,
              diversity: float = 0.0,
              retrain: bool = False, model_path: str = "models/lda_model.pkl",
              **train_kwargs) -> list[dict]:
    """Top-N recipes the user can make tonight.

    User-facing options
      top_n     : how many recommendations to return (default 5)
      exclude   : ingredients the recipe must NOT contain
      must_use  : ingredients the recipe MUST all contain
      diet      : 'vegetarian' / 'vegan' (heuristic, see DIET_BLOCKLIST)
      diversity : 0..1 -- MMR re-rank so the list isn't near-duplicate recipes
                  (0 = pure score; ~0.3-0.5 = noticeably more varied)

    Model resolution order (so we never retrain needlessly):
      1. in-memory cache (_STATE) for this df,
      2. a previously saved model at `model_path`,
      3. otherwise train the Bayesian LDA, then cache + save it.
    Pass retrain=True to force a fresh fit. `train_kwargs` go to train_lda.
    """
    import os
    global _STATE
    if not retrain and _STATE["model"] is not None and _STATE["df_id"] == id(df):
        model = _STATE["model"]
    elif not retrain and os.path.exists(model_path):
        model = load_model(model_path)
        _STATE.update(model=model, df_id=id(df))
    else:
        model = train_lda(df, **train_kwargs)
        save_model(model, model_path)
        _STATE.update(model=model, df_id=id(df))

    candidates = filter_candidates(user_ingredients, df)            # Step 2
    candidates = _apply_pref_filters(candidates, exclude=exclude,   # Step 2b: prefs
                                     must_use=must_use, diet=diet)
    if candidates.empty:
        return []
    _, user_post = infer_user_posterior(user_ingredients, model)    # Step 3
    ranked = score_recipes(user_ingredients, candidates, user_post, # Step 4
                           model, df, return_internal=True)
    ranked = _mmr_select(ranked, top_n=top_n, diversity=diversity)  # Step 5: Top-N / MMR
    for d in ranked:                                                # strip internals
        d.pop("_score", None)
        d.pop("_ingr_set", None)
    return ranked


# --------------------------------------------------------------------------- #
#  Demo when run directly                                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Bayesian LDA recipe recommender demo")
    ap.add_argument("--data", default="data/recipes_clean.csv",
                    help="cleaned recipe CSV (cols: recipe_id, recipe_name, "
                         "ingredients, avg_rating, n_ratings)")
    ap.add_argument("--ingredients", nargs="+",
                    default=["chicken", "garlic", "onion", "olive oil", "tomato",
                             "salt", "pepper", "rice"])
    ap.add_argument("--top-n", type=int, default=5, help="how many to return")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="ingredients the recipe must NOT contain")
    ap.add_argument("--must-use", nargs="*", default=None,
                    help="ingredients the recipe MUST all contain")
    ap.add_argument("--diet", choices=["vegetarian", "vegan"], default=None)
    ap.add_argument("--diversity", type=float, default=0.0,
                    help="0..1 MMR re-rank for a more varied list")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} recipes")
    recs = recommend(args.ingredients, df, top_n=args.top_n, exclude=args.exclude,
                     must_use=args.must_use, diet=args.diet, diversity=args.diversity)
    print(f"\n=== TOP-{args.top_n} RECOMMENDATIONS ===")
    print(json.dumps(recs, indent=2, ensure_ascii=False))
