"""
recipe_recommender.py
=====================
A *fully Bayesian* LDA recipe recommender built on PyMC.

Pipeline ("What Can I Cook Tonight?")
-------------------------------------
    Step 1  train_lda()             -> Bayesian LDA over latent "flavor topics" (NUTS / MCMC)
    Step 2  filter_candidates()     -> keep recipes the user can actually make (coverage)
    Step 3  infer_user_posterior()  -> Bayesian posterior over flavor topics for an ingredient set
    Step 4  score_recipes()         -> composite Bayesian score using the *full* posterior
    Step 5  recommend()             -> Top-5 recipes as a list of dicts

Why this is "fully Bayesian"
----------------------------
We never collapse to a single point estimate of phi (topic->ingredient) or theta
(recipe->topic). NUTS gives us posterior *samples* of phi, and every downstream
quantity (the user's flavor profile, each recipe's flavor profile, the flavor
alignment similarity) is computed *per posterior sample and then averaged*. This
propagates the model's epistemic uncertainty all the way to the final score, and
lets us report a `posterior_uncertainty` for every recommendation.

A note on the generative model & NUTS
-------------------------------------
The classic LDA generative story has a *discrete* per-token topic assignment
    z_mn ~ Categorical(theta_m).
NUTS is a gradient-based sampler and cannot move over discrete latent variables.
The standard, mathematically exact remedy is to **marginalize z out**:

    p(w_mn = v | theta_m, phi) = sum_k theta_m[k] * phi[k, v]   = (theta_m @ phi)[v]

i.e. each ingredient token is Categorical with probability vector  p_m = theta_m @ phi.
This is the same model -- z is simply integrated away analytically -- and it leaves
only the continuous simplex variables phi and theta for NUTS to explore. The full
posterior over phi and theta is preserved.

A note on label switching (non-identifiability)
-----------------------------------------------
Topics are exchangeable: permuting the K topic labels leaves the likelihood
p_m = theta_m @ phi unchanged. Therefore:
  * WAIC (which depends only on the likelihood) is invariant to label switching,
    so we can compare K across *all* chains safely.
  * Topic-wise quantities (per-topic ingredient distributions, KL between topic
    profiles) are NOT label-invariant across chains. So for everything downstream
    of model selection we keep the posterior of a **single chain**, within which
    the topic labels stay coherent.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.special import softmax

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
    """Everything downstream steps need from the fitted Bayesian LDA.

    phi_samples : np.ndarray, shape (S, K, V)
        Posterior SAMPLES of the topic->ingredient distributions (single chain,
        so topic labels are coherent across samples). The whole point of keeping
        samples (not a mean) is to propagate uncertainty into the recommendation.
    topic_prior : np.ndarray, shape (K,)
        Empirical marginal topic frequency  P(topic)  = average recipe topic mix,
        itself averaged over posterior samples (a fully Bayesian marginal prior).
    vocab / ingr2idx : the modeled ingredient space (the LDA "words").
    topic_labels : human-readable tag per topic (its top ingredients).
    waic_table : DataFrame comparing candidate K -- held-out predictive lppd
        (the selection metric, higher=better) plus WAIC and PSIS-LOO diagnostics.
    """
    best_k: int
    phi_samples: np.ndarray          # (S, K, V)
    topic_prior: np.ndarray          # (K,)
    vocab: list[str]
    ingr2idx: dict[str, int]
    topic_labels: list[str]
    waic_table: pd.DataFrame
    idata: az.InferenceData = field(repr=False, default=None)
    ess_min: float = None            # within-chain ESS of phi (convergence diag)
    ess_median: float = None

    @property
    def n_samples(self) -> int:
        return self.phi_samples.shape[0]


def save_model(model: LDAModel, path: str = "models/lda_model.pkl") -> None:
    """Persist the fitted posterior (drops the heavy InferenceData)."""
    import pickle
    payload = {
        "best_k": model.best_k,
        "phi_samples": model.phi_samples,
        "topic_prior": model.topic_prior,
        "vocab": model.vocab,
        "ingr2idx": model.ingr2idx,
        "topic_labels": model.topic_labels,
        "waic_table": model.waic_table,
        "ess_min": model.ess_min,
        "ess_median": model.ess_median,
    }
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_model(path: str = "models/lda_model.pkl") -> LDAModel:
    """Load a model saved by save_model (so recommend() needn't retrain)."""
    import pickle
    with open(path, "rb") as f:
        p = pickle.load(f)
    return LDAModel(idata=None, **p)


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


def _build_corpus(df: pd.DataFrame, n_train: int, vocab_size: int,
                  min_df: int, seed: int):
    """Subsample recipes and build the bag-of-words tensors for LDA.

    Returns
    -------
    doc_idx, word_idx : 1-D int arrays (one entry per ingredient *token*)
    vocab, ingr2idx   : the modeled ingredient space
    M                 : number of training documents (recipes)
    """
    rng = np.random.default_rng(seed)

    ingredient_lists = _get_recipe_ingredients(df)

    # ---- subsample recipes (fully-Bayesian NUTS does not scale to ~200k docs) --
    n_train = min(n_train, len(df))
    sample_pos = rng.choice(len(df), size=n_train, replace=False)
    sample_lists = [ingredient_lists.iloc[i] for i in sample_pos]

    # ---- vocabulary = most frequent ingredients (by document frequency) --------
    from collections import Counter
    df_count = Counter()
    for lst in sample_lists:
        df_count.update(set(lst))
    vocab = [w for w, c in df_count.most_common() if c >= min_df][:vocab_size]
    ingr2idx = {w: i for i, w in enumerate(vocab)}

    # ---- encode each recipe as a list of in-vocab word indices -----------------
    doc_idx, word_idx = [], []
    m = 0
    for lst in sample_lists:
        toks = [ingr2idx[w] for w in lst if w in ingr2idx]
        if len(toks) < 2:           # need >=2 tokens for a meaningful topic mix
            continue
        doc_idx.extend([m] * len(toks))
        word_idx.extend(toks)
        m += 1

    return (np.asarray(doc_idx, dtype="int64"),
            np.asarray(word_idx, dtype="int64"),
            vocab, ingr2idx, m)


# =========================================================================== #
#  Manual WAIC (arviz 1.x dropped the standalone waic())                       #
# =========================================================================== #
def _pointwise_loglik(theta_s: np.ndarray, phi_s: np.ndarray,
                      doc_idx: np.ndarray, word_idx: np.ndarray) -> np.ndarray:
    """Per-token, per-sample log-likelihood  log p(w_n | theta, phi).

    theta_s : (S, M, K)   phi_s : (S, K, V)
    returns : (S, N_tokens)  where  loglik[s, n] = log( (theta_{m_n} @ phi)[w_n] )
    """
    # p_all[s, m, v] = sum_k theta_s[s,m,k] * phi_s[s,k,v]   (marginalize z)
    p_all = np.einsum("smk,skv->smv", theta_s, phi_s)
    p_tok = p_all[:, doc_idx, word_idx]            # (S, N_tokens)
    return np.log(p_tok + _EPS)


def _waic(loglik: np.ndarray) -> dict:
    """Widely Applicable Information Criterion from pointwise log-likelihood.

    loglik : (S, N) samples x observations.
        lppd   = sum_n log mean_s exp(loglik[s,n])      (log pointwise predictive density)
        p_waic = sum_n var_s(loglik[s,n])               (effective # parameters)
        elpd   = lppd - p_waic                          (expected log predictive density)
        WAIC   = -2 * elpd                              (deviance scale; lower = better)
    """
    S = loglik.shape[0]
    from scipy.special import logsumexp
    lppd = np.sum(logsumexp(loglik, axis=0) - np.log(S))
    p_waic = np.sum(np.var(loglik, axis=0, ddof=1))
    elpd = lppd - p_waic
    return {"waic": -2.0 * elpd, "elpd_waic": elpd, "p_waic": p_waic, "lppd": lppd}


# =========================================================================== #
#  Out-of-sample model selection (held-out token predictive) + PSIS-LOO check  #
# =========================================================================== #
# WHY held-out tokens?  WAIC and LOO are *in-sample*: they reuse the training
# tokens, so a more flexible model (larger K) can keep lowering them by fitting
# the training corpus better -- there is no clean interior optimum, and a large
# p_waic / p_loo just flags that the penalty has stopped being trustworthy.
# Holding tokens out turns model selection into genuine prediction: extra topics
# that merely memorize the training tokens do NOT help predict the held-out ones,
# so the held-out predictive density *does* peak at a finite K.
def _holdout_split(doc_idx: np.ndarray, word_idx: np.ndarray, frac: float = 0.15,
                   seed: int = 0, min_train_per_doc: int = 2):
    """Randomly mark a fraction of tokens as held-out, while leaving at least
    `min_train_per_doc` tokens in every document (so each document's theta stays
    identified during the fit). Returns (train_mask, test_mask)."""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(doc_idx), dtype=bool)
    for d in np.unique(doc_idx):
        pos = np.where(doc_idx == d)[0]
        n_test = min(int(round(frac * len(pos))), max(0, len(pos) - min_train_per_doc))
        if n_test > 0:
            is_test[rng.choice(pos, size=n_test, replace=False)] = True
    return ~is_test, is_test


def _heldout_lppd(theta_s: np.ndarray, phi_s: np.ndarray,
                  doc_te: np.ndarray, word_te: np.ndarray) -> float:
    """Log pointwise predictive density on held-out tokens (higher = better).
    theta_s comes from the fit on the *training* tokens; we score the held-out
    tokens of each document under that document's posterior theta and phi."""
    if len(doc_te) == 0:
        return float("nan")
    from scipy.special import logsumexp
    ll = _pointwise_loglik(theta_s, phi_s, doc_te, word_te)      # (S, N_te)
    S = ll.shape[0]
    return float(np.sum(logsumexp(ll, axis=0) - np.log(S)))


def _loo_from_loglik(loglik_cd: np.ndarray) -> dict:
    """PSIS-LOO (elpd_loo, p_loo) + Pareto-k diagnostic from a (chain, draw, obs)
    pointwise log-likelihood. Reported -- NOT used for selection -- as an honest
    reliability check: many Pareto-k above arviz's `good_k` threshold (or a large
    p_loo) means the in-sample criteria cannot be trusted, which is exactly the
    pathology a flexible LDA exhibits. Best-effort: NaNs if arviz can't compute it.

    arviz needs a posterior group to derive the relative ESS (reff). The
    log-likelihood theta@phi is label-invariant, so we hand it in as both the
    `posterior` and `log_likelihood` group and let arviz compute reff itself."""
    try:
        idll = az.from_dict(
            {"posterior": {"w": loglik_cd}, "log_likelihood": {"w": loglik_cd}},
            dims={"w": ["obs"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = az.loo(idll, pointwise=True, var_name="w")
        pk = np.asarray(res.pareto_k.values)
        good_k = float(getattr(res, "good_k", 0.7) or 0.7)
        return {"elpd_loo": float(res.elpd), "p_loo": float(res.p),
                "pct_bad_k": float(np.mean(pk > good_k))}
    except Exception:
        return {"elpd_loo": float("nan"), "p_loo": float("nan"),
                "pct_bad_k": float("nan")}


# =========================================================================== #
#  STEP 1 · Bayesian LDA (PyMC + NUTS); K by held-out predictive log-likelihood #
# =========================================================================== #
def _fit_one_k(job: tuple) -> dict:
    """Fit the marginalized LDA for a single K on the TRAIN tokens and return the
    selection metrics (held-out predictive lppd; WAIC & PSIS-LOO as diagnostics)
    plus the single-chain posterior. Module-level (picklable) for worker processes.

    Generative model (z marginalized so NUTS sees only continuous simplices):
        phi_k   ~ Dirichlet(beta)
        theta_m ~ Dirichlet(alpha)
        w_mn    ~ Categorical(theta_m @ phi)

    Pass empty test arrays to fit on the full corpus (used for the final refit).
    """
    (K, doc_tr, word_tr, doc_te, word_te, V, M, draws, tune, chains, seed,
     nuts_sampler, progressbar) = job

    with pm.Model() as m:
        phi = pm.Dirichlet("phi", a=BETA_PRIOR * np.ones(V), shape=(K, V))
        theta = pm.Dirichlet("theta", a=ALPHA_PRIOR * np.ones(K), shape=(M, K))
        p = pm.math.dot(theta, phi)                              # (M, V)
        pm.Categorical("w", p=p[doc_tr], observed=word_tr)       # fit on TRAIN tokens

        common = dict(draws=draws, tune=tune, chains=chains,
                      random_seed=seed + K, progressbar=progressbar)
        if nuts_sampler in ("numpyro", "blackjax"):
            # JAX backend: run all chains *vectorized* (vmap) in one JIT kernel,
            # which on a single CPU device makes extra chains nearly free.
            # We call the jax sampler directly because pm.sample funnels
            # nuts_sampler_kwargs into the NUTS *kernel*, not chain_method.
            from pymc.sampling import jax as _pmjax
            sampler_fn = (_pmjax.sample_numpyro_nuts if nuts_sampler == "numpyro"
                          else _pmjax.sample_blackjax_nuts)
            idata = sampler_fn(model=m, chain_method="vectorized",
                               compute_convergence_checks=False, **common)
        else:                                                    # e.g. "nutpie"
            idata = pm.sample(nuts_sampler=nuts_sampler, cores=1,
                              compute_convergence_checks=False, **common)

    th = idata.posterior["theta"].values                        # (C, D, M, K)
    ph = idata.posterior["phi"].values                          # (C, D, K, V)
    C, D = th.shape[:2]
    theta_s = th.reshape(C * D, M, K)                           # pool chains (S=C*D)
    phi_s = ph.reshape(C * D, K, V)

    # ---- in-sample diagnostics on TRAIN tokens --------------------------------
    # label-invariant (depend only on theta@phi), so pooling chains is safe.
    loglik_tr = _pointwise_loglik(theta_s, phi_s, doc_tr, word_tr)   # (S, N_tr)
    waic = _waic(loglik_tr)
    loo = _loo_from_loglik(loglik_tr.reshape(C, D, -1))

    # ---- PRIMARY selection metric: out-of-sample predictive on HELD-OUT tokens -
    heldout_lppd = _heldout_lppd(theta_s, phi_s, doc_te, word_te)

    # ---- within-chain ESS of the chain we KEEP (chain 0). Cross-chain ESS is
    #      meaningless under label switching, so we slice to one chain first. ----
    ess = az.ess(idata.posterior.isel(chain=[0]), var_names=["phi"])["phi"]

    return {
        "K": K, "waic": waic, "loo": loo, "heldout_lppd": heldout_lppd,
        "phi_c0": ph[0],                                         # (S, K, V) chain 0
        "theta_c0": th[0],                                       # (S, M, K) chain 0
        "ess_min": float(ess.min()), "ess_med": float(ess.median()),
    }


def train_lda(df: pd.DataFrame,
              k_values: Sequence[int] = (2, 4, 6, 8, 10, 12),
              n_train: int = 400,
              vocab_size: int = 60,
              min_df: int = 5,
              draws: int = 300,
              tune: int = 400,
              chains: int = 2,
              seed: int = 42,
              nuts_sampler: str = "nutpie",
              parallel: bool = True,
              progressbar: bool = False,
              holdout_frac: float = 0.15) -> LDAModel:
    """Fit Bayesian LDA for several K, pick K by *held-out* predictive
    log-likelihood, then refit the winning K on the full corpus and return its
    posterior.

    Generative model (z marginalized for NUTS):
        phi_k ~ Dirichlet(beta)            # ingredient distribution for topic k
        theta_m ~ Dirichlet(alpha)         # topic distribution for recipe m
        w_mn ~ Categorical(theta_m @ phi)  # observed ingredient (z integrated out)

    Model selection is out-of-sample: a fraction of tokens is held out, every K is
    fit on the rest, and the winner maximizes the held-out predictive density.
    WAIC and PSIS-LOO are still computed (on the training tokens) and reported as
    diagnostics, but they are in-sample and tend to keep improving with K, so they
    are NOT used to choose K. The winning K is then refit on the *full* corpus.
    """
    doc_idx, word_idx, vocab, ingr2idx, M = _build_corpus(
        df, n_train=n_train, vocab_size=vocab_size, min_df=min_df, seed=seed)
    V = len(vocab)
    print(f"[train_lda] corpus: {M} recipes, {len(word_idx)} ingredient tokens, "
          f"vocab V={V}")

    empty = np.empty(0, dtype="int64")

    def _row(f: dict) -> dict:                 # one model-comparison table row
        return {"K": f["K"], "heldout_lppd": f["heldout_lppd"],
                "waic": f["waic"]["waic"], "p_waic": f["waic"]["p_waic"],
                "elpd_loo": f["loo"]["elpd_loo"], "p_loo": f["loo"]["p_loo"],
                "pct_bad_k": f["loo"]["pct_bad_k"]}

    if len(k_values) == 1:
        # ---- Directly-chosen K: no selection, no held-out split, so we do a
        #      SINGLE fit on the full corpus and skip the redundant diagnostic fit.
        best_k = int(k_values[0])
        print(f"[train_lda] single K={best_k}: one fit on the full corpus "
              f"(sampler={nuts_sampler}) ...")
        final = _fit_one_k((best_k, doc_idx, word_idx, empty, empty, V, M, draws,
                            tune, chains, seed, nuts_sampler, progressbar))
        table = pd.DataFrame([_row(final)])    # held-out NaN; WAIC/LOO are in-sample
    else:
        # ---- Out-of-sample model selection over the K grid ----------------------
        # One shared train/held-out token split, reused for every K (fair).
        train_mask, test_mask = _holdout_split(doc_idx, word_idx, frac=holdout_frac,
                                               seed=seed)
        doc_tr, word_tr = doc_idx[train_mask], word_idx[train_mask]
        doc_te, word_te = doc_idx[test_mask], word_idx[test_mask]
        print(f"[train_lda] holdout: {len(word_tr)} train / {len(word_te)} "
              f"held-out tokens")

        jobs = [(K, doc_tr, word_tr, doc_te, word_te, V, M, draws, tune, chains,
                 seed, nuts_sampler, progressbar) for K in k_values]
        if parallel and len(jobs) > 1:
            import os
            from concurrent.futures import ProcessPoolExecutor
            max_workers = min(len(jobs), max(1, (os.cpu_count() or 2) // 2))
            print(f"[train_lda] fitting K={list(k_values)} in parallel "
                  f"({max_workers} workers, sampler={nuts_sampler}) ...")
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                fitted = list(ex.map(_fit_one_k, jobs))
        else:
            print(f"[train_lda] fitting K={list(k_values)} sequentially "
                  f"(sampler={nuts_sampler}) ...")
            fitted = [_fit_one_k(j) for j in jobs]

        table = pd.DataFrame([_row(f) for f in fitted])
        if table["heldout_lppd"].notna().any():
            table = table.sort_values("heldout_lppd",
                                      ascending=False).reset_index(drop=True)
            sel_by = "held-out predictive lppd"
        else:                                  # no tokens held out -> fall back
            table = table.sort_values("waic").reset_index(drop=True)
            sel_by = "WAIC"
        best_k = int(table.iloc[0]["K"])
        print(f"[train_lda] selected K={best_k} by {sel_by}")

        # ---- Refit the winning K on the FULL corpus (CV picks K; refit all data)
        print(f"[train_lda] refitting K={best_k} on the full corpus ...")
        final = _fit_one_k((best_k, doc_idx, word_idx, empty, empty, V, M, draws,
                            tune, chains, seed, nuts_sampler, progressbar))

    # ---- Assemble the winning model's posterior (single chain => coherent topic
    #      labels; label switching makes cross-chain topics incomparable). ------
    phi_chain0 = final["phi_c0"]                                 # (S, K, V)
    theta_chain0 = final["theta_c0"]                            # (S, M, K)

    # ---- Empirical marginal topic prior  P(topic) -----------------------------
    # Marginal probability a random ingredient-slot belongs to topic k, i.e. the
    # average recipe topic mix, averaged over posterior samples. Fully Bayesian.
    topic_prior = theta_chain0.mean(axis=(0, 1))                 # (K,)
    topic_prior = topic_prior / topic_prior.sum()

    topic_labels = _label_topics(phi_chain0.mean(axis=0), vocab)

    return LDAModel(best_k=best_k, phi_samples=phi_chain0, topic_prior=topic_prior,
                    vocab=vocab, ingr2idx=ingr2idx, topic_labels=topic_labels,
                    waic_table=table, idata=None,
                    ess_min=final["ess_min"], ess_median=final["ess_med"])


def _label_topics(phi_mean: np.ndarray, vocab: list[str], top_n: int = 3) -> list[str]:
    """Human-readable label for each topic = its top ingredients (by posterior
    mean phi). Used for `flavor_tags`."""
    labels = []
    for k in range(phi_mean.shape[0]):
        top = np.argsort(phi_mean[k])[::-1][:top_n]
        labels.append(" / ".join(vocab[i] for i in top))
    return labels


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

    We do this for every MCMC sample s of phi, giving a posterior over the topic
    weights that *carries the model's uncertainty* (rather than using a single
    point estimate of phi).

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
    """Flavor alignment = exp(-KL(recipe || user)), averaged over MCMC samples.

    recipe_post, user_post : (S, K)  -- paired sample-by-sample (same phi draw s),
    so the KL is evaluated within a single coherent posterior draw and then
    averaged. We also return the std of the per-sample similarity, which becomes
    the recipe's `posterior_uncertainty`.
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
