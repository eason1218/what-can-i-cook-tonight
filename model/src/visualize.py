"""
visualize.py
============
Visualizations for the hybrid LDA recipe recommender. Everything is built from the
*cached model* (models/lda_model.pkl) -- no retraining -- so it runs in seconds.

Figures (saved as PNGs):
  figures/fig1_model_selection.png        : held-out perplexity across K (model selection)
                                    -> how K is chosen on the full corpus
  figures/fig2_topic_phi_posterior.png    : top ingredients per topic with 94% Bootstrap intervals
                                    -> phi point estimate + Bootstrap "stability" band (small)
  figures/fig3_user_topic_posterior.png   : P(topic | pantry) -- the user's flavor posterior (Step 3)
                                    -> the genuinely Bayesian step (Bayes per phi-sample)
  figures/fig4_recommendation_uncertainty.png : flavor-alignment spread for the Top-5
                                    -> uncertainty propagated to the ranking (Bootstrap draws)

Usage:  python visualize.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless: write files, no GUI
import matplotlib.pyplot as plt
import seaborn as sns

import recipe_recommender as rr

sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)

DATA = "../data/recipes_clean.csv"        # Stage-1 data lives in the top-level data/ dir
MODEL = "models/lda_model.pkl"
os.makedirs("figures", exist_ok=True)
HDI = 94                                     # credible-interval width (%)
_LO, _HI = (100 - HDI) / 2, 100 - (100 - HDI) / 2
_EPS = 1e-12

ITAL = ["chicken", "garlic", "onion", "olive oil", "tomato", "basil",
        "parmesan cheese", "pasta", "salt", "pepper"]
BAKE = ["flour", "sugar", "butter", "egg", "vanilla", "baking soda",
        "milk", "salt", "chocolate"]


def _sim_samples(recipe_post: np.ndarray, user_post: np.ndarray) -> np.ndarray:
    """Per-Bootstrap-sample flavor alignment exp(-KL(recipe || user))  -> shape (S,)."""
    kl = np.sum(recipe_post * (np.log(recipe_post + _EPS) - np.log(user_post + _EPS)),
                axis=1)
    return np.exp(-kl)


# --------------------------------------------------------------------------- #
def fig_model_selection(path="figures/fig1_model_selection.png"):
    """Held-out perplexity across K (lower = better). Falls back to the model's
    perplexity_table if model_selection.json is absent."""
    t = None
    if os.path.exists("model_selection.json"):
        t = pd.DataFrame(json.load(open("model_selection.json")))
    if t is None or t.empty or "holdout_perplexity" not in t:
        m = rr.load_model(MODEL)
        t = m.perplexity_table
    if t is None or t.empty:
        print("skip fig1: no perplexity-by-K data (model trained with a fixed K)"); return
    t = t.sort_values("K")
    K = t["K"].to_numpy()
    best = int(K[np.argmin(t["holdout_perplexity"])])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K, t["holdout_perplexity"], "o-", color="#2a9d8f", lw=2.5)
    ax.axvline(best, color="grey", ls="--", alpha=.6, label=f"argmin = K {best}")
    ax.set_xlabel("K  (number of flavor topics)")
    ax.set_ylabel("held-out perplexity  (lower = better)")
    ax.set_title("Model selection on the full corpus (held-out perplexity)\n"
                 "monotone in K here -> the data favors few, coarse topics", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); print("wrote", path)


def fig_topic_phi(path="figures/fig2_topic_phi_posterior.png", top_n=8):
    m = rr.load_model(MODEL)
    phi = m.phi_samples                          # (S, K, V)
    mean = phi.mean(0)
    lo = np.percentile(phi, _LO, axis=0)
    hi = np.percentile(phi, _HI, axis=0)
    K = mean.shape[0]
    ncol = 3
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).flatten()
    for k in range(len(axes)):
        ax = axes[k]
        if k >= K:
            ax.axis("off"); continue
        top = np.argsort(mean[k])[::-1][:top_n]
        y = np.arange(len(top))[::-1]
        xerr = np.vstack([mean[k][top] - lo[k][top], hi[k][top] - mean[k][top]])
        ax.errorbar(mean[k][top], y, xerr=xerr, fmt="o", color=f"C{k % 10}",
                    capsize=3, lw=1.5, ms=5)
        ax.set_yticks(y); ax.set_yticklabels([m.vocab[i] for i in top], fontsize=8)
        ax.set_title(f"Topic {k}  (P={m.topic_prior[k]:.2f})", fontsize=10)
        ax.set_xlabel("φ = P(ingredient | topic)", fontsize=8)
        ax.set_xlim(left=0)
    fig.suptitle(f"Topic→ingredient φ (point estimate) with {HDI}% Bootstrap intervals\n"
                 f"(intervals are small: φ is well determined on the full corpus)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path, dpi=130)
    plt.close(fig); print("wrote", path)


def fig_user_posterior(path="figures/fig3_user_topic_posterior.png"):
    # Bars = posterior mean, error bars = 94% credible interval. (A violin can't show
    # a delta-posterior: some pantries collapse to one topic with zero variance.)
    m = rr.load_model(MODEL)
    pantries = {"Italian-ish": ITAL, "Baker's pantry": BAKE}
    fig, axes = plt.subplots(1, len(pantries), figsize=(7 * len(pantries), 4.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, pantry) in zip(axes, pantries.items()):
        _, post = rr.infer_user_posterior(pantry, m)   # (S, K)
        K = post.shape[1]
        mean = post.mean(0)
        lo = np.percentile(post, _LO, axis=0)
        hi = np.percentile(post, _HI, axis=0)
        x = np.arange(K)
        err = np.vstack([np.maximum(0.0, mean - lo), np.maximum(0.0, hi - mean)])
        ax.bar(x, mean, color=[f"C{k % 10}" for k in range(K)], alpha=0.8,
               yerr=err, capsize=4, error_kw=dict(lw=1.5))
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{k}\n{m.topic_labels[k].split(' / ')[0]}"
                            for k in range(K)], fontsize=7)
        ax.set_title(f"P(topic | '{name}')", fontsize=11)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("posterior topic weight (mean ± 94% across Bootstrap φ)")
    fig.suptitle("Step 3 — Bayesian posterior over flavor topics for a pantry\n"
                 "(bars = P(topic|pantry); Italian collapses onto one topic, Baker's "
                 "spreads across two — error bars small as φ is well determined)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(path, dpi=130)
    plt.close(fig); print("wrote", path)


def fig_reco_uncertainty(path="figures/fig4_recommendation_uncertainty.png",
                         pantry=ITAL, name="Italian-ish", top_n=5):
    df = pd.read_csv(DATA)
    m = rr.load_model(MODEL)
    rr._STATE = {"model": m, "df_id": id(df)}
    recs = rr.recommend(pantry, df, top_n=top_n)                  # ordered Top-N
    _, user_post = rr.infer_user_posterior(pantry, m)
    cand = rr.filter_candidates(pantry, df)
    by_name = {row["recipe_name"]: row["_ingredients_norm"]
               for _, row in cand.iterrows()}

    sims, names = [], []
    for r in recs:
        ingr = by_name.get(r["recipe_name"])
        if ingr is None:
            continue
        _, rp = rr.infer_user_posterior(ingr, m)
        sims.append(_sim_samples(rp, user_post))
        names.append(r["recipe_name"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    vp = ax.violinplot(sims, positions=range(len(sims)), vert=False,
                       showmeans=True, widths=0.85)
    for b in vp["bodies"]:
        b.set_alpha(0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("flavor alignment  exp(−KL(recipe ‖ user))  — over Bootstrap φ draws")
    ax.set_title(f"Top-{top_n} for '{name}': flavor-alignment spread\n"
                 f"propagated from the φ Bootstrap into the ranking", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); print("wrote", path)


def main():
    fig_model_selection()
    fig_topic_phi()
    fig_user_posterior()
    fig_reco_uncertainty()
    print("\nAll figures written.")


if __name__ == "__main__":
    main()
