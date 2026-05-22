"""
visualize.py
============
Bayesian visualizations for the LDA recipe recommender. Everything is built from
the *cached posterior* (lda_model.pkl) -- no retraining -- so it runs in seconds.

Figures (saved as PNGs):
  fig1_model_selection.png        : held-out predictive vs in-sample WAIC/LOO across K
                                    -> why K is chosen out-of-sample (in-sample overfits)
  fig2_topic_phi_posterior.png    : top ingredients per topic with 94% credible intervals
                                    -> "fully Bayesian": we keep distributions over phi
  fig3_user_topic_posterior.png   : P(topic | pantry) posterior over MCMC samples (Step 3)
                                    -> the flavor profile carries uncertainty
  fig4_recommendation_uncertainty.png : flavor-alignment posterior for the Top-5
                                    -> uncertainty propagated all the way to the ranking

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

DATA = "data/recipes_clean.csv"
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
    """Per-MCMC-sample flavor alignment exp(-KL(recipe || user))  -> shape (S,)."""
    kl = np.sum(recipe_post * (np.log(recipe_post + _EPS) - np.log(user_post + _EPS)),
                axis=1)
    return np.exp(-kl)


# --------------------------------------------------------------------------- #
def fig_model_selection(path="figures/fig1_model_selection.png"):
    if not os.path.exists("model_selection.json"):
        print("skip fig1: model_selection.json not found"); return
    t = pd.DataFrame(json.load(open("model_selection.json"))).sort_values("K")
    K = t["K"].to_numpy()
    k_oos = int(K[np.argmax(t["heldout_lppd"])])
    k_in = int(K[np.argmin(t["waic"])])

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    l1, = ax1.plot(K, t["heldout_lppd"], "o-", color="#2ca02c", lw=2.5,
                   label="held-out lppd  (out-of-sample, ↑ better)")
    ax1.axvline(k_oos, color="#2ca02c", ls=":", alpha=.6)
    ax1.set_xlabel("K  (number of flavor topics)")
    ax1.set_ylabel("held-out predictive lppd", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")

    ax2 = ax1.twinx()
    l2, = ax2.plot(K, t["waic"], "s--", color="#d62728",
                   label="WAIC  (in-sample, ↓ better)")
    l3, = ax2.plot(K, -2 * t["elpd_loo"], "^--", color="#ff7f0e",
                   label="PSIS-LOO deviance  (in-sample, ↓ better)")
    ax2.axvline(k_in, color="#d62728", ls=":", alpha=.6)
    ax2.set_ylabel("WAIC / LOO deviance (in-sample)")
    ax2.grid(False)

    ax1.set_title(f"Model selection: held-out peaks at K={k_oos}, but in-sample "
                  f"WAIC/LOO keep\n'improving' to K={k_in} (overfitting) "
                  f"-> we choose K out-of-sample", fontsize=11)
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8.5)
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
    fig.suptitle(f"Topic→ingredient posterior φ with {HDI}% credible intervals\n"
                 f"(fully Bayesian: posterior distributions, not point estimates)",
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
    axes[0].set_ylabel("posterior topic weight (mean ± 94% CI)")
    fig.suptitle("Step 3 — Bayesian posterior over flavor topics for a pantry\n"
                 "(error bar = 94% credible interval; Italian collapses onto one "
                 "topic, Baker's stays uncertain across two)", fontsize=11)
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
    ax.set_xlabel("flavor alignment  exp(−KL(recipe ‖ user))  — posterior over MCMC samples")
    ax.set_title(f"Top-{top_n} for '{name}': flavor-alignment uncertainty\n"
                 f"propagated from the φ posterior into the ranking", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); print("wrote", path)


def main():
    fig_model_selection()
    fig_topic_phi()
    fig_user_posterior()
    fig_reco_uncertainty()
    print("\nAll figures written.")


if __name__ == "__main__":
    main()
