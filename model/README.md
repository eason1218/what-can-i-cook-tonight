**English** | [中文](README.zh-CN.md)

# What Can I Cook Tonight? — Bayesian LDA Recipe Recommender

A **fully Bayesian** Latent Dirichlet Allocation (LDA) recommender built on **PyMC**.
Given the ingredients you have on hand, it returns the Top-5 recipes you can make,
ranked by a composite score that fuses *coverage*, *latent-flavor alignment*, and a
*Bayesian-smoothed rating* — with uncertainty propagated from the MCMC posterior all
the way to the final ranking.

## Pipeline at a glance

![Pipeline flowchart](figures/pipeline_flowchart.png)

*Editable Mermaid source in [`docs/flowchart.md`](docs/flowchart.md); regenerate the PNG with `python src/make_flowchart.py`.*

## Project structure

```
model/
├── README.md · README.zh-CN.md · requirements.txt
├── src/
│   ├── recipe_recommender.py    # model + the 5 functions: train_lda, filter_candidates,
│   │                            #   infer_user_posterior, score_recipes, recommend
│   ├── prepare_data.py          # download Food.com  ->  data/recipes_clean.csv
│   ├── run_demo.py              # end-to-end demo    ->  results.json (+ trains the model)
│   ├── visualize.py             # Bayesian figures   ->  figures/
│   └── make_flowchart.py        # pipeline diagram   ->  figures/pipeline_flowchart.png
├── notebooks/
│   ├── model_pipeline.ipynb     # end-to-end construction walkthrough (executed)
│   ├── usage_example.ipynb      # a single recommendation use case (executed)
│   └── Data.ipynb               # original data-engineering notebook (spaCy)
├── figures/                     # fig1–fig4 + pipeline_flowchart.png
├── docs/                        # presentation_script.md · flowchart.md
├── data/                        # recipes_clean.csv  (generated; gitignored)
├── models/                      # lda_model.pkl      (trained;   gitignored)
├── model_selection.json         # K-sweep data behind fig1
└── results.json                 # last demo run
```

The five required functions live in `src/recipe_recommender.py`: `train_lda`,
`filter_candidates`, `infer_user_posterior`, `score_recipes`, `recommend`.

## Quick start

```bash
pip install pymc arviz nutpie kagglehub inflect numpy pandas scipy
python src/prepare_data.py      # -> data/recipes_clean.csv
python src/run_demo.py          # train + recommend
# or, as a library:
python src/recipe_recommender.py --ingredients chicken garlic onion tomato rice salt
```

```python
import sys; sys.path.insert(0, "src")          # run from the model/ folder
import pandas as pd, recipe_recommender as rr
df = pd.read_csv("data/recipes_clean.csv")
rr.recommend(["chicken", "garlic", "onion", "tomato", "rice", "salt"], df)
```

> **Note.** `data/recipes_clean.csv` and `models/lda_model.pkl` are **gitignored** (large /
> regenerable). Run `python src/prepare_data.py` then `python src/run_demo.py` once to generate
> them — the notebooks load `models/lda_model.pkl`. The committed notebooks already have their
> outputs and figures embedded, so they render on GitHub without running anything.

## The five steps

### Step 1 — `train_lda()` : Bayesian LDA with NUTS, K by held-out predictive likelihood
Generative model:

```
phi_k   ~ Dirichlet(beta=0.01)       # sparse ingredient distribution per topic
theta_m ~ Dirichlet(alpha=0.1)       # sparse topic mix per recipe
w_mn    ~ Categorical(theta_m @ phi)  # observed ingredient
```

**Why `theta_m @ phi` instead of explicit `z_mn`?** The classic story samples a
discrete topic `z_mn ~ Categorical(theta_m)` then `w_mn ~ Categorical(phi_z)`. NUTS
is gradient-based and cannot traverse discrete latents, so we **marginalize z
analytically** — `p(w | theta, phi) = Σ_k theta_k phi_k = (theta @ phi)`. This is the
*same* model with z integrated out, leaving only the continuous simplex variables for
NUTS. We keep the **full posterior** of `phi` and `theta` (samples, not point
estimates).

**K selection by held-out predictive likelihood.** WAIC and PSIS-LOO are *in-sample*
criteria, and for this flexible marginalized LDA they keep improving as `K` grows (more
topics simply fit the training tokens better) — there is no clean interior optimum, and
a large `p_waic`/`p_loo` or many Pareto-`k` above arviz's `good_k` just signals the
complexity penalty has stopped being trustworthy. So we select `K` **out-of-sample**:
15% of ingredient tokens are held out, every `K` is fit on the rest, and the winner
maximizes the held-out predictive density `Σ_n log mean_s p(w_n | θ, φ)`. Topics that
merely memorize the training tokens don't help predict held-out ones, so this criterion
peaks at a finite `K`. WAIC and PSIS-LOO (with the Pareto-`k` reliability check) are
still computed on the training tokens and **reported as diagnostics**. The winning `K`
is then **refit on the full corpus**. All three criteria depend only on the
label-invariant likelihood `theta @ phi`, so they pool all chains safely. *(arviz 1.x
dropped the standalone `waic()`; WAIC is computed from the posterior samples directly,
PSIS-LOO via `az.loo`.)*

### Step 2 — `filter_candidates()`
`coverage = |user ∩ recipe| / |recipe|`. Keep `coverage ≥ 0.7`; if fewer than 20
recipes survive, relax to `0.5`. Both sides are run through the same ingredient
canonicalizer so the set intersection is meaningful.

### Step 3 — `infer_user_posterior()` : posterior over flavor topics
Treat the user's ingredients as evidence and apply Bayes' theorem **per posterior
sample of phi**:

```
P(topic=k | ingredients) ∝ P(ingredients | topic=k) · P(topic=k)
P(ingredients | topic=k)  = Π_i phi[k, i]
P(topic=k)                = empirical marginal topic frequency (mean recipe mix)
```

We evaluate this for every MCMC sample and keep the per-sample posteriors, so the
"flavor profile" carries the model's uncertainty rather than collapsing to a point
estimate. The *same* function profiles each recipe in Step 4.

### Step 4 — `score_recipes()` : composite Bayesian score
```
score = coverage^2 · overlap_bonus · flavor_alignment^1 · rating_prior^1
```
- **coverage^2** — dominant term; a recipe you cannot make is useless.
- **overlap_bonus** = `1 − exp(−|user ∩ recipe| / τ)`, `τ=4`. coverage is a *ratio*, so
  a 2–3 ingredient recipe trivially scores 1.0; this saturating bonus in the *absolute*
  number of matched ingredients demotes those trivial matches and rewards substantively
  bigger ones (a real 6/7 over a token 3/3).
- **flavor_alignment** = `exp(−KL(recipe_topics ‖ user_topics))`, computed per MCMC
  sample (recipe & user posteriors paired by draw) then averaged. Its per-sample
  **std** is reported as `posterior_uncertainty`.
- **rating_prior** — Beta-Binomial / shrinkage smoothing
  `(avg·n + μ_global·κ) / (n + κ)`, `κ=5`; recipes with few ratings shrink toward the
  global mean.

### Step 5 — `recommend()`
Returns the Top-N as dicts:
```python
{
  "recipe_name": str,
  "score": float,
  "coverage": "5/6 ingredients",
  "missing_ingredients": [...],
  "flavor_tags": [...],            # top-2 topic labels from the posterior
  "predicted_rating": float,
  "posterior_uncertainty": float, # std of flavor alignment across MCMC samples
}
```

**User-facing options.** `recommend(pantry, df, **opts)` accepts:

| option | default | effect |
|--------|---------|--------|
| `top_n` | `5` | how many recipes to return |
| `exclude` | `None` | recipe must contain **none** of these ingredients |
| `must_use` | `None` | recipe must contain **all** of these ingredients |
| `diet` | `None` | `"vegetarian"` / `"vegan"` — drop recipes with a blocked ingredient (`DIET_BLOCKLIST`, editable). Vegetarian is reliable; vegan is *conservative* (it also catches e.g. "coconut milk" via the word "milk"). |
| `diversity` | `0.0` | `0..1` MMR re-rank: trade score against Jaccard similarity to already-picked recipes so the list isn't near-duplicates. `0`=pure score, `~0.4`=noticeably varied. |

```python
rr.recommend(pantry, df, diet="vegetarian", must_use=["tomato"], top_n=8)
rr.recommend(pantry, df, diversity=0.5)          # spread the Top-5 across recipe types
# CLI: python src/recipe_recommender.py --ingredients ... --diet vegan --diversity 0.5 --top-n 8
```
Matching is on the same normalized ingredient tokens as coverage, so it stays consistent.

## Visualizations (`python visualize.py`)

Built from the cached posterior (`models/lda_model.pkl`) — no retraining, runs in seconds.

**Model selection — why K is chosen out-of-sample**

![Model selection: held-out predictive vs in-sample WAIC/LOO across K](figures/fig1_model_selection.png)

*Held-out predictive lppd (green) peaks at a small K, while in-sample WAIC/LOO (right axis)
keep "improving" with K — the classic overfitting signature. So K is chosen out-of-sample.*

**Topic→ingredient posterior φ (94% credible intervals)**

![Per-topic top ingredients with 94% credible intervals on phi](figures/fig2_topic_phi_posterior.png)

*The "fully Bayesian" view: we keep posterior distributions over φ, not point estimates.*

**Step 3 — the user's flavor posterior**

![P(topic | pantry) with 94% credible intervals](figures/fig3_user_topic_posterior.png)

*`P(topic | pantry)` (mean ± 94% CI): the Italian pantry collapses onto one topic (certain),
the baker's pantry splits across two (genuinely uncertain) — same machinery, different certainty.*

**Uncertainty propagated into the ranking**

![Top-5 flavor-alignment posterior](figures/fig4_recommendation_uncertainty.png)

*Per-MCMC-sample flavor alignment for the Top-5 — the φ posterior's uncertainty reaches the
final score.*

## Design decisions & honest caveats

- **Scale.** Fully Bayesian NUTS does not scale to ~50k recipes × thousands of
  ingredients. `train_lda` fits `phi` on a manageable **subsample of recipes** and a
  **top-N ingredient vocabulary**; per-recipe and per-user topic profiles for the
  *whole* catalogue are then obtained from the `phi` posterior via the Step-3 Bayes
  update (fast, and consistent with the user computation). Increase `n_train`,
  `vocab_size`, `draws` for higher fidelity at the cost of runtime.
- **Number of topics is corpus-bound.** Because `K` is chosen *out-of-sample*, it can
  only be as large as the training corpus actually supports — a small `n_train` will
  honestly select very few topics (coarse `flavor_tags`). For richer topics, raise
  `n_train`/`vocab_size` so the held-out criterion can justify a larger `K`; this
  trades runtime (NUTS scales with the `theta` dimension `M×K`).
- **Label switching.** Topics are exchangeable, so topic labels are not comparable
  across chains. The likelihood `theta @ phi` *is* label-invariant, so WAIC, PSIS-LOO
  and the held-out predictive pool all chains; everything topic-wise (the kept
  posterior, the topic KLs) **and the reported `ESS`** use a **single chain**, within
  which labels stay coherent. Cross-chain ESS/r-hat of `phi` would be deflated purely by
  label switching, so reporting it would understate convergence — hence per-chain.
- **Sampler.** Uses `nutpie` (numba backend) because no C/C++ compiler is available
  for PyTensor's default backend. First compile per model takes ~1–2 min.
- **Backend note.** `recommend()` lazily trains and caches the model on first call
  (keyed on the DataFrame); pass `retrain=True` or different `train_kwargs` to refit.
- **Performance.** Ingredient normalization is the hot path, so `normalize_token` is
  `lru_cache`d (≈474k calls collapse to ≈7k distinct strings) and the catalogue's
  normalized lists are cached per DataFrame. A `recommend()` call costs ~2 s cold and
  ~0.05 s warm (was ~38 s every call). Passing a single `k_values=(K,)` skips the
  held-out diagnostic fit and trains with one NUTS fit. The NUTS fit itself is the only
  inherent cost; it scales super-linearly in the number of training *documents*
  (`n_train`), so keep `n_train` modest (~400) and widen `vocab_size` (cheap) instead.
