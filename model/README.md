**English** | [中文](README.zh-CN.md)

# What Can I Cook Tonight? — Bayesian LDA Recipe Recommender

A Bayesian Latent Dirichlet Allocation (LDA) recommender over latent "flavor topics".
Given the ingredients you have on hand, it returns the Top-5 recipes you can make,
ranked by a composite score that fuses *coverage*, *latent-flavor alignment*, and a
*Bayesian-smoothed rating* — with uncertainty propagated all the way to the ranking.

**The model.** A fully-Bayesian NUTS fit of `φ` (topic→ingredient) does not scale past a few
hundred recipes, so `φ` is fit as a **point estimate** with `sklearn`'s
`LatentDirichletAllocation` on the **full 53k-recipe corpus**, and `φ`'s uncertainty is
approximated by **Bootstrap** resampling (refit on resampled recipes, kept as pseudo-samples).
Every downstream quantity is computed *per `φ`-sample and then averaged*, so uncertainty flows to
the final score; the genuinely-Bayesian step is the **user's flavor posterior** (Step 3), inferred
from a handful of ingredients via Bayes' theorem. Fast (~25 s for the full corpus) and Bayesian
where it matters.

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
│   ├── topic_alignment.py       # Hungarian topic-label alignment for the Bootstrap refits
│   ├── prepare_data.py          # download Food.com  ->  data/recipes_clean.csv
│   ├── run_demo.py              # end-to-end demo    ->  results.json (+ trains the model)
│   ├── visualize.py             # figures            ->  figures/
│   └── make_flowchart.py        # pipeline diagram   ->  figures/pipeline_flowchart.png
├── notebooks/
│   ├── model_pipeline.ipynb     # end-to-end construction walkthrough (executed)
│   ├── usage_example.ipynb      # a single recommendation use case (executed)
│   └── Data.ipynb               # original data-engineering notebook (spaCy)
├── tests/                       # test_alignment.py  (pytest)
├── figures/                     # fig1–fig4 + pipeline_flowchart.png
├── docs/                        # presentation_script.md · flowchart.md
├── data/                        # recipes_clean.csv  (generated; gitignored)
├── models/                      # lda_model.pkl      (trained;   gitignored)
├── model_selection.json         # held-out perplexity per K (behind fig1)
└── results.json                 # last demo run
```

The five required functions live in `src/recipe_recommender.py`: `train_lda`,
`filter_candidates`, `infer_user_posterior`, `score_recipes`, `recommend`.

## Quick start

```bash
pip install scikit-learn joblib kagglehub inflect numpy pandas scipy matplotlib seaborn
python src/prepare_data.py      # -> data/recipes_clean.csv
python src/run_demo.py          # train (full corpus, ~25–60 s) + recommend
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

### Step 1 — `train_lda()` : sklearn point estimate of φ + Bootstrap pseudo-posterior

`train_lda(df, ...) -> LDAModel`. The model keeps `phi_samples` (the Bootstrap pseudo-posterior),
which is what every downstream quantity is averaged over.

1. **Point estimate.** Fit one `sklearn.decomposition.LatentDirichletAllocation` on the **full
   corpus** → `φ̂` (`components_`, row-normalized). Doc-term matrix vocabulary = the top
   `vocab_top_n` ingredients by document frequency.
2. **Bootstrap pseudo-posterior.** Refit `B=50` times on resampled recipes (m-out-of-n,
   `m≈10k`), each **warm-started** from `φ̂` and run a few online passes; the `B` aligned fits
   become `phi_samples`, so the headline "uncertainty propagation" flows through Steps 3–5.
3. **Label alignment.** Each refit's topics come out permuted, so every `φ̂_b` is matched back
   onto `φ̂` with the **Hungarian algorithm on cosine similarity** (`topic_alignment.align_phi`).
   After alignment `phi_samples[:, k, :]` is a coherent "topic k" across draws, so the downstream
   KL is meaningful.
4. **K selection by held-out perplexity** over `K ∈ {4,6,8,10,12}` (train 90% / score 10%).
   *Honest result:* perplexity is **monotonic in `K`** on this corpus, so it picks the smallest
   (`K=4`) — the data genuinely favors few, coarse topics. `K` stays overridable
   (`train_lda(df, K=6)`) for richer `flavor_tags`; recommendations are robust to `K` because
   coverage and rating dominate the score.

> **Honest caveat — `φ` is a point estimate.** The Bootstrap spread is **bootstrap stability**,
> not a Bayesian posterior: on a large corpus `φ` is very well determined, so it is small *by
> design* (mean per-element `φ` std ≈ `5e-4`). We therefore report it as *stability*, not
> *posterior uncertainty*, and `posterior_uncertainty` in the output is ≈ 0. This is a defensible
> approximation **because** `φ` carries little uncertainty at 53k recipes; the genuinely uncertain
> quantity is the *user's* topic posterior (Step 3), inferred from a handful of ingredients, and
> **that stays fully Bayesian**.

The hyper-priors enter as sklearn's Dirichlet priors: `alpha = doc_topic_prior = 0.1` (sparse
topic mix per recipe), `beta = topic_word_prior = 0.01` (sparse ingredient set per topic).

### Step 2 — `filter_candidates()`
`coverage = |user ∩ recipe| / |recipe|`. Keep `coverage ≥ 0.7`; if fewer than 20
recipes survive, relax to `0.5`. Both sides are run through the same ingredient
canonicalizer so the set intersection is meaningful.

### Step 3 — `infer_user_posterior()` : posterior over flavor topics
Treat the user's ingredients as evidence and apply Bayes' theorem **per `φ`-sample**:

```
P(topic=k | ingredients) ∝ P(ingredients | topic=k) · P(topic=k)
P(ingredients | topic=k)  = Π_i phi[k, i]
P(topic=k)                = empirical marginal topic frequency (mean recipe mix)
```

We evaluate this for every Bootstrap sample of `φ` and keep the per-sample posteriors, so the
"flavor profile" carries uncertainty rather than collapsing to a point estimate. This is the
genuinely Bayesian step: the posterior over topics is peaked for a focused pantry and spread for
an ambiguous one. The *same* function profiles each recipe in Step 4.

### Step 4 — `score_recipes()` : composite score
```
score = coverage^2 · overlap_bonus · flavor_alignment^1 · rating_prior^1
```
- **coverage^2** — dominant term; a recipe you cannot make is useless.
- **overlap_bonus** = `1 − exp(−|user ∩ recipe| / τ)`, `τ=4`. coverage is a *ratio*, so
  a 2–3 ingredient recipe trivially scores 1.0; this saturating bonus in the *absolute*
  number of matched ingredients demotes those trivial matches and rewards substantively
  bigger ones (a real 6/7 over a token 3/3).
- **flavor_alignment** = `exp(−KL(recipe_topics ‖ user_topics))`, computed per Bootstrap
  sample (recipe & user posteriors paired by draw) then averaged. Its per-sample **std** is
  reported as `posterior_uncertainty` (bootstrap *stability*; ≈ 0 because `φ` is well determined).
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
  "flavor_tags": [...],            # top-2 topic labels
  "predicted_rating": float,
  "posterior_uncertainty": float, # std of flavor alignment across Bootstrap draws
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

Built from the cached model (`models/lda_model.pkl`) — no retraining, runs in seconds.

**Model selection — held-out perplexity across K**

![Held-out perplexity across K](figures/fig1_model_selection.png)

*Held-out perplexity (lower = better) is monotone in K on this corpus, so the data favors few,
coarse topics; K is overridable for richer topics.*

**Topic→ingredient φ (point estimate, 94% Bootstrap intervals)**

![Per-topic top ingredients with 94% Bootstrap intervals on phi](figures/fig2_topic_phi_posterior.png)

*The Bootstrap "stability" band is small — `φ` is well determined on the full corpus.*

**Step 3 — the user's flavor posterior**

![P(topic | pantry)](figures/fig3_user_topic_posterior.png)

*`P(topic | pantry)`: the Italian pantry collapses onto one topic (certain), the baker's pantry
spreads across two (genuinely uncertain) — this is the Bayesian step.*

**Uncertainty propagated into the ranking**

![Top-5 flavor-alignment spread](figures/fig4_recommendation_uncertainty.png)

*Per-Bootstrap-draw flavor alignment for the Top-5 — the `φ` Bootstrap reaches the final score.*

## Design decisions & honest caveats

- **`φ` is a point estimate (the core trade).** A fully-Bayesian NUTS fit does not scale to ~50k
  recipes × thousands of ingredients, so `φ` is fit once with sklearn on the full corpus and its
  uncertainty is approximated by **Bootstrap**. The cost: the Bootstrap spread is *stability*, not
  a true posterior, and is small because `φ` is well determined at scale
  (`posterior_uncertainty` ≈ 0). The justification: the uncertainty that actually matters is the
  *user's* topic posterior (Step 3, a few ingredients), which **stays fully Bayesian**.
- **Bootstrap label alignment.** Each Bootstrap refit's topics are arbitrarily ordered;
  `topic_alignment.align_phi` re-permutes every `φ̂_b` onto `φ̂` via the Hungarian algorithm on
  cosine similarity, so `phi_samples[:, k, :]` is a coherent topic `k` across draws.
- **K is corpus-bound.** Held-out perplexity is monotone in K here (it favors the smallest
  candidate), so coarse `flavor_tags` are the honest default; raise `K` (or `vocab_top_n`) for
  richer topics. Recommendations are robust to `K` because coverage and rating dominate.
- **Backend note.** `recommend()` lazily trains and caches the model on first call (keyed on the
  DataFrame); pass `retrain=True` or different `train_kwargs` to refit. `sklearn` / `joblib` are
  imported lazily, so importing the module stays light and Bootstrap workers spawn cheaply.
- **Performance.** Ingredient normalization is the hot path, so `normalize_token` is `lru_cache`d
  (≈474k calls collapse to ≈7k distinct strings) and the catalogue's normalized lists are cached
  per DataFrame. A warm `recommend()` call costs ~0.05 s. Training the full corpus is ~25 s with a
  fixed `K` and ~60 s with the K-selection sweep; the Bootstrap refits run in parallel
  (`n_jobs`), each on an m-out-of-n resample to keep them cheap.
