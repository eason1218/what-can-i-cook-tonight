# What Can I Cook Tonight? — Image → Ingredients → Recipes

**Authors:** Qixin Cui · Kevin Fan · Yizhuo Li · Elaine Wang · Zhetao Zhang (University of Chicago)

An end-to-end pipeline: **take a photo of your ingredients, get the Top-5 recipes you can make.**
Three stages, one entry point.

```
 photo ─▶ ┌─────────────┐   ┌──────────────────────┐   ┌───────────────────────────┐
          │ 1 · data    │   │ 2 · yolo             │   │ 3 · model                 │
          │ recipes_    │   │ YOLOv5 detect(image) │   │ Bayesian LDA recommender  │
          │ clean.csv   │──▶│ → ingredient labels  │──▶│ recommend(ingredients)    │──▶ Top-5
          └─────────────┘   │ → map to vocab       │   │ coverage·flavor·rating    │     recipes
                            └──────────────────────┘   └───────────────────────────┘
              data/              yolo/                       model/
                          └────────────────  run_pipeline.py  ────────────────┘
```

## Methodology

> A recipe is a **document**, an ingredient is a **word**, and a cuisine or flavor is a latent
> **topic** — so we model the corpus with **Latent Dirichlet Allocation (LDA)** and turn the learned
> topics into recommendations *that carry their own uncertainty*.

**Honest Bayesian at scale.** A fully-Bayesian NUTS posterior over `φ` (topic → ingredient) doesn't
scale past a few hundred recipes, so we fit `φ` as a **point estimate** with scikit-learn on the
**full 53,573-recipe corpus** (~25 s) and approximate its uncertainty by **Bootstrap**: refit 50×
on resampled recipes, each warm-started from `φ̂` and realigned with the **Hungarian algorithm**
(LDA topics are exchangeable, so label-switching must be undone). The genuinely uncertain, genuinely
Bayesian quantity is the **user's** topic posterior — inferred from a handful of ingredients by
Bayes' theorem, *per `φ`-sample*, so uncertainty flows all the way to the ranking:

```math
P(\text{topic}=k \mid \text{ingredients}) \;\propto\; \Big(\textstyle\prod_{i}\varphi_{k,i}\Big)\,P(\text{topic}=k)
```

**The five steps** — in `model/src/recipe_recommender.py`:

| # | function | what it does |
|:-:|----------|--------------|
| 1 | `train_lda` | point-estimate `φ` on the full corpus + Bootstrap pseudo-posterior (Hungarian-aligned); choose `K` by held-out perplexity |
| 2 | `filter_candidates` | keep recipes you can mostly make — *coverage* (the share of a recipe's ingredients you have) ≥ 0.7 |
| 3 | `infer_user_posterior` | Bayes' theorem **per `φ`-sample** → a flavor profile that keeps its uncertainty |
| 4 | `score_recipes` | the composite score below, computed per sample then averaged |
| 5 | `recommend` | Top-N, with `diet` / `must_use` / `diversity` (MMR) options |

**The ranking score** — four factors, each fixing a concrete failure mode:

```math
\text{score}=\underbrace{\text{coverage}^{2}}_{\text{can you make it?}}\cdot\underbrace{\big(1-e^{-|U\cap R|/\tau}\big)}_{\text{absolute overlap}}\cdot\underbrace{e^{-\mathrm{KL}(\text{recipe}\,\|\,\text{user})}}_{\text{flavor alignment}}\cdot\underbrace{\tfrac{\bar r\,n+\mu\kappa}{n+\kappa}}_{\text{Bayes-shrunk rating}}
```

**Where the Bayes lands.** On 53k recipes `φ` is *very* well determined, so its Bootstrap spread is
tiny (per-element std ≈ 5e-4) — we report that honestly as resampling **stability**, not posterior
width. The real uncertainty is on the **user side**, and it behaves: a focused pantry collapses onto
one topic (certain), an ambiguous one spreads across two (uncertain).

<table>
<tr>
<td width="50%"><img src="model/figures/fig1_model_selection.png" width="100%"></td>
<td width="50%"><img src="model/figures/fig3_user_topic_posterior.png" width="100%"></td>
</tr>
<tr>
<td align="center"><sub><b>Model selection.</b> Held-out perplexity is monotone in K — the data honestly favors few, coarse topics (K=4).</sub></td>
<td align="center"><sub><b>The Bayesian step.</b> P(topic | pantry): the Italian pantry collapses to one topic, the baker's splits across two.</sub></td>
</tr>
</table>

Full derivations, all four figures, and the design caveats: **[`model/README.md`](model/README.md)**.

## Quick start

```bash
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5 yolo/yolov5   # detector backend (once)
# place the trained detector weights at:  yolo/weights/best.pt

python data/prepare_data.py          # Stage 1 → data/recipes_clean.csv
python model/src/run_demo.py         # Stage 3 → model/models/lda_model.pkl (train recommender)
python yolo/build_mapping.py         # Stage 2 → yolo/yolo_vocab_mapping.json (YOLO label → ingredient)

python run_pipeline.py --image samples/fridge.jpg       # end-to-end (bundled sample)
python run_pipeline.py --image my_fridge.jpg --top-n 8 --diet vegetarian --diversity 0.4
```

Bundled example images are in [`samples/`](samples/) (`samples/fridge.jpg` detects garlic, broccoli,
cherry tomato, egg, onion → Top-5). The detector is for **raw/whole ingredients** (the fridge use
case), not plated dishes.

`run_pipeline.py` does: **detect** ingredients in the image (`yolo/detect.py`) → **map** YOLO
labels to canonical ingredients (`yolo/yolo_mapping.py`) → **recommend** recipes
(`model/src/recipe_recommender.py`).

## Web demo (`app.py`)

A [Gradio](https://gradio.app) UI over the **same** recommender + detector — the interactive
counterpart to the `run_pipeline.py` CLI:

```bash
pip install -r requirements.txt        # includes gradio
python app.py                          # → http://127.0.0.1:7860
```

One streamlined flow — **photo → detect → recommend:** upload a fridge /
ingredient photo, YOLO detects the ingredients (fixed confidence `YOLO_CONF = 0.25`), and the
Bayesian recommender returns recipes. Light controls: top-N, diet filter, diversity. Detection runs
automatically on upload (or click *Detect & recommend*). Needs `torch`/`opencv` +
`yolo/weights/best.pt`; if the detector is unavailable the UI says so instead of crashing.

`app.py` uses the repo it lives in as the project root. To run the UI against a project copy
elsewhere, set `FINAL_BAYESIAN_ROOT=/path/to/project`. Port override: `GRADIO_SERVER_PORT=7861`.

## The three stages

### 1 · `data/` — the recipe corpus
- `prepare_data.py` — download Food.com, clean & normalize → `recipes_clean.csv` (53,573 recipes;
  columns `recipe_id, recipe_name, ingredients, avg_rating, n_ratings`). *(gitignored; regenerable)*
- `Data.ipynb` — the original data-engineering notebook.

### 2 · `yolo/` — ingredient detection
- `detect.py` — load the trained **YOLOv5** detector (`weights/best.pt`, **95 ingredient classes**)
  and run it on an image → detected ingredient labels.
- `build_mapping.py` → `yolo_vocab_mapping.json` — maps each YOLO class (e.g. `Cherry_tomatoes`,
  `Green_bell_pepper`) to the canonical ingredient token the recommender uses (reuses the
  recommender's normalizer + rapidfuzz; 93/95 classes map).
- `yolo_mapping.py` — applies that mapping at query time.
- `weights/best.pt` (you provide) and `yolov5/` (cloned upstream repo) are **gitignored**.

### 3 · `model/` — the Bayesian LDA recommender
The heart of the project — see **[Methodology](#methodology)** above for the model, the five steps,
and the score. Full write-up: [`model/README.md`](model/README.md); code lives in
`model/src/recipe_recommender.py`.

## Notes
- **Windows + this `best.pt`:** the checkpoint was trained on Linux (Colab); `yolo/detect.py`
  handles the `PosixPath` unpickling quirk and silences yolov5's pip requirements check.
- The recommender alone (no image) still works: `recommend(["chicken","garlic","tomato",...], df)`.
