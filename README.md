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

One streamlined flow — **拍照 → 识别 → 推荐 (photo → detect → recommend):** upload a fridge /
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
Latent "flavor topics" over recipes: sklearn **point estimate** of `φ` on the full corpus +
**Bootstrap** pseudo-posterior; recommends by coverage + latent-flavor alignment + Bayesian-smoothed
rating, with uncertainty propagated to the ranking. See [`model/README.md`](model/README.md)
([中文](model/README.zh-CN.md)) for the full write-up, and `model/src/recipe_recommender.py` for the
five functions (`train_lda, filter_candidates, infer_user_posterior, score_recipes, recommend`).

## Notes
- **Windows + this `best.pt`:** the checkpoint was trained on Linux (Colab); `yolo/detect.py`
  handles the `PosixPath` unpickling quirk and silences yolov5's pip requirements check.
- The recommender alone (no image) still works: `recommend(["chicken","garlic","tomato",...], df)`.
