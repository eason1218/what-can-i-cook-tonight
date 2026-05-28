# Stage 2 · Detection — photo → ingredient tokens

Turn a photo of your fridge or counter into a clean list of **canonical ingredient tokens** the
recommender can use. Two steps: a **YOLOv5 detector** spots ingredients, then a precomputed
**label → ingredient map** rewrites the detector's class names into the recipe vocabulary.

```text
photo ──▶ detect.py (YOLOv5) ──▶ [(label, conf), …] ──▶ yolo_mapping.py ──▶ ["tomato", "egg", …]
                                                              ▲
                                          build_mapping.py ───┘  (offline: builds the map once)
```

## Files

| File | Role |
|------|------|
| `detect.py` | Load the trained detector and run it on an image → `[(label, confidence)]` |
| `build_mapping.py` | **Offline, run once:** build `yolo_vocab_mapping.json` (detector label → recipe ingredient) |
| `yolo_mapping.py` | **Query time:** apply that map, dropping unmapped labels and duplicates |
| `yolo_vocab_mapping.json` | the committed map — 93 of 95 classes → 87 distinct ingredients |
| `mapping_vocab.ipynb` | the notebook the label → ingredient mapping was explored in |
| `weights/best.pt` | the detector checkpoint from [HuggingFace `HYUNAHKO/Ingredients_object_detection`](https://huggingface.co/HYUNAHKO/Ingredients_object_detection) (gitignored; you download) |
| `yolov5/` | upstream [Ultralytics](https://github.com/ultralytics/yolov5) backend (cloned; gitignored) |

## The detector

The weights `weights/best.pt` are a **fine-tuned YOLOv5** model with **95 ingredient classes**, taken
from the Hugging Face checkpoint
[`HYUNAHKO/Ingredients_object_detection`](https://huggingface.co/HYUNAHKO/Ingredients_object_detection).
The class vocabulary — `Wakame`, `Napa_cabbage`, `Cabbage_kimchi`, `Radish_kimchi`, `Enoki_mushrooms`,
`Somen`, `Udon`, `Ramen`, `Soybean_sprouts`, `Seasoned_seaweed`, alongside Western staples like
`Tomato`, `Egg`, `Chicken`, `Cheese` — is an **Asian / Korean fridge-staples** set. We don't train it
ourselves: this repo ships only inference, expects the checkpoint at `weights/best.pt`, and clones the
upstream `yolov5` repo as the backend.

### Inference (`detect.py`)

`detect(image, conf=0.25)` returns `[(label, confidence), …]`:

1. **Load once, cached.** The model is loaded via
   `torch.hub.load(yolov5_dir, "custom", path=weights, source="local")` and memoized with
   `lru_cache`, so repeated calls don't reload the network. The returned AutoShape model accepts a
   path / PIL image / ndarray and runs **non-max suppression** internally.
2. **Confidence threshold.** Detections below `conf` (default `0.25`) are dropped.
3. **One box per class.** We sort by confidence and keep the **highest-confidence detection per
   class** (`drop_duplicates("name")`). For recommendation we only care *whether* an ingredient is
   present, not how many instances there are — keeping duplicates would only add noise to the pantry
   set.

### Two portability gotchas

The checkpoint was saved on Linux, which bites on Windows — both handled in `detect.py`:

- **`PosixPath` unpickling.** A Linux-saved checkpoint pickles a `PosixPath`, which cannot
  instantiate on Windows. We temporarily alias `pathlib.PosixPath = pathlib.WindowsPath` around the
  load, then restore it.
- **`check_requirements`.** yolov5's `hubconf` calls `check_requirements()`, which shells out to
  `pip` and is slow/noisy (it can choke on requirements markers). We neutralize it before the load —
  the environment already has torch / opencv.

## The label → ingredient map

The detector emits class names (`Cherry_tomatoes`, `Green_bell_pepper`); the recommender's *coverage*
works on the corpus's **normalized ingredient tokens** (`cherry tomato`, `bell pepper`). The two must
be reconciled, or every detection would miss the set-intersection that drives the recommendation.

### Why build against the corpus vocabulary

`build_mapping.py` maps each label **to a token that actually occurs in `data/recipes_clean.csv`**
(via the recommender's own `normalize_token`). This is the key design choice: it guarantees a mapped
ingredient lines up with coverage, so there's no second, drifting vocabulary to maintain.

### The mapping cascade (per label, deterministic — first hit wins)

1. **Exact.** Normalize the label (`Cherry_tomatoes → cherry tomato`); if it's in the recipe vocab, use it.
2. **Head noun.** Else try the last word (`green bell pepper → pepper`) — recipes talk about *pepper*.
3. **Fuzzy.** Else take the best RapidFuzz `WRatio` match in the vocab, if it scores **≥ 90**.
4. **Unmapped.** Else leave it out (e.g. very region-specific classes with no Western recipe token).

**Coverage:** 93 of the 95 classes map, to 87 distinct ingredients.

### Honest mapping artifacts

The cascade is cheap and deterministic, which means a handful of mappings are coarse — worth knowing:

| Label | Mapped to | Why |
|-------|-----------|-----|
| `Crab_sticks` | `stick` | head-noun fired on the wrong word |
| `Tomato_pasta_sauce`, `Hot_sauce` | `sauce` | head-noun collapses specific sauces to "sauce" |
| `Soybean_sprouts` | `bean` | head-noun, not "bean sprout" |
| `Ramen`, `Somen` | `raman noodle`, `soman noodle` | fuzzy-matched mis-spelled tokens that exist in the Food.com corpus |

These are tolerable because **coverage is forgiving** — a slightly-off token just changes which
recipes a detected item helps unlock; it can't corrupt the Bayesian model (`φ` is fixed) or invent
recipes. The map is plain JSON, so you can edit `yolo_vocab_mapping.json` by hand, or re-run
`build_mapping.py` after changing `FUZZY_THRESHOLD`.

### Apply at query time (`yolo_mapping.py`)

`map_labels(detections)` loads the JSON map and rewrites detections into canonical tokens, **dropping
unmapped labels and de-duplicating** while preserving order. It accepts either bare strings or
`(label, confidence)` tuples, so it slots directly behind `detect()`.

## Usage

```bash
# one-time setup
git clone https://github.com/ultralytics/yolov5 yolo/yolov5
# download best.pt from HuggingFace (HYUNAHKO/Ingredients_object_detection) -> yolo/weights/best.pt

# detect on an image (prints "conf  label")
python yolo/detect.py samples/fridge.jpg --conf 0.25

# (re)build the label → ingredient map  (needs data/recipes_clean.csv)
python yolo/build_mapping.py            # → yolo/yolo_vocab_mapping.json
```

In code — exactly what `run_pipeline.py` and `app.py` do:

```python
import detect, yolo_mapping
detections  = detect.detect("samples/fridge.jpg", conf=0.25)   # [("Tomato", 0.88), …]
ingredients = yolo_mapping.map_labels(detections)              # ["tomato", …]
```

> **No detector?** The recommender is fully usable without it — just pass typed ingredients to
> `rr.recommend([...], df)`. The detector is the convenience front-end, not a dependency of the
> Bayesian model.
