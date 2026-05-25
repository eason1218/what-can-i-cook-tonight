# 🍳 What Can I Cook Tonight?

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LDA-F7931E?logo=scikitlearn&logoColor=white)
![YOLOv5](https://img.shields.io/badge/YOLOv5-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/demo-Gradio-F97316?logo=gradio&logoColor=white)
![Recipes](https://img.shields.io/badge/corpus-53%2C573%20recipes-success)
![License](https://img.shields.io/badge/license-MIT-green)

Snap a photo of what's in your fridge and get the **recipes you can actually make tonight** — ranked by a Bayesian flavor model that also tells you how confident it is.

Bayesian Methods Final Project — by **Qixin Cui**, **Kevin Fan**, **Yizhuo Li**, **Elaine Wang**, and **Zhetao Zhang** (University of Chicago).

![Demo](assets/demo.png)

> *Upload a fridge photo → the detector spots the ingredients → the recommender returns ranked recipes, each with its full ingredient list and what you're still missing.*

<details>
<summary><b>Table of Contents</b></summary>

- [Features](#features)
- [Architecture](#architecture)
- [The idea](#the-idea)
- [Ranking recipes](#ranking-recipes)
- [How sure is it?](#how-sure-is-it)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

</details>

---

# 📌 Overview

## Features

- 📷 **Photo → ingredients** — a Hugging Face Fine-Tuned YOLOv5 detector (95 ingredient classes) reads a fridge or counter photo.
- 🧠 **Recommendations with confidence** — a Bayesian flavor model ranks recipes *and* reports how sure it is.
- 🧾 **The whole recipe at a glance** — every suggestion lists all the ingredients it needs and flags the ones you're missing.
- 🥗 **Real-life filters** — vegetarian / vegan, "must use this", and a diversity dial so you don't get five near-identical dishes.
- 🖥️ **Web app *and* command line** — a friendly Gradio UI and a one-line command, both running the exact same engine.
- 📚 **53,573 real recipes** — cleaned from the public Food.com dataset.

## Architecture

Three stages, one entry point: a photo goes in, ranked recipes come out.

```mermaid
flowchart LR
    P["photo of your<br/>ingredients"] --> Y
    subgraph Y["2 · yolo — detect"]
      y1["YOLOv5 detector<br/>label → ingredient"]
    end
    subgraph M["3 · model — recommend"]
      m1["Bayesian LDA<br/>flavor model"]
    end
    subgraph D["1 · data"]
      d1["53,573 cleaned<br/>Food.com recipes"]
    end
    Y --> M --> R["ranked recipes<br/>+ what you're missing"]
    D -. corpus .-> M
```

| Stage | Folder | What it does | Tech |
|-------|--------|--------------|------|
| 1 · Data | `data/` | Download and clean the Food.com corpus → `recipes_clean.csv` | pandas · kagglehub |
| 2 · Detect | `yolo/` | Spot ingredients in a photo, map labels to the recipe vocabulary | YOLOv5 · PyTorch · OpenCV · RapidFuzz |
| 3 · Recommend | `model/` | Rank recipes with a Bayesian flavor model | scikit-learn · NumPy · SciPy |
| — | `app.py` · `run_pipeline.py` | Web UI and CLI over the same engine | Gradio |

**Tech stack:** Python 3.13 · scikit-learn (LDA) · YOLOv5 (PyTorch) · Gradio · OpenCV · pandas / NumPy / SciPy.

---

# 🍳 How it works

## The idea

You open the fridge, see a handful of ingredients, and wonder what you can actually cook tonight. We wanted an answer that only suggests dishes you can **mostly make right now** — and that is **honest about how confident it is**. That second goal, uncertainty, is what makes this a Bayesian project.

The trick is to treat cooking like language: a **recipe is a document**, an **ingredient is a word**, and a **cuisine or flavor is a hidden theme**. A topic model (**LDA**) discovers those flavor themes on its own across the whole corpus — one theme leans on olive oil, basil and parmesan; another on flour, sugar and eggs.

Learning these themes the *fully* Bayesian way doesn't scale to tens of thousands of recipes, so we make a pragmatic trade: learn them **once** over all 53,573 recipes, then **re-learn them many times on resampled recipes** to see how much they wobble. They barely move — with this much data the flavor themes are pinned down — which is itself an honest result, not something we paper over.

## Ranking recipes

Every recipe you could make gets a score that blends four common-sense factors, so no single one can win on its own:

- **Can you make it?** How much of the recipe you already have. A dish you can't make is useless, so this counts the most.
- **Real overlap, not technicalities.** A 3-ingredient recipe shouldn't win just because you happen to have all three — bigger, genuine matches are rewarded.
- **Flavor fit.** Does the recipe's flavor theme match the one your ingredients suggest?
- **A rating you can trust.** A dish with a single 5-star vote is nudged toward the average, so one lucky rating can't top the list.

On top of that you can ask for **vegetarian / vegan** only, force a **must-use** ingredient, or turn up **diversity** so the list isn't five variations of the same dish.

## How sure is it?

This is the Bayesian heart. From your handful of ingredients we infer a **flavor profile** — and, crucially, *how sure* we are of it. A focused pantry (clearly Italian) gives a confident, single-theme profile; a vague one (could be baking, could be breakfast) spreads across themes. That confidence then rides along into the recommendations.

<table>
<tr>
<td width="50%"><img src="model/figures/fig3_user_topic_posterior.png" width="100%"></td>
<td width="50%"><img src="model/figures/fig1_model_selection.png" width="100%"></td>
</tr>
<tr>
<td align="center"><sub>A clearly-Italian pantry lands on one flavor theme (confident); a baker's pantry splits across two (uncertain).</sub></td>
<td align="center"><sub>The data favors just a few broad flavor themes — more isn't better here, and we don't pretend otherwise.</sub></td>
</tr>
</table>

Want the full technical story — the math, the design choices, every figure? It's in **[`model/README.md`](model/README.md)**.

---

# 🚀 Getting started

## Project structure

<details>
<summary>Click to expand the file tree</summary>

```
.
├── app.py                        # Gradio web app  (photo → recipes)
├── run_pipeline.py               # command line     (photo → recipes)
├── data/                         # 1 · DATA
│   └── prepare_data.py           #   download + clean Food.com → recipes_clean.csv
├── yolo/                         # 2 · DETECT
│   ├── detect.py                 #   run the YOLOv5 ingredient detector
│   ├── build_mapping.py          #   map detector labels → recipe ingredients
│   └── yolo_mapping.py           #   apply that mapping at query time
├── model/                        # 3 · RECOMMEND   → model/README.md
│   ├── src/recipe_recommender.py #   the recommender (train + recommend)
│   ├── notebooks/                #   walkthrough + a usage example
│   └── figures/                  #   the plots shown above
├── samples/                      # example fridge / ingredient photos
├── requirements.txt
└── LICENSE                       # MIT
```

> **Run everything from the project root** (e.g. `python run_pipeline.py ...`). The recipe CSV, the trained model, and the detector weights are large/regenerable and gitignored — the steps below create them.

</details>

## Installation

```bash
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5 yolo/yolov5   # detector backend (once)
```

Put the trained detector weights at `yolo/weights/best.pt`. *(No detector? The recommender still works — just type your ingredients instead of uploading a photo.)*

## Usage

**Web app** — upload a fridge photo (or pick one from `samples/`); detection and recommendations run automatically.

```bash
python app.py        # → http://127.0.0.1:7860
```

**Command line**

```bash
python run_pipeline.py --image samples/fridge.jpg --top-n 5 --diet vegetarian
```

<details>
<summary>First-time setup — build the data, model, and label map</summary>

```bash
python data/prepare_data.py     # → data/recipes_clean.csv
python model/src/run_demo.py    # → trains the recommender (model/models/lda_model.pkl)
python yolo/build_mapping.py    # → yolo/yolo_vocab_mapping.json
```

</details>

---

# 👥 Project info

## Authors

| Name | | Name |
|------|--|------|
| Qixin Cui | | Kevin Fan |
| Yizhuo Li | | Elaine Wang |
| Zhetao Zhang | | |

## Acknowledgments

- [Food.com Recipes & Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) — the recipe corpus.
- [Ultralytics Yolov5](https://github.com/ultralytics/yolov5) — used as the ingredient-detector backbone.
- [Ingredient Object Detection Model](https://huggingface.co/HYUNAHKO/Ingredients_object_detection) — used as the pre-trained ingredient detection model checkpoint (`best.pt`).
- Built with [scikit-learn](https://scikit-learn.org/), [Gradio](https://www.gradio.app/), and [PyTorch](https://pytorch.org/).

## License

Released under the [MIT License](LICENSE). The recipe data comes from the public Food.com dataset and is used for study and research only.
