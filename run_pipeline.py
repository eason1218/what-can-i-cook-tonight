"""
run_pipeline.py  —  end-to-end: a photo of your ingredients -> Top-N recipes.

Three stages wired together:
    data    : recipes_clean.csv (Stage 1, built by data/prepare_data.py)
    yolo    : detect ingredients in the image  (Stage 2, yolo/detect.py)
              + map YOLO labels -> canonical ingredients (yolo/yolo_mapping.py)
    model   : recommend recipes from those ingredients (Stage 3, model/src/recipe_recommender.py)

Usage:
    python run_pipeline.py --image fridge.jpg
    python run_pipeline.py --image fridge.jpg --top-n 8 --diet vegetarian --diversity 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "yolo"))
sys.path.insert(0, os.path.join(ROOT, "model", "src"))

import pandas as pd                       # noqa: E402

import detect as yolo_detect              # noqa: E402  (yolo/detect.py)
import yolo_mapping                       # noqa: E402  (yolo/yolo_mapping.py)
import recipe_recommender as rr           # noqa: E402  (model/src/)

DATA = os.path.join(ROOT, "data", "recipes_clean.csv")
MODEL = os.path.join(ROOT, "model", "models", "lda_model.pkl")
WEIGHTS = os.path.join(ROOT, "yolo", "weights", "best.pt")


def run(image: str, top_n: int = 5, conf: float = 0.25, weights: str = WEIGHTS,
        **recommend_opts):
    """image -> (detections, mapped ingredients, recommendations)."""
    df = pd.read_csv(DATA)
    model = rr.load_model(MODEL)
    rr._STATE = {"model": model, "df_id": id(df)}            # recommend() reuses it

    detections = yolo_detect.detect(image, weights=weights, conf=conf)   # [(label, conf)]
    ingredients = yolo_mapping.map_labels(detections)                    # canonical tokens
    recs = rr.recommend(ingredients, df, top_n=top_n, **recommend_opts) if ingredients else []
    return detections, ingredients, recs


def main():
    ap = argparse.ArgumentParser(description="Photo of ingredients -> Top-N recipes",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="path to an image of your ingredients")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--diet", choices=["vegetarian", "vegan"], default=None)
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--must-use", nargs="*", default=None)
    ap.add_argument("--diversity", type=float, default=0.0)
    args = ap.parse_args()

    detections, ingredients, recs = run(
        args.image, top_n=args.top_n, conf=args.conf, weights=args.weights,
        diet=args.diet, exclude=args.exclude, must_use=args.must_use,
        diversity=args.diversity)

    print("\n=== STAGE 2 · detected ingredients ===")
    for label, c in detections:
        print(f"  {c:.2f}  {label}")
    print("\nmapped to recipe ingredients:", ingredients or "(none — try a clearer photo / lower --conf)")

    print(f"\n=== STAGE 3 · TOP-{args.top_n} RECIPES ===")
    print(json.dumps(recs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
