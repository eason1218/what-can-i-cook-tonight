"""
prepare_data.py
===============
Download the Food.com dataset via kagglehub and emit `data/recipes_clean.csv` with the
schema the recommender expects:

    recipe_id, recipe_name, ingredients (JSON list of standardized names),
    avg_rating, n_ratings

This mirrors the filtering logic of Data.ipynb (>=5 ratings, 2..35 ingredients) but
uses the lightweight, dependency-free ingredient canonicalizer from
recipe_recommender.normalize_list (Food.com ingredients are already phrase-level,
so we only need lowercase + de-punctuate + drop generic modifiers + singularize).
A per-unique-string cache keeps it fast.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from recipe_recommender import _coerce_ingredients, normalize_token

MIN_RATINGS = 5
MIN_INGR, MAX_INGR = 2, 35
OUT_PATH = "data/recipes_clean.csv"


def _download() -> str:
    import kagglehub
    return kagglehub.dataset_download(
        "shuyangli94/food-com-recipes-and-user-interactions")


def _aggregate_ratings(interactions_csv: str) -> pd.DataFrame:
    """Weighted mean rating + count per recipe, computed in chunks (file is ~350MB)."""
    chunks = []
    for chunk in pd.read_csv(interactions_csv, usecols=["recipe_id", "rating"],
                             chunksize=200_000):
        chunks.append(chunk.groupby("recipe_id")["rating"].agg(
            n_ratings="count", rating_sum="sum"))
    agg = pd.concat(chunks).groupby(level=0).sum()
    agg["avg_rating"] = agg["rating_sum"] / agg["n_ratings"]
    return agg[["n_ratings", "avg_rating"]]


def main() -> None:
    base = _download()
    recipes_csv = os.path.join(base, "RAW_recipes.csv")
    interactions_csv = os.path.join(base, "RAW_interactions.csv")

    print("Aggregating ratings ...")
    ratings = _aggregate_ratings(interactions_csv)

    print("Loading recipes ...")
    rec = pd.read_csv(recipes_csv, usecols=["id", "name", "ingredients",
                                            "n_ingredients"], low_memory=False)

    # ---- merge + filter (mirror the notebook) ---------------------------------
    df = rec.merge(ratings, left_on="id", right_index=True, how="inner")
    df = df.dropna(subset=["name", "ingredients", "n_ingredients"])
    df = df[df["n_ratings"] >= MIN_RATINGS]
    df = df[(df["n_ingredients"] >= MIN_INGR) & (df["n_ingredients"] <= MAX_INGR)]
    print(f"After filtering: {len(df):,} recipes")

    # ---- normalize ingredients (cached per unique raw string) -----------------
    cache: dict[str, str] = {}

    def norm_one(raw: str) -> str:
        if raw not in cache:
            cache[raw] = normalize_token(raw)
        return cache[raw]

    def norm_list(cell) -> list[str]:
        out, seen = [], set()
        for it in _coerce_ingredients(cell):
            t = norm_one(it)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    print("Normalizing ingredients ...")
    df["ingredients_norm"] = df["ingredients"].map(norm_list)
    df = df[df["ingredients_norm"].map(len) >= MIN_INGR]

    out = pd.DataFrame({
        "recipe_id": df["id"].values,
        "recipe_name": df["name"].values,
        "ingredients": df["ingredients_norm"].map(json.dumps).values,
        "avg_rating": df["avg_rating"].round(4).values,
        "n_ratings": df["n_ratings"].astype(int).values,
    })
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {len(out):,} recipes")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
