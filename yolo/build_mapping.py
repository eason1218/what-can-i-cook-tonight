"""
yolo/build_mapping.py
=====================
Build the YOLO-label -> canonical-ingredient mapping once, and write it to
`yolo_vocab_mapping.json`.

Target vocabulary = the *normalized ingredient tokens that actually occur in
recipes_clean.csv* (via `recipe_recommender.normalize_token`), so a mapped label is
guaranteed to line up with the recommender's coverage set-intersection. Strategy per
label (cheap, deterministic, and reusing the recommender's own canonicalizer):

  1. normalize the label ("Cherry_tomatoes" -> "cherry tomato"); if it occurs in the
     recipe vocabulary, use it;
  2. else if its head noun (last word, e.g. "pepper" of "green bell pepper") occurs,
     use that;
  3. else take the best rapidfuzz match in the vocabulary above a score threshold;
  4. else leave it unmapped (e.g. very Korean-specific classes with no Western recipe
     token).

Run:  python yolo/build_mapping.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "model", "src"))   # recipe_recommender
sys.path.insert(0, _HERE)                                  # detect

import recipe_recommender as rr           # noqa: E402
from detect import class_names            # noqa: E402

DATA = os.path.join(_ROOT, "data", "recipes_clean.csv")
OUT = os.path.join(_HERE, "yolo_vocab_mapping.json")
FUZZY_THRESHOLD = 90                       # WRatio score required to accept a fuzzy match


def build_recipe_vocab(df: pd.DataFrame) -> Counter:
    """Counter of normalized ingredient tokens over the whole catalogue."""
    vocab = Counter()
    for lst in rr._get_recipe_ingredients(df):
        vocab.update(lst)
    return vocab


def map_one(label: str, vocab: Counter, vocab_tokens: list[str]) -> tuple[str | None, str]:
    """Map a single YOLO label -> (ingredient or None, how)."""
    norm = rr.normalize_token(label.replace("_", " "))
    if not norm:
        return None, "empty"
    if norm in vocab:
        return norm, "exact"
    head = norm.split()[-1]
    if head in vocab:
        return head, "head-noun"
    match = process.extractOne(norm, vocab_tokens, scorer=fuzz.WRatio)
    if match and match[1] >= FUZZY_THRESHOLD:
        return match[0], f"fuzzy({match[1]:.0f})"
    return None, "unmapped"


def build() -> dict:
    df = pd.read_csv(DATA)
    vocab = build_recipe_vocab(df)
    vocab_tokens = [w for w, _ in vocab.most_common()]    # frequent first (fuzzy tie-break)
    labels = class_names()
    print(f"[build_mapping] {len(labels)} YOLO classes vs {len(vocab_tokens)} recipe tokens")

    mapping, report = {}, []
    for label in labels:
        ingr, how = map_one(label, vocab, vocab_tokens)
        if ingr:
            mapping[label] = ingr
        report.append((label, ingr, how))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    mapped = sum(1 for _, i, _ in report if i)
    print(f"[build_mapping] mapped {mapped}/{len(labels)} labels -> {OUT}")
    print("[build_mapping] unmapped:",
          [l for l, i, _ in report if not i] or "none")
    for label, ingr, how in report:
        print(f"  {label:<24s} -> {str(ingr):<20s} [{how}]")
    return mapping


if __name__ == "__main__":
    build()
