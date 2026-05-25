"""
yolo/yolo_mapping.py
====================
Stage 2 (mapping): turn raw YOLO class labels (e.g. "Cherry_tomatoes",
"Green_bell_pepper") into the canonical ingredient tokens the recommender uses for
coverage (e.g. "tomato", "bell pepper"). The mapping itself is precomputed once by
`build_mapping.py` into `yolo_vocab_mapping.json`; this module just loads & applies it.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
MAPPING_PATH = os.path.join(_HERE, "yolo_vocab_mapping.json")


def load_mapping(path: str = MAPPING_PATH) -> dict:
    """Load the {yolo_label: canonical_ingredient} mapping (built by build_mapping.py)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it once:  python yolo/build_mapping.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def map_labels(labels, mapping: dict | None = None) -> list[str]:
    """Map YOLO labels to canonical ingredients, dropping unmapped labels and
    duplicates while preserving order. `labels` may be plain strings or
    (label, confidence) tuples."""
    if mapping is None:
        mapping = load_mapping()
    out, seen = [], set()
    for item in labels:
        label = item[0] if isinstance(item, (tuple, list)) else item
        ingr = mapping.get(label) or mapping.get(str(label).replace(" ", "_"))
        if ingr and ingr not in seen:
            seen.add(ingr)
            out.append(ingr)
    return out
