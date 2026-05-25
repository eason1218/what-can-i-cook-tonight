"""
yolo/detect.py
==============
Stage 2 (detection): run the trained YOLOv5 ingredient detector on an image and
return the detected ingredient labels.

The weights (`weights/best.pt`, 95 ingredient classes) were trained with the
ultralytics/yolov5 repo, so we load them through that repo (cloned at `yolo/yolov5`).
Two portability details are handled here:
  * the checkpoint was saved on Linux (Colab) and pickles a `PosixPath`, which cannot
    instantiate on Windows -> we temporarily alias `PosixPath` to `WindowsPath`;
  * the model is cached (lru_cache) so repeated calls don't reload it.
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_YOLOV5_DIR = os.path.join(_HERE, "yolov5")
DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "best.pt")


@lru_cache(maxsize=2)
def load_detector(weights: str = DEFAULT_WEIGHTS, repo_dir: str = _YOLOV5_DIR):
    """Load the YOLOv5 detector (cached). Returns an AutoShape model that accepts a
    path / PIL / ndarray and does NMS internally."""
    import torch

    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(
            f"yolov5 repo not found at {repo_dir}. Clone it once:\n"
            f"    git clone https://github.com/ultralytics/yolov5 {repo_dir}")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"weights not found at {weights}. Put the trained best.pt there.")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # yolov5's hubconf calls check_requirements(), which shells out to pip and is both
    # slow and noisy (it can choke on requirements.txt markers). Neutralize it before
    # hubconf imports the name -- our environment already has torch/opencv/etc.
    try:
        import utils.general as _yolo_general
        _yolo_general.check_requirements = lambda *a, **k: None
    except Exception:
        pass

    _posix = pathlib.PosixPath
    if os.name == "nt":                       # Windows: load a Linux-saved checkpoint
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = torch.hub.load(repo_dir, "custom", path=weights, source="local")
    finally:
        pathlib.PosixPath = _posix
    return model


def class_names(weights: str = DEFAULT_WEIGHTS, repo_dir: str = _YOLOV5_DIR) -> list[str]:
    """The detector's class list (95 ingredient labels)."""
    names = load_detector(weights, repo_dir).names
    return list(names.values()) if isinstance(names, dict) else list(names)


def detect(image, weights: str = DEFAULT_WEIGHTS, conf: float = 0.25,
           repo_dir: str = _YOLOV5_DIR) -> list[tuple[str, float]]:
    """Detect ingredients in `image` (path / PIL.Image / ndarray).

    Returns a list of (label, confidence), one per detected class (the highest-
    confidence box per class), sorted by confidence descending, filtered to
    confidence >= `conf`.
    """
    model = load_detector(weights, repo_dir)
    model.conf = float(conf)
    results = model(image)
    det = results.pandas().xyxy[0]                       # x..,confidence,class,name
    if det.empty:
        return []
    det = det[det["confidence"] >= conf]
    det = det.sort_values("confidence", ascending=False).drop_duplicates("name")
    return [(str(r["name"]), float(r["confidence"])) for _, r in det.iterrows()]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detect ingredients in an image")
    ap.add_argument("image", help="path to an image")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    for label, c in detect(args.image, weights=args.weights, conf=args.conf):
        print(f"{c:.2f}  {label}")
