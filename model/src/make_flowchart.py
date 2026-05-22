"""make_flowchart.py — render the end-to-end pipeline diagram to figures/pipeline_flowchart.png.
Horizontal (16:9), English, polished. Pure matplotlib. Run: python make_flowchart.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# palette
BLUE_F, BLUE_E = "#dbe7f6", "#2f6bb3"
GREEN_F, GREEN_E = "#d8efdf", "#2e8b57"
GOLD_F, GOLD_E = "#ffe7a3", "#d99800"
GREY_F, GREY_E = "#eef1f4", "#8a98a6"
OUT_F, OUT_E = "#bfe3c8", "#1c7a3e"
BAND_OFF, BAND_ON = "#eef4fb", "#edf7f0"
INK = "#23303a"

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 160); ax.set_ylim(0, 90); ax.set_aspect("equal"); ax.axis("off")


def box(cx, cy, w, h, title, body="", fc="#fff", ec="#888", badge=None,
        fs_t=11, fs_b=8.6, tcol=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2 + 0.7, cy - h / 2 - 0.7), w, h,
                 boxstyle="round,pad=0.5", fc="#0000001a", ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=2.0, zorder=2))
    if badge is not None:
        bx, by = cx - w / 2, cy + h / 2          # sit on the top-left corner (outside text)
        ax.add_patch(Circle((bx, by), 1.95, fc=ec, ec="white", lw=1.4, zorder=5))
        ax.text(bx, by, str(badge), ha="center", va="center", color="white",
                fontsize=9.5, weight="bold", zorder=6)
    if body:
        ax.text(cx, cy + h * 0.22, title, ha="center", va="center",
                fontsize=fs_t, weight="bold", color=tcol, zorder=3)
        ax.text(cx, cy - h * 0.20, body, ha="center", va="center",
                fontsize=fs_b, color="#3a4750", zorder=3, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fs_t,
                weight="bold", color=tcol, zorder=3)


def harrow(x1, x2, y, color="#5a6672", label=None):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), zorder=3,
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=color,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2, y + 2.0, label, ha="center", va="bottom",
                fontsize=7.6, color="#5a6672", zorder=3)


def varrow(x, y1, y2, color="#5a6672"):
    ax.annotate("", xy=(x, y2), xytext=(x, y1), zorder=3,
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=color))


# ---- title ---------------------------------------------------------------
ax.text(80, 86, "What Can I Cook Tonight?", ha="center", fontsize=19,
        weight="bold", color=INK)
ax.text(80, 81.5, "Bayesian LDA recipe recommender  ·  end-to-end pipeline",
        ha="center", fontsize=11, color="#6b7884")

# ---- phase bands ---------------------------------------------------------
ax.add_patch(FancyBboxPatch((4, 52), 152, 23, boxstyle="round,pad=0.6",
             fc=BAND_OFF, ec="#c9d8ea", lw=1.2, zorder=0))
ax.add_patch(FancyBboxPatch((4, 6), 152, 40, boxstyle="round,pad=0.6",
             fc=BAND_ON, ec="#c7e3d1", lw=1.2, zorder=0))
ax.text(9, 71.5, "OFFLINE  ·  build the model (trained once)", fontsize=11.5,
        weight="bold", color=BLUE_E, zorder=1)
ax.text(9, 42.5, "ONLINE  ·  answer a query (reuses the posterior)", fontsize=11.5,
        weight="bold", color=GREEN_E, zorder=1)

# ---- OFFLINE lane --------------------------------------------------------
YO = 62
box(22, YO, 28, 13, "Food.com data", "RAW_recipes\n+ RAW_interactions", GREY_F, GREY_E)
harrow(36.5, 41.5, YO, label="prepare_data.py")
box(55, YO, 26, 13, "data/recipes_clean.csv", "53,573 recipes", GREY_F, GREY_E)
harrow(68.5, 74.5, YO)
box(99, YO, 42, 15, "train_lda", "Bayesian LDA via NUTS  (z marginalized)\n"
    "choose K by held-out predictive lppd\nrefit best K on the full corpus",
    BLUE_F, BLUE_E, badge=1, fs_t=12)
harrow(120.5, 126.5, YO)
box(140, YO, 26, 14, "posterior φ", "samples (S×K×V)\nmodels/lda_model.pkl", GOLD_F, GOLD_E,
    fs_t=12, tcol="#7a5400")

# ---- bridge: model feeds the online steps --------------------------------
ax.add_patch(FancyArrowPatch((140, 54.5), (110, 34.8),
             connectionstyle="arc3,rad=-0.28", arrowstyle="-|>",
             mutation_scale=22, lw=2.4, color=GOLD_E, zorder=3))
ax.text(150, 46, "φ posterior\nused in Steps 3–4", fontsize=8.2, color="#a87b00",
        ha="center", va="center", style="italic", zorder=3)

# ---- ONLINE lane ---------------------------------------------------------
YN = 27
box(16, YN, 22, 14, "your pantry", "chicken, garlic,\ntomato, salt, ...", "#ffffff",
    GREEN_E, fs_t=11)
harrow(27, 32, YN)
box(46, YN, 27, 14, "filter_candidates", "coverage =\n|U ∩ R| / |R| ≥ 0.7",
    GREEN_F, GREEN_E, badge=2, fs_t=10)
harrow(59.5, 63.5, YN)
box(78, YN, 29, 14, "infer_user_posterior", "P(topic | ingredients)\nper φ-sample",
    GREEN_F, GREEN_E, badge=3, fs_t=10)
harrow(92.5, 95.5, YN)
box(110, YN, 28, 15, "score_recipes", "coverage² · overlap ·\nalignment · rating\n"
    "per-sample → uncertainty", GREEN_F, GREEN_E, badge=4, fs_t=10.5, fs_b=8.0)
harrow(124, 127.5, YN)
box(140, YN, 24, 14, "recommend", "Top-N  + diet /\nmust_use / diversity",
    GREEN_F, GREEN_E, badge=5, fs_t=10.5)
varrow(140, 19.5, 13.5)
box(140, 9.5, 32, 7.5, "Top-N recommendations with uncertainty", "", OUT_F, OUT_E,
    fs_t=9.5, tcol="#14552b")

# ---- caption -------------------------------------------------------------
ax.text(80, 1.8, "Steps 1–5 of the pipeline.  Offline training is cached in "
        "models/lda_model.pkl; every online query reuses the posterior, so uncertainty "
        "propagates to each recommendation.",
        ha="center", fontsize=8.4, color="#6b7884", style="italic")

import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/pipeline_flowchart.png", dpi=150, bbox_inches="tight")
print("wrote figures/pipeline_flowchart.png")
