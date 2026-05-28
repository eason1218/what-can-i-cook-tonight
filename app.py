from __future__ import annotations

import html
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

APP_ROOT = Path(__file__).resolve().parent          # repo root: app.py lives here
LOCAL_CACHE_DIR = APP_ROOT / ".cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
(LOCAL_CACHE_DIR / "matplotlib").mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))


def resolve_project_root() -> Path:
    """Project root = the folder holding data/recipes_clean.csv.

    app.py now lives at the repo root, so that is the default. An optional
    FINAL_BAYESIAN_ROOT override is kept for pointing the UI at a project copy
    that lives somewhere else (the old teammate-handoff use case).
    """
    override = os.getenv("FINAL_BAYESIAN_ROOT")
    root = Path(override).expanduser() if override else APP_ROOT
    if not (root / "data" / "recipes_clean.csv").exists():
        raise FileNotFoundError(
            f"Could not find data/recipes_clean.csv under {root}.\n"
            "Run Stage 1 first:  python data/prepare_data.py\n"
            "(or set FINAL_BAYESIAN_ROOT to a project copy that already has it)."
        )
    return root


PROJECT_ROOT = resolve_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "recipes_clean.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "models" / "lda_model.pkl"
RESULTS_PATH = PROJECT_ROOT / "model" / "results.json"
MODEL_SELECTION_PATH = PROJECT_ROOT / "model" / "model_selection.json"
YOLO_MAPPING_PATH = PROJECT_ROOT / "yolo" / "yolo_vocab_mapping.json"
YOLO_WEIGHTS_PATH = PROJECT_ROOT / "yolo" / "weights" / "best.pt"
SAMPLES_DIR = PROJECT_ROOT / "samples"
FIGURES_DIR = PROJECT_ROOT / "model" / "figures"

MODEL_SRC_PATH = PROJECT_ROOT / "model" / "src"
YOLO_SRC_PATH = PROJECT_ROOT / "yolo"

if str(MODEL_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC_PATH))
if str(YOLO_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(YOLO_SRC_PATH))

import recipe_recommender as rr  # noqa: E402


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@400;500;600&display=swap');

:root {
  --yellow: #F5C518;
  --yellow-deep: #E8A800;
  --cream: #FFFBF0;
  --brown-dark: #2C1A0E;
  --terracotta: #C85C38;
  --herb-green: #5C8A44;
  --ink: #2C1A0E;
  --muted: #6B5740;
  --shadow: 0 8px 40px rgba(44, 26, 14, 0.12);
}

@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
  0%   { opacity: 0; transform: scale(0); }
  70%  { transform: scale(1.08); }
  100% { opacity: 1; transform: scale(1); }
}

@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

@keyframes barGrow {
  from { width: 0; }
}

body, .gradio-container {
  background: #FFFBF0;
  color: var(--ink);
  font-family: 'Inter', 'Segoe UI', sans-serif;
}

.gradio-container {
  max-width: 100%;
  margin: 0;
  padding: 18px 24px 40px;
}

/* ── Hero ─────────────────────────────── */
.hero {
  position: relative;
  overflow: hidden;
  padding: 48px 52px;
  border-radius: 28px;
  background-color: var(--brown-dark);
  box-shadow: var(--shadow);
  animation: fadeSlideUp 0.6s ease-out both;
}

.hero-glow {
  position: absolute;
  bottom: -40px;
  right: -40px;
  width: 320px;
  height: 320px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(245,197,24,0.35) 0%, transparent 70%);
  pointer-events: none;
}

.hero-hat {
  position: absolute;
  bottom: 0;
  right: 52px;
  opacity: 0.08;
  color: var(--yellow);
}

.hero h1 {
  margin: 0;
  font-family: 'Playfair Display', serif;
  font-size: 52px;
  font-weight: 800;
  line-height: 1.1;
  color: var(--yellow);
}

.hero-illustration {
  position: absolute;
  right: 52px;
  top: 50%;
  transform: translateY(-50%);
}

/* ── Input card ───────────────────────── */
.input-card {
  background: #fff;
  border-radius: 28px;
  border-left: 4px solid var(--yellow);
  box-shadow: var(--shadow);
  padding: 28px;
  animation: fadeSlideUp 0.5s 0.2s ease-out both;
}

/* ── Section heading ──────────────────── */
.section-heading {
  margin: 0 0 14px;
  font-family: 'Playfair Display', serif;
  font-size: 20px;
  font-weight: 700;
  color: var(--brown-dark);
}

/* ── Pills ────────────────────────────── */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  animation: popIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}

.pill.sage {
  background: var(--herb-green);
  color: #fff;
}

.pill.terracotta {
  background: var(--terracotta);
  color: #fff;
}

.pill.gold {
  background: rgba(245, 197, 24, 0.18);
  color: #7A5C00;
}

/* ── Recipe grid ──────────────────────── */
.recipe-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
  margin-top: 10px;
}

/* ── Recipe card ──────────────────────── */
.recipe-card {
  background: var(--cream);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: var(--shadow);
  animation: fadeSlideUp 0.5s ease-out both;
}

.recipe-card-topbar {
  height: 6px;
  background: linear-gradient(90deg, var(--yellow) 0%, var(--yellow-deep) 100%);
}

.recipe-card-body {
  padding: 20px;
}

.recipe-card-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 14px;
}

.rank-badge {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--yellow);
  color: var(--brown-dark);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Playfair Display', serif;
  font-size: 18px;
  font-weight: 700;
  box-shadow: 0 2px 8px rgba(245, 197, 24, 0.4);
}

.recipe-title {
  margin: 0;
  font-family: 'Playfair Display', serif;
  font-size: 17px;
  font-weight: 700;
  color: var(--brown-dark);
  line-height: 1.3;
}

.recipe-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0 14px;
}

.subhead {
  margin: 12px 0 8px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
}

/* ── Match bar ────────────────────────── */
.match-bar-wrap {
  margin-top: 14px;
}

.match-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 6px;
}

.match-bar-track {
  width: 100%;
  height: 8px;
  border-radius: 999px;
  background: rgba(245, 197, 24, 0.15);
  overflow: hidden;
}

.match-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--yellow) 0%, var(--yellow-deep) 100%);
  animation: barGrow 0.6s ease-out both;
}

/* ── Empty / message card ─────────────── */
.section-card {
  background: #fff;
  border-radius: 20px;
  border: 1px solid rgba(245, 197, 24, 0.25);
  padding: 24px;
  box-shadow: var(--shadow);
}

.section-card h3 {
  margin: 0 0 8px;
  font-family: 'Playfair Display', serif;
  color: var(--brown-dark);
}

.section-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
}

/* ── Gradio overrides ─────────────────── */
.gradio-container .block {
  border-radius: 20px;
}

.flush {
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.flush > .html-container,
.flush > .prose,
.flush > .gr-html {
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
}

.gradio-container .prose {
  overflow: visible !important;
}

.gradio-container .prose > * {
  padding-left: 6px;
  overflow: visible;
}

.gradio-container button.primary {
  background: var(--yellow) !important;
  color: var(--brown-dark) !important;
  border: none !important;
  font-weight: 600 !important;
  border-radius: 999px !important;
}

.footer-note {
  color: var(--muted);
  font-size: 13px;
  margin-top: 16px;
  text-align: center;
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="orange",
    neutral_hue="stone",
    radius_size="lg",
)


# Fixed knobs for the photo -> detect -> recommend flow.
YOLO_CONF = 0.25
TOP_N = 4


def _read_json(path: Path, fallback):
    if not path.exists():
        return fallback
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def load_model_artifacts() -> dict:
    artifacts = _read_json(RESULTS_PATH, {})
    return {
        "results": artifacts,
    }


def parse_ingredient_text(text: str | None) -> list[str]:
    if not text:
        return []
    cleaned = []
    for chunk in text.replace("\n", ",").split(","):
        token = chunk.strip().lower()
        if token and token not in cleaned:
            cleaned.append(token)
    return cleaned


def render_hero_metrics() -> str:
    return """
    <div class="hero">
      <div class="hero-glow"></div>
      <svg class="hero-hat" width="220" height="220" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="0.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M6 13.87A4 4 0 0 1 7.41 6a5.11 5.11 0 0 1 1.05-1.54 5 5 0 0 1 7.08 0A5.11 5.11 0 0 1 16.59 6 4 4 0 0 1 18 13.87V21H6Z"/>
        <line x1="6" x2="18" y1="17" y2="17"/>
      </svg>
      <svg class="hero-illustration" width="110" height="110" viewBox="0 0 120 120"
           fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M30 50 L30 85 C30 90 35 95 40 95 L70 95 C75 95 80 90 80 85 L80 50 M25 50 L85 50"
              stroke="#F5C518" stroke-width="2" stroke-linecap="round"/>
        <circle cx="35" cy="50" r="2" fill="#F5C518"/>
        <circle cx="75" cy="50" r="2" fill="#F5C518"/>
        <ellipse cx="55" cy="45" rx="8" ry="4" stroke="#F5C518" stroke-width="2" fill="none"/>
        <path d="M90 35 L95 70" stroke="#F5C518" stroke-width="2" stroke-linecap="round"/>
        <path d="M92 70 L98 70 L98 80 L92 80 Z" stroke="#F5C518" stroke-width="2" fill="none"/>
      </svg>
      <h1>What Can I Cook Tonight?</h1>
    </div>
    """


def recommendation_table(recommendations: list[dict]) -> pd.DataFrame:
    if not recommendations:
        return pd.DataFrame(
            columns=[
                "Recipe",
                "Score",
                "Coverage",
                "Predicted Rating",
                "Uncertainty",
                "Flavor Tags",
                "Missing Ingredients",
            ]
        )

    rows = []
    for item in recommendations:
        rows.append(
            {
                "Recipe": item["recipe_name"],
                "Score": item["score"],
                "Coverage": item["coverage"],
                "Predicted Rating": item["predicted_rating"],
                "Uncertainty": item["posterior_uncertainty"],
                "Flavor Tags": ", ".join(item["flavor_tags"]),
                "Missing Ingredients": ", ".join(item["missing_ingredients"]) or "None",
            }
        )
    return pd.DataFrame(rows)


_RANK_BADGES = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧"]

def render_recipe_cards(recommendations: list[dict]) -> str:
    if not recommendations:
        return """
        <div class="section-card">
          <h3>No recommendations yet</h3>
          <p>Enter ingredients or upload an image to generate live recipe suggestions.</p>
        </div>
        """

    cards = []
    for rank, item in enumerate(recommendations, start=1):
        all_ingr = item.get("all_ingredients") or item.get("missing_ingredients", [])
        missing_set = set(item.get("missing_ingredients", []))
        total_n = len(all_ingr)
        have_n = sum(1 for ing in all_ingr if ing not in missing_set)

        ingredient_chips = "".join(
            (f'<span class="pill terracotta" style="animation-delay:{i*0.05:.2f}s">'
             f'{html.escape(ing)} &middot; need</span>'
             if ing in missing_set
             else f'<span class="pill sage" style="animation-delay:{i*0.05:.2f}s">'
             f'{html.escape(ing)}</span>')
            for i, ing in enumerate(all_ingr)
        ) or '<span class="pill gold">No ingredient data</span>'

        tags = "".join(
            f'<span class="pill gold" style="animation-delay:{i*0.05:.2f}s">'
            f'{html.escape(tag)}</span>'
            for i, tag in enumerate(item["flavor_tags"])
        )

        try:
            score_val = float(item["score"])
            match_pct = min(100, max(0, int(score_val * 100)))
        except (ValueError, TypeError):
            match_pct = 0

        badge = _RANK_BADGES[rank - 1] if rank <= len(_RANK_BADGES) else str(rank)
        delay = f"{(rank - 1) * 0.08:.2f}s"

        cards.append(
            f"""
            <div class="recipe-card" style="animation-delay:{delay}">
              <div class="recipe-card-topbar"></div>
              <div class="recipe-card-body">
                <div class="recipe-card-header">
                  <div class="rank-badge">{badge}</div>
                  <h4 class="recipe-title">{html.escape(item["recipe_name"].title())}</h4>
                </div>
                <div class="recipe-meta">
                  <span class="pill gold">Score {html.escape(str(item["score"]))}</span>
                  <span class="pill terracotta">{html.escape(item["coverage"])}</span>
                  <span class="pill sage">&#9733; {html.escape(str(item["predicted_rating"]))}</span>
                </div>
                <p class="subhead">Ingredients &middot; you have {have_n}/{total_n}</p>
                <div class="recipe-meta">{ingredient_chips}</div>
                <p class="subhead">Flavor profile</p>
                <div class="recipe-meta">{tags}</div>
                <div class="match-bar-wrap">
                  <div class="match-bar-label">
                    <span>Match</span><span>{match_pct}%</span>
                  </div>
                  <div class="match-bar-track">
                    <div class="match-bar-fill" style="width:{match_pct}%; animation-delay:{delay}"></div>
                  </div>
                </div>
              </div>
            </div>
            """
        )
    return f'<div class="recipe-grid">{"".join(cards)}</div>'


def render_message_card(title: str, message: str) -> str:
    return f"""
    <div class="section-card">
      <h3>{html.escape(title)}</h3>
      <p>{html.escape(message)}</p>
    </div>
    """


def run_recommender(
    ingredients: list[str],
    diet: str,
    diversity: float,
):
    cleaned = parse_ingredient_text(",".join(ingredients))
    if not cleaned:
        return (
            "No ingredients detected — try another photo.",
            render_recipe_cards([]),
        )

    df = load_dataset()
    recommendations = rr.recommend(
        cleaned,
        df,
        top_n=TOP_N,
        diet=diet or None,
        diversity=float(diversity),
        model_path=str(MODEL_PATH),
    )

    summary = f"""
    ### Detected ingredients
    {", ".join(cleaned)}

    Retrieved **{len(recommendations)}** recommendation(s).
    """
    return summary, render_recipe_cards(recommendations)


def detector_status() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        return (
            False,
            "Image detection is not ready on this machine yet. The missing dependency is "
            f"`{exc}`. To make photo upload work, install the vision packages first. "
            "Manual pantry recommendations still work right now.",
        )

    if not YOLO_WEIGHTS_PATH.exists():
        return False, f"YOLO weights are missing at `{YOLO_WEIGHTS_PATH}`."

    try:
        import detect  # noqa: F401
        import yolo_mapping  # noqa: F401
    except Exception as exc:
        return False, f"Detector modules could not be loaded: {exc}"

    return True, "YOLO detector is available for image-based ingredient extraction."


def recommend_from_image(
    image,
    diet: str,
    diversity: float,
):
    status_ok, status_message = detector_status()
    detections_df = pd.DataFrame(columns=["Detected Label", "Confidence"])

    if image is None:
        return (
            status_message,
            detections_df,
            "Upload a photo of your ingredients to continue.",
            render_message_card(
                "Nothing to analyze yet",
                "Upload a fridge / ingredient photo and the app will detect and recommend.",
            ),
        )

    if not status_ok:
        return (
            status_message,
            detections_df,
            "### Image pipeline unavailable\n" + status_message,
            render_message_card(
                "Detector not available",
                "The photo was received, but the YOLO detector could not run in this environment.",
            ),
        )

    import detect
    import yolo_mapping

    detections = detect.detect(image, weights=str(YOLO_WEIGHTS_PATH), conf=YOLO_CONF)
    detections_df = pd.DataFrame(detections, columns=["Detected Label", "Confidence"])
    mapped_ingredients = yolo_mapping.map_labels(detections)
    mapped_summary = "### Mapped ingredient set\n" + (", ".join(mapped_ingredients) or "(none detected)")

    if not mapped_ingredients:
        return (
            status_message,
            detections_df,
            mapped_summary,
            render_message_card(
                "No ingredients recognized",
                "The detector did not find known ingredients — try a clearer shot of whole / raw items.",
            ),
        )

    _summary, cards_html = run_recommender(
        ingredients=mapped_ingredients,
        diet=diet,
        diversity=diversity,
    )
    return status_message, detections_df, mapped_summary, cards_html


with gr.Blocks(
    title="What Can I Cook Tonight?",
) as demo:
    gr.HTML(render_hero_metrics(), elem_classes=["flush"])
    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(label="Ingredient photo", type="pil")
            with gr.Row():
                diversity = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Diversity")
                diet = gr.Dropdown(
                    choices=["", "vegan"],
                    value="",
                    label="Diet filter",
                )
            image_button = gr.Button("Detect & recommend", variant="primary")
            image_status_note = gr.Markdown(value=detector_status()[1])
        with gr.Column(scale=4):
            gr.HTML('<div class="section-heading">Detected Ingredients</div>')
            detector_table = gr.Dataframe(
                value=pd.DataFrame(columns=["Detected Label", "Confidence"]),
                interactive=False,
                show_label=False,
            )
            mapped_summary = gr.Markdown(value="Detected ingredients will appear here.")

    image_cards = gr.HTML(render_recipe_cards([]), elem_classes=["flush"])

    _flow_inputs = [image_input, diet, diversity]
    _flow_outputs = [image_status_note, detector_table, mapped_summary, image_cards]
    image_button.click(fn=recommend_from_image, inputs=_flow_inputs, outputs=_flow_outputs)
    image_input.upload(fn=recommend_from_image, inputs=_flow_inputs, outputs=_flow_outputs)

    gr.HTML(
        """
        <div class="footer-note">
          Gradio front-end for the What-Can-I-Cook-Tonight pipeline (data &middot; yolo &middot; model). Same recommender as run_pipeline.py.
        </div>
        """,
        elem_classes=["flush"],
    )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "show_error": True,
        "css": APP_CSS,
        "theme": APP_THEME,
    }
    if os.getenv("GRADIO_SERVER_PORT"):
        launch_kwargs["server_port"] = int(os.environ["GRADIO_SERVER_PORT"])
    demo.queue().launch(**launch_kwargs)
