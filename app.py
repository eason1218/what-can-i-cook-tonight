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
:root {
  --bg: #f6f1e8;
  --bg-accent: #fff8ef;
  --card: rgba(255, 252, 246, 0.92);
  --card-strong: rgba(250, 242, 228, 0.98);
  --ink: #1f2933;
  --muted: #5f6c76;
  --line: rgba(110, 86, 61, 0.14);
  --gold: #c9892f;
  --terracotta: #bf5f3c;
  --sage: #6d8a55;
  --shadow: 0 24px 80px rgba(80, 52, 25, 0.12);
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(201, 137, 47, 0.18), transparent 30%),
    radial-gradient(circle at top right, rgba(109, 138, 85, 0.18), transparent 28%),
    linear-gradient(180deg, #fbf6ee 0%, var(--bg) 55%, #efe6d8 100%);
  color: var(--ink);
  font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
}

.gradio-container {
  max-width: 1260px;
  margin: 0 auto;
  padding: 18px 18px 40px;
}

.hero {
  position: relative;
  overflow: hidden;
  padding: 32px 34px 26px;
  border-radius: 30px;
  background:
    linear-gradient(135deg, rgba(24, 30, 35, 0.96), rgba(58, 43, 24, 0.88)),
    linear-gradient(160deg, rgba(201, 137, 47, 0.18), rgba(109, 138, 85, 0.16));
  box-shadow: var(--shadow);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #fff9f1;
}

.hero::after {
  content: "";
  position: absolute;
  inset: auto -50px -60px auto;
  width: 280px;
  height: 280px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(255, 197, 96, 0.36), transparent 70%);
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 14px;
}

.hero h1 {
  margin: 0;
  font-size: 42px;
  line-height: 1.02;
  letter-spacing: -0.03em;
  color: #fff9f1;
}

.hero p {
  max-width: 800px;
  margin: 14px 0 0;
  font-size: 17px;
  line-height: 1.6;
  color: rgba(255, 249, 241, 0.86);
}

.metric-grid,
.recipe-grid {
  display: grid;
  gap: 16px;
}

.metric-grid {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin-top: 22px;
}

.recipe-grid {
  grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
  margin-top: 10px;
}

.metric-card,
.recipe-card,
.section-card {
  border-radius: 24px;
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}

.metric-card {
  background: rgba(255, 255, 255, 0.1);
  padding: 18px 18px 16px;
  backdrop-filter: blur(8px);
}

.metric-card .label {
  display: block;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: rgba(255, 244, 230, 0.72);
}

.metric-card .value {
  display: block;
  margin-top: 8px;
  font-size: 28px;
  font-weight: 700;
}

.metric-card .hint {
  display: block;
  margin-top: 6px;
  color: rgba(255, 244, 230, 0.72);
  font-size: 13px;
}

.metric-card.hintless .hint {
  display: none;
}

.section-card {
  background: var(--card);
  padding: 22px;
}

.section-card h3 {
  margin: 0 0 10px;
  font-size: 22px;
}

.section-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
}

.section-heading {
  margin: 0 0 12px;
  padding-left: 4px;
  color: #f29a17;
  font-size: 19px;
  font-weight: 600;
}

.recipe-card h4 {
  margin: 0 0 8px;
}

.recipe-card p,
.recipe-card li,
.summary-block {
  color: var(--muted);
  line-height: 1.58;
}

.recipe-card {
  background: linear-gradient(180deg, rgba(255, 253, 249, 1), rgba(251, 243, 231, 0.98));
  padding: 20px;
}

.recipe-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0 14px;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 7px 10px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(201, 137, 47, 0.11);
  color: #8f5610;
}

.pill.sage {
  background: rgba(109, 138, 85, 0.12);
  color: #4c6637;
}

.pill.terracotta {
  background: rgba(191, 95, 60, 0.12);
  color: #9a4729;
}

.subhead {
  margin: 0 0 10px;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8b6b4a;
}

.compact-list {
  margin: 0;
  padding-left: 18px;
}

.compact-list li + li {
  margin-top: 4px;
}

.gradio-container .block {
  border-radius: 24px;
}

/* Full-bleed HTML blocks (hero, recipe cards, footer): strip ONLY Gradio's
   block chrome (the card bg/border/shadow/padding + any inner html wrapper) so
   they sit flush to the same left/right edges as the panels below -- this lines
   the dark header up with the content row. The inner content (.hero,
   .recipe-card, .footer-note) keeps its own styling. */
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

/* Markdown text (e.g. the "Mapped ingredient set" list): give the prose a
   small left pad and stop the block from clipping it, so the first glyph
   isn't cut off. Scoped to .prose, so the flush HTML header/cards are
   untouched (those render in .html-container, not .prose). */
.gradio-container .prose {
  overflow: visible !important;
}
.gradio-container .prose > * {
  padding-left: 6px;
  overflow: visible;
}

.gradio-container .gr-button-primary {
  background: linear-gradient(135deg, var(--terracotta), #d98057) !important;
  border: none !important;
  font-weight: 600 !important;
}

.footer-note {
  color: #7a6755;
  font-size: 13px;
  margin-top: 12px;
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="amber",
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
      <div class="eyebrow">Bayesian Recipe Intelligence Demo</div>
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
        # Full list of everything the recipe needs; fall back to the missing-only
        # list for older model outputs that predate the `all_ingredients` field.
        all_ingr = item.get("all_ingredients") or item.get("missing_ingredients", [])
        missing_set = set(item.get("missing_ingredients", []))
        total_n = len(all_ingr)
        have_n = sum(1 for ing in all_ingr if ing not in missing_set)

        # One chip per ingredient: the ones you have (sage) and the ones you still
        # need (terracotta + "need"), so the card shows the complete recipe.
        ingredient_chips = "".join(
            (f'<span class="pill terracotta">{html.escape(ing)} &middot; need</span>'
             if ing in missing_set
             else f'<span class="pill sage">{html.escape(ing)}</span>')
            for ing in all_ingr
        ) or '<span class="pill">No ingredient data</span>'

        tags = "".join(
            f'<span class="pill">{html.escape(tag)}</span>' for tag in item["flavor_tags"]
        )
        cards.append(
            f"""
            <div class="recipe-card">
              <div class="subhead">Recommendation {rank}</div>
              <h4>{html.escape(item["recipe_name"].title())}</h4>
              <div class="recipe-meta">
                <span class="pill">Score {html.escape(str(item["score"]))}</span>
                <span class="pill terracotta">{html.escape(item["coverage"])}</span>
                <span class="pill sage">Predicted rating {html.escape(str(item["predicted_rating"]))}</span>
              </div>
              <p class="summary-block">
                Posterior uncertainty: {html.escape(str(item["posterior_uncertainty"]))}
              </p>
              <p class="subhead">Full recipe &middot; you have {have_n}/{total_n}</p>
              <div class="recipe-meta">{ingredient_chips}</div>
              <p class="subhead">Flavor profile</p>
              <div class="recipe-meta">{tags}</div>
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
            gr.HTML('<div class="section-heading">Detected ingredients</div>')
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
