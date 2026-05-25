"""
run_demo.py
===========
End-to-end demo of the LDA recipe recommender on the cleaned Food.com data.
Trains the model (sklearn point estimate of phi on the full corpus + Bootstrap
pseudo-posterior; K by held-out perplexity), reports the model-selection table and
diagnostics, then prints Top-5 recommendations for a few pantries.

Outputs:
  models/lda_model.pkl   the fitted model (gitignored)
  model_selection.json   held-out perplexity per K (behind fig1)
  results.json           this demo run (topics + recommendations)
"""

import json
import time

import pandas as pd

import recipe_recommender as rr

DATA = "../data/recipes_clean.csv"        # Stage-1 data lives in the top-level data/ dir

PANTRIES = {
    "Italian-ish": ["chicken", "garlic", "onion", "olive oil", "tomato",
                    "basil", "parmesan cheese", "pasta", "salt", "pepper"],
    "Baker's pantry": ["flour", "sugar", "butter", "egg", "vanilla",
                       "baking soda", "milk", "salt", "chocolate"],
}


def main():
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} recipes\n", flush=True)

    # ---- Step 1: train (K auto-selected by held-out perplexity) ---------------
    t0 = time.time()
    model = rr.train_lda(df)                       # K=None -> auto-select
    train_secs = time.time() - t0
    rr.save_model(model, "models/lda_model.pkl")
    rr._STATE = {"model": model, "df_id": id(df)}      # reuse for recommend()

    # K-selection table behind fig1 (held-out perplexity, lower=better)
    with open("model_selection.json", "w", encoding="utf-8") as f:
        json.dump(model.perplexity_table.to_dict(orient="records"), f, indent=2)

    report = {
        "selected_K": model.best_k,
        "selection_metric": "held-out perplexity (lower=better) on the full corpus",
        "train_wall_seconds": round(train_secs, 1),
        "n_recipes_trained": int(len(df)),
        "perplexity_by_K": model.perplexity_table.to_dict(orient="records"),
        "bootstrap_stability": round(float(model.bootstrap_stability), 6),
        "n_bootstrap": int(model.n_samples),
        "topics": [{"topic": k, "label": lab,
                    "P(topic)": round(float(model.topic_prior[k]), 4)}
                   for k, lab in enumerate(model.topic_labels)],
        "recommendations": {},
    }

    print(f"\n=== Model diagnostics (K={model.best_k}, "
          f"trained on {len(df):,} recipes in {train_secs:.1f}s) ===", flush=True)
    print("held-out perplexity by K (lower = better):", flush=True)
    for r in report["perplexity_by_K"]:
        print(f"  K={r['K']:>2d}  perplexity={r['holdout_perplexity']:.1f}", flush=True)
    print(f"bootstrap_stability (mean phi std over {model.n_samples} resamples) "
          f"= {model.bootstrap_stability:.5f}", flush=True)
    print("\n=== Flavor topics ===", flush=True)
    for t in report["topics"]:
        print(f"  topic {t['topic']}: {t['label']}  (P={t['P(topic)']})", flush=True)

    # ---- Steps 2-5: recommend for each pantry --------------------------------
    for name, pantry in PANTRIES.items():
        recs = rr.recommend(pantry, df)
        report["recommendations"][name] = {"ingredients": pantry, "top5": recs}
        print(f"\n--- TOP-5 for '{name}' ---", flush=True)
        print(json.dumps(recs, indent=2, ensure_ascii=False), flush=True)

    # ---- Showcase the user-facing options (Step 5): diet / must_use / diversity
    showcases = {
        "Italian-ish [vegetarian + must-use tomato]":
            (PANTRIES["Italian-ish"], dict(diet="vegetarian", must_use=["tomato"])),
        "Baker's pantry [diversity=0.6]":
            (PANTRIES["Baker's pantry"], dict(diversity=0.6)),
    }
    for name, (pantry, opts) in showcases.items():
        recs = rr.recommend(pantry, df, **opts)
        report["recommendations"][name] = {"ingredients": pantry, "options": opts,
                                           "top5": recs}
        print(f"\n--- {name} ---", flush=True)
        print(json.dumps(recs, indent=2, ensure_ascii=False), flush=True)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nWrote results.json, model_selection.json and models/lda_model.pkl",
          flush=True)


if __name__ == "__main__":
    main()
