"""
run_demo.py
===========
End-to-end demo of the Bayesian LDA recipe recommender on the cleaned Food.com
data. Trains the model (NUTS via nutpie), reports the WAIC model-selection table
and convergence diagnostics, then prints Top-5 recommendations for a few pantries.

All results are also written to `results.json` (so nothing is lost to terminal
truncation) and the fitted posterior is saved to `models/lda_model.pkl`.
"""

import json

import arviz as az
import numpy as np
import pandas as pd

import recipe_recommender as rr

DATA = "data/recipes_clean.csv"

PANTRIES = {
    "Italian-ish": ["chicken", "garlic", "onion", "olive oil", "tomato",
                    "basil", "parmesan cheese", "pasta", "salt", "pepper"],
    "Baker's pantry": ["flour", "sugar", "butter", "egg", "vanilla",
                       "baking soda", "milk", "salt", "chocolate"],
}


def main():
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} recipes\n", flush=True)

    # ---- Step 1: train Bayesian LDA at a single, directly-chosen K ------------
    # We skip the K-sweep and go straight to K=6: a good compromise between the
    # held-out-optimal (very few topics on this small corpus) and the richer
    # flavor_tags we want. (The held-out lppd / WAIC / LOO are still reported for
    # this K as diagnostics.) NOTE on scale: NUTS cost explodes super-linearly in
    # the number of training *documents*, so n_train stays ~400 (the fast regime;
    # n_train=1200 did not finish). Vocabulary is cheap, so we widen that instead.
    model = rr.train_lda(df, k_values=(6,), n_train=400,
                         vocab_size=100, draws=300, tune=400, chains=2,
                         seed=42)                                # nutpie
    rr.save_model(model, "models/lda_model.pkl")
    rr._STATE = {"model": model, "df_id": id(df)}      # reuse for recommend()

    # ---- Convergence diagnostics (within-chain ESS of the kept chain) ---------
    # NOTE: cross-chain r-hat/ESS for phi/theta is meaningless under LDA label
    # switching; train_lda reports az.ess on a SINGLE chain (the one we keep).
    ess_min, ess_med = model.ess_min, model.ess_median

    report = {
        "selected_K": model.best_k,
        "selection_metric": "K chosen directly; held-out lppd / WAIC / LOO are diagnostics",
        "model_comparison": model.waic_table.round(3).to_dict(orient="records"),
        "topics": [{"topic": k, "label": lab,
                    "P(topic)": round(float(model.topic_prior[k]), 4)}
                   for k, lab in enumerate(model.topic_labels)],
        "ess_phi_chain0": {"min": round(ess_min, 1), "median": round(ess_med, 1)},
        "recommendations": {},
    }

    print(f"\n=== Model diagnostics (K={model.best_k}, chosen directly) ===", flush=True)
    print("(held-out predictive lppd / WAIC / PSIS-LOO for the chosen K)", flush=True)
    print(model.waic_table.to_string(index=False), flush=True)
    print(f"ESS(phi, chain 0): min={ess_min:.0f}, median={ess_med:.0f}", flush=True)
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
    print("\nWrote results.json and models/lda_model.pkl", flush=True)


if __name__ == "__main__":
    main()
