# 流程图 / Pipeline flowchart

Two equivalent forms: a rendered **PNG** (for slides / any viewer) and an editable
**Mermaid** source (GitHub renders it inline). Same pipeline either way.

![Pipeline flowchart](pipeline_flowchart.png)

*Regenerate the PNG with `python make_flowchart.py` (run from the repo root).*

---

## Editable source (Mermaid)

```mermaid
flowchart LR
    subgraph OFF["OFFLINE · build the model / 构建模型"]
      direction LR
      A["Food.com raw data<br/>RAW_recipes + RAW_interactions"]
      A -->|"prepare_data.py · clean / normalize"| B["recipes_clean.csv<br/>53,573 recipes"]
      B --> C["<b>Step 1 · train_lda</b> — Bayesian LDA (NUTS)<br/>phi~Dir(0.01), theta~Dir(0.1), w~Cat(theta·phi) — z marginalized<br/>hold out 15% tokens → choose K by held-out lppd<br/>refit best K on full corpus"]
      C --> D["phi posterior samples (S×K×V)<br/>→ <b>lda_model.pkl</b> (final model)"]
    end
    subgraph ON["ONLINE · answer a query / 回答查询"]
      direction LR
      P["your pantry<br/>chicken, garlic, tomato, …"]
      F["<b>Step 2 · filter_candidates</b><br/>coverage = |U ∩ R| / |R| ≥ 0.7"]
      G["<b>Step 3 · infer_user_posterior</b><br/>P(topic | ingredients) per phi-sample"]
      H["<b>Step 4 · score_recipes</b><br/>coverage² · overlap_bonus · flavor_alignment · rating<br/>per-sample → posterior_uncertainty"]
      I["<b>Step 5 · recommend</b> — Top-N<br/>+ diet / must_use / diversity"]
      J["Top-N recommendations<br/>with uncertainty"]
      P --> F --> G --> H --> I --> J
    end
    D -->|"phi posterior used in steps 3–4"| G

    classDef model fill:#fff3cd,stroke:#d9a300,stroke-width:2px;
    classDef out fill:#c9ead1,stroke:#1c7a3e,stroke-width:2px;
    class D model;
    class J out;
```
