# Sample images

Example inputs for `run_pipeline.py` (the detector is for **raw / whole ingredients** — the
fridge/counter use case — not plated dishes).

| file | contents | detects |
|------|----------|---------|
| `fridge.jpg` | a 5-ingredient "fridge shelf" (composite of the singles below) | garlic, broccoli, cherry tomato, egg, onion |
| `tomato.jpg` | a tomato | Tomato |
| `broccoli.jpg` | broccoli | Broccoli |
| `garlic.jpg` | garlic | Garlic |
| `egg.jpg` | an egg | Egg |
| `onion.jpg` | an onion | Onion |
| `vegetables.jpg` | a 6-item produce mix (composite) | carrot, cherry tomato, garlic, broccoli, lemon, onion |

Try it:
```bash
python run_pipeline.py --image samples/fridge.jpg          # 5 ingredients -> rich Top-5
python run_pipeline.py --image samples/tomato.jpg          # single ingredient
```

Source: photos downloaded from **Wikimedia Commons** (CC / public-domain), resized for the repo;
`fridge.jpg` is a grid composite of `tomato/broccoli/garlic/egg/onion.jpg`.
