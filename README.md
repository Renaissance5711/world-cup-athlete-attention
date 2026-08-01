# World Cup athlete attention replication archive

This repository contains the code, data derivatives, tests and model outputs used to reproduce analyses of athlete attention following performances at the 2022 FIFA World Cup.

## Scope

The repository supports:

- player-match proportional and additive attention estimates;
- player-day persistence models;
- opportunity-adjusted shooter comparisons;
- shooter and goalkeeper relational analyses;
- observed scorer rankings, full-ranking reversal decomposition and bootstrap uncertainty;
- overlapping-window and structural-zero sensitivity analyses.

The central measurement distinction is between proportional pageview response and additional pageviews. These are treated as different selection rules for identifying athletes for subsequent evaluation, not as a new mathematical identity.

## Repository structure

- `base_archive/`: core processing code, licensed Fjelstul and Wikimedia inputs, processed panels, metadata and outputs;
- `code/`: ranking and sensitivity scripts;
- `data/raw/news/`: Google News-indexed metadata and links only; no article bodies;
- `data/raw/statsbomb/manifest.json`: audit metadata for the StatsBomb retrieval;
- `ijsms/`: shot-opportunity and goalkeeper analysis modules;
- `outputs/`: model, ranking and sensitivity outputs;
- `docs/`: analysis design, source documentation and data dictionaries;
- `tests/`: automated tests.

The large StatsBomb event and lineup JSON files are not committed. They can be retrieved from the official StatsBomb Open Data repository using `ijsms.statsbomb_acquisition.download_statsbomb_world_cup`.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Retrieve StatsBomb Open Data

```bash
python - <<'PY'
from pathlib import Path
from ijsms.statsbomb_acquisition import download_statsbomb_world_cup

download_statsbomb_world_cup(Path("data/raw/statsbomb/combined"))
PY
```

## Reproduce analyses

```bash
python base_archive/code/run_full_pipeline.py --skip-download
python run_near_miss_pipeline.py
python run_goalkeeper_pipeline.py
python code/run_fitted_ranking_analysis.py
python code/run_sensitivity_analysis.py
python code/run_observed_ranking_bootstrap.py
python code/run_rank_reversal_decomposition.py
```

## Rank-reversal audit outputs

The compact versioned outputs are `outputs/r25/rank_reversal_summary.json`, `outputs/r25/rank_reversal_decomposition_table.csv`, and `outputs/r25/rank_reversal_athlete_displacements.csv`. Running `code/run_rank_reversal_decomposition.py` also creates the complete pair-level file and all 10,000 bootstrap draws locally; those larger deterministic derivatives are intentionally not versioned.

## Verify

```bash
python -m pytest -q -W error::FutureWarning
sha256sum -c SHA256SUMS_PUBLIC.txt
sha256sum -c outputs/r25/SHA256SUMS_R25.txt
```

## Data and licensing

See `LICENSE.md`, `LICENSE_AND_THIRD_PARTY_TERMS.md`, `base_archive/docs/SOURCE_MANIFEST.md` and `docs/SOURCE_MANIFEST_V1.4.md`. Third-party terms remain controlling for their respective components.

## Manuscript files

No manuscript, tables, figures or submission package is included in this public repository. This separation is intentional to preserve the integrity of anonymous journal review.
