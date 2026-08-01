# World Cup athlete attention replication materials

This public repository contains code, tests, documentation, selected public-source inputs, and derived outputs for a study of athlete attention following performances at the 2022 FIFA World Cup.

## Research scope

The analyses distinguish four quantities that should not be treated as interchangeable:

1. proportional change in athlete attention;
2. additional pageview volume;
3. persistence of attention after the event; and
4. relative attention across shooter and goalkeeper roles.

The mathematical distinction between proportional and additive change is not presented as a new mathematical result. The empirical contribution is to show that these estimands operate as different athlete-selection rules, generate different observed priority lists, and have temporal and relational boundaries.

## Repository contents

- `ijsms/`: shot, near-miss, goalkeeper, and workflow modules.
- `code/`: ranking and sensitivity scripts.
- `tests/`: automated tests for the included workflows.
- `outputs/`: selected derived model, ranking, and sensitivity outputs.
- `docs/`: analysis design, data dictionaries, and source documentation.
- `base_archive/`: core pageview pipeline code, tests, metadata, selected source tables, and published model outputs.

The anonymous manuscript, submission tables, submission figures, and compressed archives are intentionally excluded from this public repository.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Main workflows

```bash
python base_archive/code/run_full_pipeline.py
python run_near_miss_pipeline.py
python run_goalkeeper_pipeline.py
python code/run_fitted_ranking_analysis.py
python code/run_sensitivity_analysis.py
python code/run_observed_ranking_bootstrap.py
```

Some workflows download public data from Wikimedia and StatsBomb. Review the source manifests and current third-party terms before redistribution.

## Tests

```bash
python -m pytest -q
```

## Data and licensing boundaries

The repository does not include the large StatsBomb event and lineup archive, the full processed Wikimedia panels, or Google News-indexed records. The code and manifests document how those inputs were acquired and processed. Third-party terms are summarized in `LICENSE_AND_THIRD_PARTY_TERMS.md` and the source manifests. Those summaries do not replace the original providers' terms.

## Reproducibility note

The repository contains selected derived outputs so that reported sample sizes, coefficient files, ranking summaries, and sensitivity results can be inspected without downloading the largest source archives. Re-running every workflow may require network access and substantial disk space.
