# World Cup athlete attention replication materials

This public repository contains the principal analysis code, documentation, selected automated tests, and compact derived outputs for a study of athlete attention following performances at the 2022 FIFA World Cup.

## Research scope

The analyses distinguish four quantities that should not be treated as interchangeable:

1. proportional change in athlete attention;
2. additional pageview volume;
3. persistence of attention after the event; and
4. relative attention across shooter and goalkeeper roles.

The mathematical distinction between proportional and additive change is not presented as a new mathematical result. The empirical contribution is to show that these estimands operate as different athlete-selection rules, generate different observed priority lists, and have temporal and relational boundaries.

## Repository contents

- `ijsms/`: shot, shot-opportunity, goalkeeper, headline-audit, and workflow modules.
- `code/`: ranking and sensitivity scripts.
- `tests/`: selected synthetic and mapping tests for the public modules.
- `outputs/`: compact diagnostic, ranking, model, and sensitivity results used to audit the reported findings.
- `docs/`: analysis design, data dictionaries, and source documentation.
- `base_archive/`: the core pageview-pipeline entry point and licensing notes.

The anonymous manuscript, submission tables, submission figures, compressed archives, large raw event files, full processed pageview panels, and Google News-indexed record files are intentionally excluded from this public repository.

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

The full workflows require source and processed inputs described in `docs/ANALYSIS_DESIGN.md`, `docs/NEAR_MISS_DATA_DICTIONARY.md`, `docs/GOALKEEPER_DATA_DICTIONARY.md`, and `docs/SOURCE_MANIFEST_V1.4.md`. StatsBomb acquisition and validation logic is provided in `ijsms/statsbomb_acquisition.py`.

## Tests

The included unit tests can be run with:

```bash
python -m pytest tests/test_expected_save.py tests/test_statsbomb_mapping.py -q
```

Additional integration tests require the excluded raw and processed datasets.

## Key auditable results

- `outputs/r24/observed_ranking_bootstrap_summary.json`: 117 scorers, observed Spearman correlation, and bootstrap intervals.
- `outputs/r24/observed_ranking_topk_sensitivity.csv`: top-list overlap across 5%, 10%, 15%, and 20% thresholds.
- `outputs/near_miss_primary_results.csv`: shot-opportunity-adjusted proportional and additive results.
- `outputs/goalkeeper/goalkeeper_outcome_gradient_results.csv`: primary single-shot shooter-goalkeeper results.
- `outputs/goalkeeper/expected_save_diagnostics.json`: cross-fitted expected-save performance and leakage checks.

## Data and licensing boundaries

Some workflows retrieve public data from Wikimedia, Fjelstul, and StatsBomb. Review the source manifests and the current third-party terms before redistribution. `LICENSE_AND_THIRD_PARTY_TERMS.md` summarizes the archive boundaries but does not replace the original providers' terms.

## Reproducibility note

The compact outputs allow reported sample sizes, coefficient files, ranking summaries, and sensitivity results to be inspected without publishing the largest source archives. Re-running every workflow requires network access, source retrieval, and additional disk space.
