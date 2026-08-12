# TEM COMPOT Two-Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Stage A COMPOT–firm-participation analysis and a Stage B 400-project COMPOT moderation/conflict validation that reuses the successful pilot outputs before the 1,881-project confirmatory expansion.

**Architecture:** Keep the validated realization pilot untouched. Add focused overlay modules for project-level Stage A, candidate-level COMPOT ranking moderation, and COMPOT conflict heterogeneity; orchestrate them with a lightweight validation runner that reads the verified V3 inputs plus the already-successful 400-project pilot artifact. A dedicated GitHub Actions workflow will run tests first, then download only existing artifacts—no new OpenAlex extraction.

**Tech Stack:** Python 3.11, pandas, numpy, statsmodels, pytest, GitHub Actions.

## Global Constraints

- Stage A population is exactly 6,536 unique strict-eligible projects; positive `firm_participation` outcomes must be exactly 1,881.
- Stage A inference is associational and conditional on the V3 strict-eligible sample.
- COMPOT must not enter project-stratified conditional logit as a standalone main effect.
- Stage B COMPOT scaling must be fit on training-period projects only (`publication_year <= 2018`).
- Preserve the existing 400-project sample and baseline B0 behavior.
- Top 50 is main; Top 100 is sensitivity.
- No new OpenAlex extraction is allowed for COMPOT validation when the successful 400-project artifact is reusable.
- Null COMPOT effects are valid outcomes and must not fail the software gate.

---

### Task 1: Stage A project-level COMPOT analysis

**Files:**
- Create: `.github/overlays/stage2_compot_stagea.py`
- Create: `.github/overlays/test_stage2_compot_stagea.py`

**Interfaces:**
- Consumes: strict author-project panel with `work_id`, `compot`, `firm_participation`, `publication_year`, and a field identifier (`primary_field_id` preferred, else `primary_subfield_id`).
- Produces: `build_stagea_project_panel(strict_panel) -> (project_panel, audit)`, `build_stagea_descriptives(project_panel) -> DataFrame`, and `fit_stagea_compot_models(project_panel) -> DataFrame`.

- [ ] **Step 1: Write failing Stage A tests**

```python
import pandas as pd
from stage2_compot_stagea import build_stagea_project_panel, build_stagea_descriptives


def test_stagea_collapses_duplicate_author_rows_without_changing_project_fields():
    strict = pd.DataFrame({
        "work_id": ["W1", "W1", "W2"],
        "compot": [0.2, 0.2, 0.8],
        "firm_participation": [0, 0, 1],
        "publication_year": [2019, 2019, 2020],
        "primary_field_id": ["F1", "F1", "F2"],
    })
    panel, audit = build_stagea_project_panel(strict, expected_projects=None, expected_positive=None)
    assert panel["work_id"].tolist() == ["W1", "W2"]
    assert audit["projects"] == 2
    assert audit["positive_projects"] == 1


def test_stagea_rejects_within_project_compot_disagreement():
    strict = pd.DataFrame({
        "work_id": ["W1", "W1"],
        "compot": [0.2, 0.3],
        "firm_participation": [1, 1],
        "publication_year": [2019, 2019],
        "primary_field_id": ["F1", "F1"],
    })
    try:
        build_stagea_project_panel(strict, expected_projects=None, expected_positive=None)
    except ValueError as exc:
        assert "within-project" in str(exc).lower()
    else:
        raise AssertionError("Expected a within-project consistency error")
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests_v5/test_stage2_compot_stagea.py`
Expected: FAIL because `stage2_compot_stagea` does not exist.

- [ ] **Step 3: Implement minimal Stage A collapse/audit/descriptives/models**

`build_stagea_project_panel` must fail on missing COMPOT, duplicates after collapse, inconsistent project fields, and fixed-count mismatches. `fit_stagea_compot_models` must emit A1 LPM and A2 logit coefficient rows for `compot_z`, plus A3 quartile coefficients, with year and field controls and HC1 robust errors.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests_v5/test_stage2_compot_stagea.py`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `Add Stage A COMPOT participation analysis`.

---

### Task 2: Stage B COMPOT temporal ranking moderation

**Files:**
- Create: `.github/overlays/stage2_compot_models.py`
- Create: `.github/overlays/test_stage2_compot_models.py`
- Reuse without changing behavior: `.github/overlays/stage2_pilot_models.py`

**Interfaces:**
- Consumes: cognitive candidate-long panel already produced by the successful pilot, including `compot` and all existing baseline fields.
- Produces: `fit_compot_temporal_ranking_models(candidate_long, train_end_year=2018, ...) -> (coefficients, metrics)` with models `B0_combined`, `B1_relationship_compot`, and `B2_fit_compot`.

- [ ] **Step 1: Write failing moderation tests**

```python
from stage2_compot_models import prepare_compot_features, fit_compot_temporal_ranking_models
from test_stage2_pilot_models import make_candidate_panel


def test_compot_scaling_uses_training_projects_only():
    frame = make_candidate_panel()
    frame["compot"] = frame["publication_year"].map(lambda y: 1.0 if y <= 2018 else 100.0)
    prepared = prepare_compot_features(frame, train_end_year=2018)
    train = prepared[prepared["publication_year"].le(2018)]
    assert abs(train.groupby("work_id")["compot_z"].first().mean()) < 1e-9
    assert prepared.loc[prepared["publication_year"].gt(2018), "compot_z"].min() > 10


def test_compot_models_never_include_compot_main_effect():
    frame = make_candidate_panel()
    frame["compot"] = frame["work_id"].str.extract(r"_(\d+)$")[0].astype(float)
    coefficients, _ = fit_compot_temporal_ranking_models(frame, train_end_year=2018, bootstrap_reps=5)
    assert "compot_z" not in set(coefficients["term"])
    assert set(coefficients["model"]) == {"B0_combined", "B1_relationship_compot", "B2_fit_compot"}
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests_v5/test_stage2_compot_models.py`
Expected: FAIL because the COMPOT model module does not exist.

- [ ] **Step 3: Implement minimal B0/B1/B2 model pipeline**

Reuse the validated helper functions from `stage2_pilot_models`. Standardize COMPOT from one row per training project, map it back to candidate rows, construct `author_x_compot`, `university_x_compot`, `strong_university_x_compot`, and `fit_x_compot`, and preserve the same temporal eligibility/ranking/bootstrapping logic as the baseline.

- [ ] **Step 4: Add baseline-equivalence regression test**

For a deterministic synthetic panel, compare B0 aggregate ranking metrics against the existing `combined` model from `fit_temporal_ranking_models`; require equality within floating-point tolerance.

- [ ] **Step 5: Run GREEN and legacy regression suite**

Run: `pytest -q tests_v5/test_stage2_compot_models.py tests_v5/test_stage2_pilot_models.py`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `Add COMPOT temporal ranking moderation`.

---

### Task 3: COMPOT conflict heterogeneity

**Files:**
- Create: `.github/overlays/stage2_compot_conflict.py`
- Create: `.github/overlays/test_stage2_compot_conflict.py`

**Interfaces:**
- Consumes: cognitive candidate-long panel.
- Produces: `summarize_compot_conflict(candidate_long, thresholds=(0.02, 0.05, 0.10)) -> DataFrame`, one row per COMPOT quartile with project counts, cognitive coverage, conflict shares, fit shortfall, fit percentile, and no-top-decile-selected share.

- [ ] **Step 1: Write failing conflict tests**

Construct four projects with deterministic COMPOT ordering and one embedded selected firm plus a higher-fit unembedded alternative in selected quartiles. Assert quartile assignment is project-level, conflict thresholds are monotone (`share_0.02 >= share_0.05 >= share_0.10`), and missing candidate-level COMPOT fails.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests_v5/test_stage2_compot_conflict.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement conflict summaries**

Define embedded as author or university prior relationship. Use natural candidates where available (`forced_selected_candidate == 0`) for the higher-fit unembedded alternative; if the flag is absent in synthetic tests, treat all rows as natural. Assign COMPOT quartiles from unique projects, not candidate rows.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests_v5/test_stage2_compot_conflict.py`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `Add COMPOT conflict heterogeneity summaries`.

---

### Task 4: Validation runner and no-OpenAlex workflow

**Files:**
- Create: `.github/overlays/run_stage2_compot_validation.py`
- Create: `.github/overlays/test_stage2_compot_validation_runner.py`
- Create: `.github/workflows/tem-stage2-compot-validation.yml`

**Interfaces:**
- Consumes: verified V3 artifact `8828825689` and successful pilot artifact `9041032423` from run `31318163790`.
- Produces Stage A outputs plus Top50/Top100 COMPOT coefficients, ranking metrics, conflict quartile tables, and `pilot_compot_validation_summary.json`.

- [ ] **Step 1: Write failing runner output-contract test**

Assert `required_compot_output_filenames()` includes all Stage A outputs, Top50/Top100 Stage B outputs, and the combined summary.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests_v5/test_stage2_compot_validation_runner.py`
Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement runner**

The runner reads `author_project_panel_v3_strict.csv`, `risk_set_full_firm_projects_input_v3.csv`, and the existing `cognitive_candidate_long_top50.csv` / `top100.csv`; it writes all output-contract files and a classification that permits null results. It verifies 6,536 Stage A projects, 1,881 positives, exactly 400 pilot projects in each cognitive file, no missing COMPOT, and matching project sets across Top50/Top100.

- [ ] **Step 4: Implement workflow**

The workflow checks out the branch, sets up Python, downloads the small V3 input artifact and successful pilot artifact, extracts only the strict/project inputs plus the two cognitive candidate CSVs, copies the four COMPOT overlay modules and tests into a runtime directory, installs requirements, runs all tests, executes the validation runner, and uploads only the compact COMPOT outputs. It must contain no OpenAlex API calls and no OPENALEX_API_KEY requirement.

- [ ] **Step 5: Run RED/GREEN through GitHub Actions**

First commit tests/output contract before production modules where applicable and verify the workflow/test failure is caused by missing implementation. Then commit production files and require the dedicated `TEM Stage 2 COMPOT Validation` workflow to complete successfully.

- [ ] **Step 6: Inspect artifact results**

Read `stageA_summary.json` and `pilot_compot_validation_summary.json`. Report Stage A COMPOT association, B1/B2 moderation coefficients/ranking deltas, COMPOT conflict heterogeneity, Top50/Top100 directional stability, and the resulting specification-freeze classification.

- [ ] **Step 7: Commit**

Commit message: `Run TEM COMPOT validation without OpenAlex`.

---

### Task 5: Full-suite verification and freeze decision

**Files:**
- No production changes unless verification exposes a defect.

- [ ] **Step 1: Run the complete reconstructed test suite**

Run: `pytest -q tests_v5`
Expected: all tests PASS.

- [ ] **Step 2: Verify no baseline regression**

Confirm the original 400-pilot gate outputs and existing temporal-model tests remain unchanged; the dedicated COMPOT workflow must not trigger the expensive OpenAlex pilot workflow.

- [ ] **Step 3: Freeze or revise specification**

If software checks pass, classify COMPOT as Stage A only, Stage A + relationship moderator, Stage A + fit moderator, Stage A + both moderators, or neither substantive signal. Null findings are valid. Freeze B0/B1/B2 definitions before writing the sharded 1,881 workflow.
