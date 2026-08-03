# Stage 2 Potential-Match Realization Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic 400-project pilot that tests whether the “Potential Matches, Realized Partners” paper is empirically viable before expanding to all 1,881 projects.

**Architecture:** Reuse the existing reconstructed Stage 2 code and restartable OpenAlex cache, but insert a pilot-only sampling layer before manifest construction. Fetch the top 100 field-active firms once, derive Top 50 and Top 100 candidate specifications from the same extraction, restrict university-generated candidates to strong ties with at least five distinct prepublication joint works, append uncaptured realized firms only after natural recall is measured, compute TF–IDF cognitive fit, estimate technical/relational/combined ranking models, and write a machine-readable four-gate Go/No-Go report.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, statsmodels, SciPy, pytest, GitHub Actions, OpenAlex API, restartable GitHub Actions cache.

## Global Constraints

- Pilot size is exactly 400 unique projects unless the source contains fewer than 400 eligible projects.
- Sampling seed is `20260804`.
- Sampling is stratified by publication period, primary field, and within-period COMPOT quartile.
- Main field-active candidate specification is Top 50; Top 100 is the prespecified sensitivity specification.
- All author-prior firms enter the natural candidate set.
- University-generated candidates require at least five distinct university–firm works in the exact five-year prepublication window.
- Any university prior relationship remains an attribute for candidates even when it does not generate candidacy.
- Actual realized firms are appended only after natural recall is calculated and must be marked `forced_selected_candidate = 1`.
- Every relationship, field-activity, and cognitive-fit input must be strictly earlier than the focal project publication date.
- A candidate has measurable cognitive fit when the focal project text is nonempty and `cognitive_fit_publication_count > 0`.
- Main conflict threshold is an unembedded candidate cognitive-fit advantage of at least `0.05`; robustness thresholds are `0.02` and `0.10`.
- The four pilot gates are: natural project recall ≥80%; selected-firm cognitive-fit coverage ≥80%; conflict projects ≥10%; combined model improves selected-partner ranking relative to the technical model.
- The pilot is observational and must not use causal language.
- Do not call the candidate set a directly observed managerial consideration set.
- Do not expose or write the OpenAlex API key to files or logs.
- The existing draft PR remains unmerged.

---

## File Structure

Create these focused pilot files on branch `tem-stage2-v5-run`:

- `.github/overlays/stage2_pilot_sampling.py` — deterministic stratified project sampling.
- `.github/overlays/stage2_pilot_risk_set.py` — natural candidate construction, university strength threshold, realized-firm injection, and Top-N derivation.
- `.github/overlays/stage2_pilot_metrics.py` — cognitive-fit coverage, conflict-set metrics, realization-gap outcomes, and four-gate decision.
- `.github/overlays/stage2_pilot_models.py` — temporal technical, relational, and combined ranking models and project-level ranking metrics.
- `.github/overlays/run_stage2_realization_pilot.py` — pilot orchestration using existing extraction and cognitive-fit functions.
- `.github/overlays/test_stage2_pilot_sampling.py`
- `.github/overlays/test_stage2_pilot_risk_set.py`
- `.github/overlays/test_stage2_pilot_metrics.py`
- `.github/overlays/test_stage2_pilot_models.py`
- `.github/workflows/tem-stage2-realization-pilot.yml` — isolated pilot workflow and artifact upload.

Do not modify the base64 code payload. The workflow copies pilot overlays into the reconstructed runtime code.

---

### Task 1: Deterministic Stratified Pilot Sample

**Files:**
- Create: `.github/overlays/stage2_pilot_sampling.py`
- Create: `.github/overlays/test_stage2_pilot_sampling.py`

**Interfaces:**
- Consumes: the 1,881-row project input with `work_id`, `publication_year`, `primary_field_id`, and `compot`.
- Produces: `select_stratified_pilot_projects(projects: pd.DataFrame, sample_size: int = 400, seed: int = 20260804) -> tuple[pd.DataFrame, pd.DataFrame]`.
- The first return value is the sampled project frame.
- The second return value is a stratum audit with population, target, and sampled counts.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
from stage2_pilot_sampling import select_stratified_pilot_projects


def make_projects() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2025):
        for field in [10, 20, 30, 40]:
            for index in range(8):
                rows.append(
                    {
                        "work_id": f"W{year}_{field}_{index}",
                        "publication_year": year,
                        "primary_field_id": field,
                        "compot": (index + 1) / 9,
                    }
                )
    return pd.DataFrame(rows)


def test_stratified_sample_is_deterministic_and_unique():
    projects = make_projects()
    first, first_audit = select_stratified_pilot_projects(
        projects, sample_size=400, seed=20260804
    )
    second, second_audit = select_stratified_pilot_projects(
        projects, sample_size=400, seed=20260804
    )
    assert first["work_id"].tolist() == second["work_id"].tolist()
    assert first["work_id"].is_unique
    assert len(first) == 400
    pd.testing.assert_frame_equal(first_audit, second_audit)


def test_stratified_sample_preserves_all_nonempty_macro_periods():
    projects = make_projects()
    sample, audit = select_stratified_pilot_projects(projects, sample_size=400)
    assert set(sample["pilot_period"]) == {
        "2000-2006",
        "2007-2012",
        "2013-2018",
        "2019-2024",
    }
    assert audit["sampled_n"].sum() == 400
    assert audit.loc[audit["population_n"].gt(0), "sampled_n"].gt(0).all()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q .github/overlays/test_stage2_pilot_sampling.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'stage2_pilot_sampling'`.

- [ ] **Step 3: Implement the minimal sampler**

```python
from __future__ import annotations

import numpy as np
import pandas as pd


PERIOD_BINS = [1999, 2006, 2012, 2018, 2024]
PERIOD_LABELS = ["2000-2006", "2007-2012", "2013-2018", "2019-2024"]


def _largest_remainder_allocation(
    counts: pd.Series, sample_size: int
) -> pd.Series:
    raw = counts / counts.sum() * sample_size
    allocated = np.floor(raw).astype(int)
    allocated[counts.gt(0) & allocated.eq(0)] = 1
    excess = int(allocated.sum() - sample_size)
    if excess > 0:
        removable = (allocated - 1).clip(lower=0)
        for key in removable.sort_values(ascending=False).index:
            take = min(excess, int(removable.loc[key]))
            allocated.loc[key] -= take
            excess -= take
            if excess == 0:
                break
    deficit = int(sample_size - allocated.sum())
    if deficit > 0:
        remainder = (raw - np.floor(raw)).sort_values(ascending=False)
        for key in remainder.index:
            room = int(counts.loc[key] - allocated.loc[key])
            take = min(deficit, room)
            allocated.loc[key] += take
            deficit -= take
            if deficit == 0:
                break
    if int(allocated.sum()) != sample_size:
        raise ValueError("Could not allocate requested pilot sample")
    return allocated


def select_stratified_pilot_projects(
    projects: pd.DataFrame,
    sample_size: int = 400,
    seed: int = 20260804,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "work_id",
        "publication_year",
        "primary_field_id",
        "compot",
    }
    missing = sorted(required - set(projects.columns))
    if missing:
        raise ValueError(f"Missing sampling columns: {missing}")
    if not projects["work_id"].is_unique:
        raise ValueError("Pilot project input must contain unique work_id values")

    data = projects.copy()
    if len(data) <= sample_size:
        data["pilot_period"] = pd.cut(
            data["publication_year"],
            bins=PERIOD_BINS,
            labels=PERIOD_LABELS,
        ).astype(str)
        data["compot_quartile"] = (
            data.groupby("pilot_period", observed=True)["compot"]
            .transform(
                lambda values: pd.qcut(
                    values.rank(method="first"),
                    4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                )
            )
            .astype(str)
        )
        data["pilot_stratum"] = (
            data["pilot_period"]
            + "|"
            + data["primary_field_id"].astype(str)
            + "|"
            + data["compot_quartile"]
        )
        audit = (
            data.groupby("pilot_stratum", observed=True)
            .size()
            .rename("population_n")
            .reset_index()
        )
        audit["target_n"] = audit["population_n"]
        audit["sampled_n"] = audit["population_n"]
        return data.sort_values("work_id").reset_index(drop=True), audit

    data["pilot_period"] = pd.cut(
        data["publication_year"],
        bins=PERIOD_BINS,
        labels=PERIOD_LABELS,
    ).astype(str)
    data["compot_quartile"] = (
        data.groupby("pilot_period", observed=True)["compot"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                4,
                labels=["Q1", "Q2", "Q3", "Q4"],
            )
        )
        .astype(str)
    )
    data["pilot_stratum"] = (
        data["pilot_period"]
        + "|"
        + data["primary_field_id"].astype(str)
        + "|"
        + data["compot_quartile"]
    )

    population = data.groupby("pilot_stratum", observed=True).size()
    allocation = _largest_remainder_allocation(population, sample_size)
    rng = np.random.default_rng(seed)
    pieces = []
    for stratum, target_n in allocation.items():
        group = data[data["pilot_stratum"].eq(stratum)].copy()
        random_state = int(rng.integers(0, 2**31 - 1))
        pieces.append(group.sample(n=int(target_n), random_state=random_state))
    sample = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["pilot_period", "primary_field_id", "compot", "work_id"])
        .reset_index(drop=True)
    )

    audit = population.rename("population_n").reset_index()
    audit["target_n"] = audit["pilot_stratum"].map(allocation).astype(int)
    sampled_counts = sample.groupby("pilot_stratum", observed=True).size()
    audit["sampled_n"] = (
        audit["pilot_stratum"].map(sampled_counts).fillna(0).astype(int)
    )
    if len(sample) != sample_size or not sample["work_id"].is_unique:
        raise AssertionError("Pilot sampling contract failed")
    return sample, audit
```

- [ ] **Step 4: Run tests and verify GREEN**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_sampling.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add .github/overlays/stage2_pilot_sampling.py \
  .github/overlays/test_stage2_pilot_sampling.py
git commit -m "feat: add deterministic TEM pilot sampling"
```

---

### Task 2: Natural Candidate Set and Realized-Firm Injection

**Files:**
- Create: `.github/overlays/stage2_pilot_risk_set.py`
- Create: `.github/overlays/test_stage2_pilot_risk_set.py`

**Interfaces:**
- Consumes: sampled projects, project details, author history, university history, and Top 100 field counts.
- Produces:
  - `assemble_pilot_candidates(..., field_top_n: int, university_min_works: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `inject_realized_firms(natural_candidates, projects, details) -> pd.DataFrame`
- Natural candidates contain `natural_candidate = 1`, `forced_selected_candidate = 0`, `university_prior_work_count`, and `field_rank`.
- Injected rows contain `natural_candidate = 0`, `forced_selected_candidate = 1`.

- [ ] **Step 1: Write failing tests for strong-university generation**

```python
import pandas as pd
from stage2_pilot_risk_set import (
    assemble_pilot_candidates,
    inject_realized_firms,
)


def test_weak_university_tie_is_attribute_not_candidate_source():
    projects = pd.DataFrame(
        {
            "work_id": ["W1"],
            "publication_year": [2020],
            "publication_date": ["2020-06-01"],
            "primary_subfield_id": [100],
            "primary_field_id": [10],
            "compot": [0.5],
            "focal_author_ids": ["A1"],
        }
    )
    details = pd.DataFrame(
        {
            "work_id": ["W1"],
            "actual_company_ids": ["C9"],
            "focal_education_ids": ["U1"],
        }
    )
    author_history = pd.DataFrame(
        columns=["author_id", "company_id", "evidence_date", "prior_work_id"]
    )
    university_history = pd.DataFrame(
        {
            "university_id": ["U1"] * 6,
            "company_id": ["Cweak"] + ["Cstrong"] * 5,
            "evidence_date": ["2019-01-01"] * 6,
            "prior_work_id": ["P0", "P1", "P2", "P3", "P4", "P5"],
        }
    )
    field_counts = pd.DataFrame(
        {
            "subfield_id": [100],
            "as_of_date": ["2020-06-01"],
            "company_id": ["Cfield"],
            "prior_subfield_publication_count": [20],
            "field_rank": [1],
            "query_from_date": ["2015-06-01"],
            "query_to_date": ["2020-05-31"],
        }
    )

    natural, audit = assemble_pilot_candidates(
        projects,
        details,
        author_history,
        university_history,
        field_counts,
        field_top_n=50,
        university_min_works=5,
    )

    assert set(natural["company_id"]) == {"Cstrong", "Cfield"}
    strong = natural.set_index("company_id").loc["Cstrong"]
    assert strong["university_prior_partner"] == 1
    assert strong["university_prior_work_count"] == 5
    assert strong["strong_university_candidate"] == 1


def test_realized_firm_is_injected_after_natural_recall():
    natural = pd.DataFrame(
        {
            "work_id": ["W1", "W1"],
            "company_id": ["C1", "C2"],
            "selected": [0, 0],
            "natural_candidate": [1, 1],
            "forced_selected_candidate": [0, 0],
        }
    )
    projects = pd.DataFrame(
        {
            "work_id": ["W1"],
            "publication_year": [2020],
            "publication_date": ["2020-06-01"],
            "primary_subfield_id": [100],
            "primary_field_id": [10],
            "compot": [0.5],
            "focal_author_ids": ["A1"],
        }
    )
    details = pd.DataFrame(
        {
            "work_id": ["W1"],
            "actual_company_ids": ["C9"],
            "focal_education_ids": ["U1"],
        }
    )
    estimation = inject_realized_firms(natural, projects, details)
    injected = estimation.set_index("company_id").loc["C9"]
    assert injected["selected"] == 1
    assert injected["natural_candidate"] == 0
    assert injected["forced_selected_candidate"] == 1
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_risk_set.py
```

Expected: import failure.

- [ ] **Step 3: Implement candidate construction**

Implementation requirements:

```python
def assemble_pilot_candidates(
    projects: pd.DataFrame,
    project_details: pd.DataFrame,
    author_history: pd.DataFrame,
    university_history: pd.DataFrame,
    field_counts: pd.DataFrame,
    *,
    field_top_n: int,
    university_min_works: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

For each project:

1. Use `[publication date - 5 years, publication date)`.
2. Add all author-prior firms.
3. Aggregate university history by `company_id` using distinct `prior_work_id`.
4. Set `university_prior_partner = 1` for any university relationship.
5. Add the firm as a candidate only when `university_prior_work_count >= university_min_works`.
6. Add field firms where `field_rank <= field_top_n`.
7. Calculate natural recall before injection.
8. Do not add actual firms in this function.
9. Output one row per `work_id × company_id`.

Every natural row must include:

```python
{
    "author_prior_partner": int,
    "university_prior_partner": int,
    "university_prior_work_count": int,
    "strong_university_candidate": int,
    "subfield_active_company": int,
    "field_rank": int | None,
    "prior_subfield_publication_count": int,
    "selected": int,
    "natural_candidate": 1,
    "forced_selected_candidate": 0,
}
```

`inject_realized_firms` must preserve all existing candidate attributes and add missing realized firms with zero relationship/field attributes. The later orchestration task will enrich injected firms with any available author/university history before cognitive extraction.

- [ ] **Step 4: Run risk-set tests**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_risk_set.py
```

Expected: `2 passed`.

- [ ] **Step 5: Add field-rank support to the runtime extraction overlay**

Modify `.github/overlays/stage2_openalex_extract.py` inside `fetch_exact_field_counts` so that selected field firms are sorted by:

```python
[
    "prior_subfield_publication_count",
    "company_id",
]
```

with count descending and company ID ascending, then assign:

```python
selected["field_rank"] = np.arange(1, len(selected) + 1)
```

Add `field_rank` to output columns.

Add a failing regression test to `.github/overlays/test_cache_compaction.py`:

```python
def test_fetch_exact_field_counts_emits_deterministic_field_rank(...):
    ...
    assert list(out.sort_values("field_rank")["field_rank"]) == [1, 2]
```

Run:

```bash
PYTHONPATH=runtime/code pytest -q \
  runtime/code/tests_v5/test_cache_compaction.py
```

- [ ] **Step 6: Commit**

```bash
git add .github/overlays/stage2_pilot_risk_set.py \
  .github/overlays/test_stage2_pilot_risk_set.py \
  .github/overlays/stage2_openalex_extract.py \
  .github/overlays/test_cache_compaction.py
git commit -m "feat: build natural pilot candidate sets"
```

---

### Task 3: Cognitive-Fit Coverage, Conflict Sets, and Realization Gap

**Files:**
- Create: `.github/overlays/stage2_pilot_metrics.py`
- Create: `.github/overlays/test_stage2_pilot_metrics.py`

**Interfaces:**
- Consumes: estimation candidates after cognitive-fit merge.
- Produces:
  - `build_project_realization_metrics(candidate_long, conflict_threshold=0.05) -> pd.DataFrame`
  - `evaluate_pilot_gates(candidate_long, ranking_comparison, project_metrics) -> dict[str, object]`

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd
from stage2_pilot_metrics import (
    build_project_realization_metrics,
    evaluate_pilot_gates,
)


def test_conflict_requires_embedded_selected_and_higher_fit_unembedded_candidate():
    frame = pd.DataFrame(
        [
            {
                "work_id": "W1",
                "company_id": "Cselected",
                "selected": 1,
                "natural_candidate": 1,
                "forced_selected_candidate": 0,
                "author_prior_partner": 1,
                "strong_university_candidate": 0,
                "cognitive_fit_cosine": 0.30,
                "cognitive_fit_publication_count": 4,
            },
            {
                "work_id": "W1",
                "company_id": "Coutside",
                "selected": 0,
                "natural_candidate": 1,
                "forced_selected_candidate": 0,
                "author_prior_partner": 0,
                "strong_university_candidate": 0,
                "cognitive_fit_cosine": 0.42,
                "cognitive_fit_publication_count": 3,
            },
        ]
    )
    out = build_project_realization_metrics(frame, conflict_threshold=0.05)
    row = out.iloc[0]
    assert row["relation_fit_conflict"] == 1
    assert abs(row["fit_shortfall"] - 0.12) < 1e-9
    assert row["selected_fit_percentile"] == 0.5


def test_gate_decision_requires_three_of_four_passes():
    candidate_long = pd.DataFrame(
        {
            "work_id": ["W1", "W2"],
            "selected": [1, 1],
            "natural_candidate": [1, 1],
            "cognitive_fit_publication_count": [2, 2],
        }
    )
    ranking = pd.DataFrame(
        {
            "model": ["technical", "combined"],
            "mean_reciprocal_rank": [0.50, 0.60],
        }
    )
    projects = pd.DataFrame(
        {
            "work_id": ["W1", "W2"],
            "relation_fit_conflict": [1, 0],
        }
    )
    decision = evaluate_pilot_gates(candidate_long, ranking, projects)
    assert decision["passed_gate_count"] == 4
    assert decision["recommendation"] == "GO_FULL_STAGE2"
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_metrics.py
```

Expected: import failure.

- [ ] **Step 3: Implement project metrics**

For every project, calculate:

- `natural_selected_count`
- `actual_selected_count`
- `natural_project_recall`
- `selected_fit_covered_count`
- `selected_fit_coverage`
- `max_natural_candidate_fit`
- `mean_selected_fit`
- `fit_shortfall = max_natural_candidate_fit - mean_selected_fit`
- selected firms’ mean within-project fit percentile
- `top_decile_fit_not_selected`
- `relation_fit_conflict` at thresholds `0.02`, `0.05`, and `0.10`

Define embedded selected firms as:

```python
author_prior_partner == 1 or strong_university_candidate == 1
```

Define unembedded alternatives as:

```python
author_prior_partner == 0 and strong_university_candidate == 0
```

Only candidates with `cognitive_fit_publication_count > 0` are eligible for conflict comparisons. Keep projects with no measurable candidates and mark metrics missing rather than setting them to zero.

- [ ] **Step 4: Implement gate decision**

The machine-readable output must include:

```python
{
    "natural_project_recall": float,
    "natural_firm_instance_recall": float,
    "selected_firm_cognitive_coverage": float,
    "all_candidate_cognitive_coverage": float,
    "conflict_project_share_0_05": float,
    "technical_mrr": float,
    "combined_mrr": float,
    "combined_mrr_improvement": float,
    "gates": {
        "natural_recall_80": bool,
        "selected_fit_coverage_80": bool,
        "conflict_share_10": bool,
        "combined_rank_improves": bool,
    },
    "passed_gate_count": int,
    "recommendation": "GO_FULL_STAGE2" | "REVISE_AND_RERUN_PILOT" | "STOP_MAIN_TOPIC",
}
```

Decision rule:

- 3–4 gates pass: `GO_FULL_STAGE2`
- 2 gates pass: `REVISE_AND_RERUN_PILOT`
- 0–1 gate passes: `STOP_MAIN_TOPIC`

Ranking improvement passes when combined MRR exceeds technical MRR by at least `0.02` or combined Recall@10 exceeds technical Recall@10 by at least `0.05`.

- [ ] **Step 5: Verify GREEN**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_metrics.py
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add .github/overlays/stage2_pilot_metrics.py \
  .github/overlays/test_stage2_pilot_metrics.py
git commit -m "feat: add realization-gap pilot gates"
```

---

### Task 4: Temporal Technical, Relational, and Combined Ranking Models

**Files:**
- Create: `.github/overlays/stage2_pilot_models.py`
- Create: `.github/overlays/test_stage2_pilot_models.py`

**Interfaces:**
- Consumes: candidate long table with project year, selected indicator, cognitive fit, capability, and relationship variables.
- Produces:
  - `fit_temporal_ranking_models(candidate_long, train_end_year=2018) -> tuple[pd.DataFrame, pd.DataFrame]`
  - First output: coefficient table.
  - Second output: ranking metric table by model and test project.

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd
from stage2_pilot_models import fit_temporal_ranking_models


def make_candidate_panel() -> pd.DataFrame:
    rows = []
    for year in range(2015, 2024):
        for project_index in range(8):
            work_id = f"W{year}_{project_index}"
            for company_index in range(5):
                fit = (5 - company_index) / 5
                relation = int(company_index == 1)
                selected = int(
                    company_index
                    == (0 if year >= 2019 else 1)
                )
                rows.append(
                    {
                        "work_id": work_id,
                        "publication_year": year,
                        "company_id": f"C{company_index}",
                        "selected": selected,
                        "cognitive_fit_cosine": fit,
                        "cognitive_fit_publication_count": 2,
                        "prior_subfield_publication_count": 10 - company_index,
                        "author_prior_partner": relation,
                        "university_prior_partner": relation,
                        "strong_university_candidate": relation,
                        "author_relationship_strength": 3 * relation,
                        "author_recency_years": 1.0 if relation else 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_models_emit_temporal_test_ranking_metrics():
    coefficients, metrics = fit_temporal_ranking_models(
        make_candidate_panel(), train_end_year=2018
    )
    assert set(coefficients["model"]) == {
        "technical",
        "relational",
        "combined",
    }
    assert set(metrics["model"]) == {
        "technical",
        "relational",
        "combined",
    }
    assert metrics["publication_year"].min() >= 2019
    assert {
        "selected_best_rank",
        "reciprocal_rank",
        "recall_at_5",
        "recall_at_10",
        "average_precision",
    }.issubset(metrics.columns)
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_models.py
```

Expected: import failure.

- [ ] **Step 3: Implement model preparation**

Create project-level standardized features using training-period means and standard deviations only.

Technical terms:

```python
[
    "cognitive_fit_z",
    "log_subfield_count_z",
    "cognitive_evidence_volume_z",
]
```

Relational terms:

```python
[
    "author_prior_partner",
    "university_prior_partner",
    "strong_university_candidate",
    "log_author_relationship_strength_z",
    "author_recency_z",
]
```

Combined terms:

```python
technical_terms + relational_terms + [
    "fit_x_author",
    "fit_x_strong_university",
]
```

When relationship-strength or recency columns are missing, construct them from available prior evidence counts and dates; otherwise fail with a clear required-column error. Do not silently use future evidence.

- [ ] **Step 4: Fit project-stratified conditional logit on training projects**

Use only training projects that contain at least one selected and one unselected candidate.

```python
ConditionalLogit(
    train["selected"].astype(int),
    train[terms].astype(float),
    groups=train["work_id"],
).fit(method="bfgs", maxiter=500, disp=False)
```

Score 2019–2024 projects with the estimated linear predictor. Conditional-logit stratum intercepts are unnecessary for within-project ranking.

- [ ] **Step 5: Calculate multi-selected project ranking metrics**

For every test project and model:

- Sort by score descending, tie-break by `company_id`.
- `selected_best_rank`: best rank among realized firms.
- `reciprocal_rank = 1 / selected_best_rank`.
- `recall_at_5`: share of selected firms in top 5.
- `recall_at_10`: share of selected firms in top 10.
- `average_precision`: average of precision at each selected-firm rank.

Also output an aggregate row with mean metrics and bootstrap confidence intervals by resampling projects 500 times with seed `20260804`.

- [ ] **Step 6: Verify GREEN**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_models.py
```

Expected: `1 passed`.

- [ ] **Step 7: Commit**

```bash
git add .github/overlays/stage2_pilot_models.py \
  .github/overlays/test_stage2_pilot_models.py
git commit -m "feat: compare technical and relational realization models"
```

---

### Task 5: Pilot Orchestration

**Files:**
- Create: `.github/overlays/run_stage2_realization_pilot.py`
- Create: `.github/overlays/test_stage2_pilot_runner.py`

**Interfaces:**
- CLI arguments:

```text
--projects
--strict-panel
--output-dir
--cache-dir
--sample-size 400
--sample-seed 20260804
--field-fetch-top-n 100
--main-field-top-n 50
--sensitivity-field-top-n 100
--university-min-works 5
--cognitive-batch-size 20
```

- Produces:
  - `pilot_projects.csv`
  - `pilot_sampling_audit.csv`
  - `project_details.csv`
  - `author_company_history.csv`
  - `university_company_history.csv`
  - `subfield_exact_date_company_counts.csv`
  - `natural_candidate_long_top50.csv`
  - `estimation_candidate_long_top50.csv`
  - `natural_candidate_long_top100.csv`
  - `estimation_candidate_long_top100.csv`
  - `candidate_firm_prepublication_text_history.csv`
  - `cognitive_candidate_long_top50.csv`
  - `cognitive_candidate_long_top100.csv`
  - `pilot_project_realization_metrics_top50.csv`
  - `pilot_project_realization_metrics_top100.csv`
  - `pilot_ranking_coefficients_top50.csv`
  - `pilot_ranking_metrics_top50.csv`
  - `pilot_ranking_coefficients_top100.csv`
  - `pilot_ranking_metrics_top100.csv`
  - `pilot_gate_decision_top50.json`
  - `pilot_gate_decision_top100.json`
  - `pilot_summary.json`

- [ ] **Step 1: Write failing runner-contract test**

```python
from pathlib import Path
from run_stage2_realization_pilot import required_output_filenames


def test_runner_declares_all_gate_outputs():
    names = set(required_output_filenames())
    assert {
        "pilot_gate_decision_top50.json",
        "pilot_gate_decision_top100.json",
        "pilot_summary.json",
    }.issubset(names)
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_runner.py
```

Expected: import failure.

- [ ] **Step 3: Implement the runner**

Execution order:

1. Load and audit the full 1,881-project input.
2. Select 400 pilot projects.
3. Subset the strict author-project panel to sampled work IDs.
4. Build manifests from the pilot only.
5. Reuse existing `fetch_project_details`.
6. Reuse existing author and university history extraction.
7. Fetch Top 100 field counts once.
8. Assemble Top 50 and Top 100 natural candidate sets.
9. Write natural recall outputs.
10. Inject uncaptured realized firms.
11. Enrich injected firms with author/university histories.
12. Fetch candidate text history for the union of Top 50 and Top 100 estimation candidates.
13. Compute cognitive fit once for the union.
14. Split back into Top 50 and Top 100 tables.
15. Calculate realization metrics, ranking models, and gate decisions.
16. Write `pilot_summary.json` comparing specifications.
17. Exit `0` only when all required files exist and all time-provenance checks pass. The workflow must still upload artifacts when gate results say `STOP_MAIN_TOPIC`; substantive failure is not a software failure.

- [ ] **Step 4: Add resume checkpoints**

Before every API-intensive phase, read existing output/cache files when valid:

```python
if output_path.exists():
    frame = pd.read_csv(output_path)
else:
    frame = fetch_...
    frame.to_csv(output_path, index=False)
```

Validate expected sampled work IDs before accepting a checkpoint. Do not reuse an output generated from a different sample seed.

- [ ] **Step 5: Verify runner tests**

```bash
PYTHONPATH=.github/overlays pytest -q \
  .github/overlays/test_stage2_pilot_runner.py
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add .github/overlays/run_stage2_realization_pilot.py \
  .github/overlays/test_stage2_pilot_runner.py
git commit -m "feat: orchestrate 400-project realization pilot"
```

---

### Task 6: GitHub Actions Pilot Workflow

**Files:**
- Create: `.github/workflows/tem-stage2-realization-pilot.yml`

**Interfaces:**
- Reconstructs the same code and V3 input as the existing V5 workflow.
- Copies all pilot overlays into `runtime/code`.
- Restores the latest reusable OpenAlex cache.
- Runs tests before any API calls.
- Runs the pilot once with field fetch Top 100.
- Uploads outputs regardless of substantive Go/No-Go result.

- [ ] **Step 1: Create workflow trigger and concurrency policy**

```yaml
name: TEM Stage 2 Realization Pilot

on:
  workflow_dispatch:
  pull_request:
    branches: [main]
    paths:
      - ".github/workflows/tem-stage2-realization-pilot.yml"
      - ".github/overlays/stage2_pilot_*.py"
      - ".github/overlays/run_stage2_realization_pilot.py"
      - ".github/overlays/test_stage2_pilot_*.py"
      - ".github/overlays/stage2_openalex_extract.py"
      - ".github/overlays/test_cache_compaction.py"

concurrency:
  group: tem-stage2-realization-pilot-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```

- [ ] **Step 2: Reconstruct runtime and copy overlays**

Reuse the verified artifact ID, SHA256, and byte-size checks from `.github/workflows/tem-stage2-v5.yml`. After reconstructing code:

```bash
cp .github/overlays/stage2_openalex_extract.py runtime/code/src/stage2_openalex_extract.py
printf '\n' >> runtime/code/src/stage2_openalex_extract.py
cat .github/overlays/stage2_institution_cache_patch.py \
  >> runtime/code/src/stage2_openalex_extract.py

cp .github/overlays/stage2_pilot_sampling.py runtime/code/stage2_pilot_sampling.py
cp .github/overlays/stage2_pilot_risk_set.py runtime/code/stage2_pilot_risk_set.py
cp .github/overlays/stage2_pilot_metrics.py runtime/code/stage2_pilot_metrics.py
cp .github/overlays/stage2_pilot_models.py runtime/code/stage2_pilot_models.py
cp .github/overlays/run_stage2_realization_pilot.py \
  runtime/code/run_stage2_realization_pilot.py

cp .github/overlays/test_stage2_pilot_*.py runtime/code/tests_v5/
```

- [ ] **Step 3: Run all regression and pilot tests**

```bash
cd runtime/code
PYTHONPATH=. pytest -q tests_v5
```

Expected: all legacy tests and all pilot tests pass before extraction.

- [ ] **Step 4: Restore OpenAlex cache**

Use:

```yaml
uses: actions/cache/restore@v4
with:
  path: runtime/openalex_cache
  key: tem-stage2-realization-pilot-${{ github.run_id }}-${{ github.run_attempt }}
  restore-keys: |
    tem-stage2-v5-openalex-30785082979-2
    tem-stage2-v5-openalex-30785082979-1
    tem-stage2-v5-openalex-
```

- [ ] **Step 5: Run pilot**

```bash
python run_stage2_realization_pilot.py \
  --projects ../outputs/v3_input/risk_set_full_firm_projects_input_v3.csv \
  --strict-panel ../outputs/v3_input/author_project_panel_v3_strict.csv \
  --cache-dir ../openalex_cache \
  --output-dir ../outputs/stage2_realization_pilot \
  --sample-size 400 \
  --sample-seed 20260804 \
  --field-fetch-top-n 100 \
  --main-field-top-n 50 \
  --sensitivity-field-top-n 100 \
  --university-min-works 5 \
  --cognitive-batch-size 20 \
  2>&1 | tee ../outputs/stage2_realization_pilot.log
```

Write the process exit code to `stage2_realization_pilot_exit_code.txt`.

- [ ] **Step 6: Save cache and upload artifact**

Artifact name:

```yaml
name: tem-stage2-realization-pilot-${{ github.run_id }}-${{ github.run_attempt }}
```

Retention: 14 days.

- [ ] **Step 7: Enforce software completion only**

The workflow fails only when:

- tests fail;
- extraction/model code exits nonzero;
- time leakage is detected;
- required outputs are missing.

The workflow does not fail merely because fewer than three substantive gates pass.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/tem-stage2-realization-pilot.yml
git commit -m "ci: add TEM realization pilot workflow"
```

---

### Task 7: Verification Before Workflow Dispatch

**Files:**
- Verify all files created in Tasks 1–6.
- No new production file.

- [ ] **Step 1: Run local tests against reconstructed code**

Reconstruct the current code snapshot in a temporary directory, apply overlays exactly as the workflow does, then run:

```bash
PYTHONPATH=. pytest -q tests_v5
```

Expected: all tests pass with no warnings indicating failed convergence or missing columns.

- [ ] **Step 2: Run an offline synthetic smoke test**

Construct 12 synthetic projects and run the pilot from candidate assembly through gate decision without API calls.

Expected:

- Top 50 and Top 100 outputs are generated.
- Realized-firm injection is marked.
- Cognitive coverage and conflict metrics are finite.
- Technical, relational, and combined model metrics are written.
- Gate decision JSON is valid.

- [ ] **Step 3: Verify no secret leakage**

```bash
grep -R --line-number --fixed-strings "OPENALEX_API_KEY=" \
  .github/overlays .github/workflows \
  | grep -v 'GITHUB_ENV'
```

Expected: no secret value or secret file is present.

- [ ] **Step 4: Verify git diff scope**

```bash
git diff --check
git status --short
git diff --stat origin/main...HEAD
```

Expected: only pilot overlays, pilot tests, pilot workflow, extraction rank support, and plan/spec files are changed.

- [ ] **Step 5: Commit verification fixes if necessary**

```bash
git add .github/overlays .github/workflows
git commit -m "test: verify TEM realization pilot"
```

Do not create an empty commit.

---

### Task 8: Dispatch, Monitor, and Analyze the Pilot

**Files:**
- Generated artifact only.
- Create after download: `stage2_realization_pilot/pilot_evaluation_report.html`
- Create after download: `stage2_realization_pilot/pilot_gate_table.csv`

**Interfaces:**
- Consumes the GitHub Actions artifact from Task 6.
- Produces a final four-gate decision and recommendation.

- [ ] **Step 1: Dispatch the workflow**

Use GitHub Actions workflow dispatch on branch `tem-stage2-v5-run`.

- [ ] **Step 2: Inspect job steps and logs**

Confirm:

- tests passed;
- sample contains exactly 400 unique projects;
- API extraction resumed from cache where available;
- process exit code is zero;
- no time-provenance violation occurred.

- [ ] **Step 3: Download and audit artifact**

Required checks:

```python
assert exit_code == 0
assert len(pilot_projects) == 400
assert pilot_projects["work_id"].is_unique
assert candidate_long.groupby(["work_id", "company_id"]).size().eq(1).all()
assert (
    pd.to_datetime(observed_evidence_date)
    < pd.to_datetime(publication_date)
).all()
```

- [ ] **Step 4: Evaluate the four gates separately for Top 50 and Top 100**

Report:

| Gate | Top 50 | Top 100 | Threshold |
|---|---:|---:|---:|
| Natural project recall | value | value | ≥80% |
| Selected-firm cognitive coverage | value | value | ≥80% |
| Relation–fit conflict project share | value | value | ≥10% |
| Combined ranking improvement | value | value | MRR +0.02 or Recall@10 +0.05 |

- [ ] **Step 5: Apply final decision rule**

- At least three gates pass in Top 50 and findings are directionally stable in Top 100: proceed to all 1,881 projects.
- Three gates pass only in Top 100: revise candidate set before full expansion.
- Two gates pass: diagnose the failing gates and run one prespecified pilot revision.
- Zero or one gate passes: stop the main topic and retain the relation-only paper as a secondary option.

- [ ] **Step 6: Produce final report**

The report must separate:

- verified results;
- robustness differences;
- missingness;
- forced-selected dependence;
- interpretation;
- exact blockers before full expansion.

- [ ] **Step 7: Do not claim completion until verified**

Do not state that the pilot succeeded unless:

- `stage2_realization_pilot_exit_code.txt == 0`;
- all required outputs are present;
- time provenance is audited;
- gate calculations are independently recomputed.

---

## Plan Self-Review

### Spec coverage

- 400-project stratified sample: Task 1 and Task 5.
- Top 50 main and Top 100 sensitivity: Tasks 2, 5, 6, and 8.
- Author ties, strong university ties, and selected-firm injection: Task 2.
- Cognitive fit: Tasks 3 and 5 using existing `src.cognitive_fit`.
- Natural recall ≥80%: Tasks 2, 3, and 8.
- Cognitive coverage ≥80%: Tasks 3 and 8.
- Conflict projects ≥10%: Tasks 3 and 8.
- Technical versus combined ranking: Tasks 4 and 8.
- Restartable OpenAlex extraction: Tasks 5 and 6.
- Time leakage protection: Tasks 2, 5, 7, and 8.
- No causal or consideration-set overclaiming: Global Constraints and Task 8.

### Placeholder scan

The plan contains no `TBD`, `TODO`, “implement later,” or unspecified test instructions.

### Type consistency

- `select_stratified_pilot_projects` returns two DataFrames and is consumed by the runner.
- `assemble_pilot_candidates` and `inject_realized_firms` return the candidate tables required by the cognitive-fit stage.
- `build_project_realization_metrics`, `fit_temporal_ranking_models`, and `evaluate_pilot_gates` consume the same candidate-long schema.
- Top 50 and Top 100 use one Top 100 field extraction and one union cognitive-history extraction.
