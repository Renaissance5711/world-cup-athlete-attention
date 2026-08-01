# Rank-Reversal Decomposition and Manuscript Strengthening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify why proportional and additive athlete-attention estimands reorder the 117 observed World Cup scorers, integrate that evidence into the IJSMS manuscript, and complete reproducibility, integrity, and adversarial-review gates.

**Architecture:** A pure analysis module will classify every unordered athlete pair using the published observed estimands, separately evaluate the denominator accounting identity with a reconstructed proportional score, and recompute reversal rates under the existing match-weighted Bayesian bootstrap. A thin runner will write stable CSV/JSON outputs. The manuscript and tables will then be revised from the approved final DOCX files without publishing anonymous submission materials in GitHub.

**Tech Stack:** Python 3.11+, pandas, NumPy, pytest, python-docx, LibreOffice-based `render_docx.py`, GitHub feature branch and pull request.

## Global Constraints

- Use the same 117 unique scorers and the same mean-across-scoring-appearances aggregation as `code/run_observed_ranking_bootstrap.py`.
- Treat observed proportional log lift and observed winsorised additional pageviews as the primary ranking variables.
- Do not describe the denominator relationship as a new theorem or the ranking comparison as causal.
- Report concordant, reversed, and tied pairs separately; exclude tied pairs only from the concordant-versus-reversed rate denominator.
- Use seed `20260730`, `10000` bootstrap draws, and shared match-level Dirichlet weights.
- Keep anonymous manuscript, submission tables, and submission figures outside the public GitHub repository.
- Preserve the approved manuscript sequence `H1 → estimand contrast → H2 → H3` and remove all `H2a`, `H2b`, and `H4` references.
- Never overwrite the approved source DOCX files. Write revised files under `/mnt/data/IJSMS_contribution_strengthened/`.
- Every meaningful DOCX edit batch must be rendered to page PNGs and visually inspected before delivery.

## Pre-implementation Technical Clarification

The approved design correctly makes observed pairwise rank reversal the primary empirical object. A pre-plan audit found that the published athlete-level proportional score is a mean of scoring-appearance log lifts, whereas the additive score is a mean of winsorised additional pageviews. Therefore, the algebraic condition based on `log(D_i / D_j)` and `log(B_i / B_j)` is exact for the reconstructed ratio score `log1p(D_i / B_i)`, but not guaranteed to reproduce every ordering of the published mean-log-lift score.

Implementation must preserve this distinction:

1. **Primary observed decomposition:** compare the published proportional and additive estimands directly.
2. **Denominator accounting diagnostic:** compare additive ordering with `log1p(D/B)` and apply the exact inequality there.
3. **Aggregation residual:** report pairs for which the denominator diagnostic and the published proportional ordering disagree; attribute them to aggregation, transformation, and winsorisation differences rather than forcing a denominator-only explanation.

The current frozen input implies 6,786 unordered pairs, 6 tied pairs, and 6,780 comparable pairs. These are reproducibility invariants, not claims to be hard-coded into the analysis logic.

## File Map

### Public repository files

- Create: `ijsms/rank_reversal.py` — validation, pairwise classification, denominator diagnostic, athlete displacement, summary, and bootstrap functions.
- Create: `code/run_rank_reversal_decomposition.py` — deterministic command-line runner and output writer.
- Create: `tests/test_rank_reversal.py` — unit tests using synthetic athletes and scoring appearances.
- Create: `tests/test_rank_reversal_integration.py` — frozen-data reproducibility checks.
- Create: `outputs/r25/rank_reversal_pairs.csv` — one row per unordered athlete pair.
- Create: `outputs/r25/rank_reversal_athlete_displacements.csv` — athlete-level rank displacement diagnostics.
- Create: `outputs/r25/rank_reversal_bootstrap_draws.csv` — draw-level reversal and identity-agreement statistics.
- Create: `outputs/r25/rank_reversal_decomposition_table.csv` — compact submission-table source.
- Create: `outputs/r25/rank_reversal_summary.json` — machine-readable primary results and definitions.
- Create: `docs/RANK_REVERSAL_DECOMPOSITION.md` — public method, interpretation, and reproduction notes.
- Modify: `README.md` — add the new runner and output description.
- Modify: `SHA256SUMS_PUBLIC.txt` — refresh public-file checksums after all repository changes.

### Submission-package files

- Source: `/mnt/data/IJSMS_exemplar_aligned_submission/01_Manuscript_IJSMS_final.docx`.
- Source: `/mnt/data/IJSMS_exemplar_aligned_submission/02_Tables_IJSMS_final.docx`.
- Create: `/mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx`.
- Create: `/mnt/data/IJSMS_contribution_strengthened/02_Tables_IJSMS_contribution_strengthened.docx`.
- Copy unchanged after verification: Figure 1 and Figure 2 publication files from the exemplar-aligned package.
- Create: `/mnt/data/IJSMS_contribution_strengthened/00_Revision_and_integrity_report.md`.

---

### Task 1: Create the isolated GitHub implementation branch

**Files:**
- No content files changed in this task.

**Interfaces:**
- Consumes: current `main` branch of `Renaissance5711/world-cup-athlete-attention`.
- Produces: branch `feature/rank-reversal-decomposition` used by Tasks 2–6.

- [ ] **Step 1: Read the current default-branch head SHA**

Run through the GitHub connector: list the most recent commit for `Renaissance5711/world-cup-athlete-attention`.

Expected: one commit SHA for `main`.

- [ ] **Step 2: Create the feature branch**

Create `feature/rank-reversal-decomposition` from the exact SHA returned in Step 1.

Expected: GitHub confirms branch creation.

- [ ] **Step 3: Confirm branch isolation**

Fetch `README.md` from both `main` and `feature/rank-reversal-decomposition` and confirm their blob SHAs match before implementation.

Expected: identical README blob SHAs.

### Task 2: Implement pure pairwise decomposition with test-first development

**Files:**
- Create: `ijsms/rank_reversal.py`
- Create: `tests/test_rank_reversal.py`

**Interfaces:**
- Consumes: a DataFrame with `player_id`, `player_name`, `observed_proportional_log_lift`, `observed_additional_pageviews`, `baseline_views`, `proportional_rank`, and `additive_rank`.
- Produces:
  - `validate_rank_input(frame: pd.DataFrame) -> None`
  - `build_pairwise_decomposition(frame: pd.DataFrame) -> pd.DataFrame`
  - `build_athlete_displacements(frame: pd.DataFrame) -> pd.DataFrame`
  - `summarize_decomposition(pairs: pd.DataFrame, athletes: pd.DataFrame) -> dict[str, object]`

- [ ] **Step 1: Write failing validation tests**

```python
import pandas as pd
import pytest

from ijsms.rank_reversal import validate_rank_input


def test_validate_rank_input_rejects_nonpositive_baseline():
    frame = pd.DataFrame({
        "player_id": ["A"],
        "player_name": ["A"],
        "observed_proportional_log_lift": [1.0],
        "observed_additional_pageviews": [100.0],
        "baseline_views": [0.0],
        "proportional_rank": [1],
        "additive_rank": [1],
    })
    with pytest.raises(ValueError, match="baseline_views must be positive"):
        validate_rank_input(frame)
```

- [ ] **Step 2: Run the validation test and verify failure**

Run: `pytest tests/test_rank_reversal.py::test_validate_rank_input_rejects_nonpositive_baseline -v`

Expected: FAIL because `ijsms.rank_reversal` does not exist.

- [ ] **Step 3: Implement input validation**

```python
REQUIRED_COLUMNS = {
    "player_id",
    "player_name",
    "observed_proportional_log_lift",
    "observed_additional_pageviews",
    "baseline_views",
    "proportional_rank",
    "additive_rank",
}


def validate_rank_input(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame["player_id"].duplicated().any():
        raise ValueError("player_id must be unique")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("rank input contains missing values")
    if (frame["baseline_views"] <= 0).any():
        raise ValueError("baseline_views must be positive")
    if (frame["observed_additional_pageviews"] <= 0).any():
        raise ValueError("observed_additional_pageviews must be positive")
```

- [ ] **Step 4: Run the validation test and verify success**

Run: `pytest tests/test_rank_reversal.py::test_validate_rank_input_rejects_nonpositive_baseline -v`

Expected: PASS.

- [ ] **Step 5: Write the failing pair-classification test**

```python
def test_build_pairwise_decomposition_distinguishes_observed_and_identity_ordering():
    frame = pd.DataFrame({
        "player_id": ["A", "B", "C"],
        "player_name": ["Alpha", "Beta", "Gamma"],
        "observed_proportional_log_lift": [1.0, 2.0, 1.5],
        "observed_additional_pageviews": [300.0, 200.0, 200.0],
        "baseline_views": [300.0, 20.0, 100.0],
        "proportional_rank": [3, 1, 2],
        "additive_rank": [1, 2, 2],
    })
    pairs = build_pairwise_decomposition(frame)
    ab = pairs.query("player_i_id == 'A' and player_j_id == 'B'").iloc[0]
    assert ab["observed_order"] == "reversed"
    assert ab["identity_order"] == "reversed"
    assert ab["identity_matches_observed"]
    bc = pairs.query("player_i_id == 'B' and player_j_id == 'C'").iloc[0]
    assert bc["observed_order"] == "tied_additive"
```

- [ ] **Step 6: Run the pair test and verify failure**

Run: `pytest tests/test_rank_reversal.py::test_build_pairwise_decomposition_distinguishes_observed_and_identity_ordering -v`

Expected: FAIL because `build_pairwise_decomposition` is undefined.

- [ ] **Step 7: Implement vectorised pair construction**

The function must create `numpy.triu_indices(n, 1)`, preserve deterministic player ordering by `player_id`, and emit these columns:

```text
player_i_id, player_j_id, player_i_name, player_j_name,
proportional_difference, additive_difference, baseline_difference,
observed_order, reconstructed_proportional_i, reconstructed_proportional_j,
identity_order, identity_matches_observed,
increment_log_gap, baseline_log_gap, dominance_margin
```

Classification rules:

```python
observed_order = np.select(
    [add_sign == 0, prop_sign == 0, add_sign != prop_sign],
    ["tied_additive", "tied_proportional", "reversed"],
    default="concordant",
)
identity_order = np.where(add_sign == 0, "tied_additive", np.where(add_sign != reconstructed_sign, "reversed", "concordant"))
```

Orient `increment_log_gap`, `baseline_log_gap`, and `dominance_margin` so that the athlete with the larger additive response is always the numerator. Set these fields to missing for additive ties.

- [ ] **Step 8: Run all unit tests for pair construction**

Run: `pytest tests/test_rank_reversal.py -v`

Expected: PASS for validation and classification tests.

- [ ] **Step 9: Write failing athlete-displacement and summary tests**

```python
def test_build_athlete_displacements_uses_absolute_and_signed_rank_changes():
    frame = synthetic_rank_frame()
    result = build_athlete_displacements(frame).set_index("player_id")
    assert result.loc["A", "rank_displacement"] == -2
    assert result.loc["A", "absolute_rank_displacement"] == 2


def test_summarize_decomposition_excludes_ties_from_reversal_rate():
    frame = synthetic_rank_frame()
    pairs = build_pairwise_decomposition(frame)
    summary = summarize_decomposition(pairs, build_athlete_displacements(frame))
    assert summary["total_pairs"] == 3
    assert summary["comparable_pairs"] == 2
    assert summary["tied_pairs"] == 1
    assert 0 <= summary["observed_reversal_rate"] <= 1
```

- [ ] **Step 10: Implement displacement and summary functions**

`rank_displacement` must equal `additive_rank - proportional_rank`. The summary must include total, comparable, concordant, reversed and tied counts; observed reversal rate; identity reversal count and rate; identity-observed agreement count and rate; median and interquartile dominance margins; median, mean and maximum absolute rank displacement; and counts moving at least 10, 25 and 50 ranks.

- [ ] **Step 11: Run the complete unit-test file**

Run: `pytest tests/test_rank_reversal.py -v`

Expected: all tests PASS.

- [ ] **Step 12: Commit the pure decomposition module**

```bash
git add ijsms/rank_reversal.py tests/test_rank_reversal.py
git commit -m "feat: add pairwise rank reversal decomposition"
```

### Task 3: Implement match-weighted Bayesian bootstrap

**Files:**
- Modify: `ijsms/rank_reversal.py`
- Modify: `tests/test_rank_reversal.py`

**Interfaces:**
- Consumes: scoring-appearance DataFrame containing `player_id`, `match_id`, `immediate_attention_log_lift`, `winsorized_additional_pageviews`, and `baseline_views`.
- Produces: `bootstrap_reversal_statistics(scoring: pd.DataFrame, draws: int, seed: int, batch_size: int = 250) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing reproducibility test**

```python
def test_bootstrap_reversal_statistics_is_reproducible_and_bounded():
    scoring = synthetic_scoring_appearances()
    first = bootstrap_reversal_statistics(scoring, draws=20, seed=11, batch_size=5)
    second = bootstrap_reversal_statistics(scoring, draws=20, seed=11, batch_size=5)
    pd.testing.assert_frame_equal(first, second)
    assert first["observed_reversal_rate"].between(0, 1).all()
    assert first["identity_agreement_rate"].between(0, 1).all()
```

- [ ] **Step 2: Run the bootstrap test and verify failure**

Run: `pytest tests/test_rank_reversal.py::test_bootstrap_reversal_statistics_is_reproducible_and_bounded -v`

Expected: FAIL because the bootstrap function is undefined.

- [ ] **Step 3: Implement shared match-level weighting**

Use this contract:

```python
rng = np.random.default_rng(seed)
weights = rng.dirichlet(np.ones(number_of_matches), size=draws)
```

For every draw, compute each athlete's weighted mean proportional and additive response using the same match weights for all athletes. Keep baseline fixed at its stored player value. Reclassify all unordered pairs in batches of at most `batch_size` draws to avoid constructing a `draws × pairs × variables` object in memory.

Each draw row must contain:

```text
draw, unique_scorers, total_pairs, comparable_pairs, tied_pairs,
observed_reversal_n, observed_reversal_rate,
identity_reversal_n, identity_reversal_rate,
identity_agreement_n, identity_agreement_rate
```

- [ ] **Step 4: Run the bootstrap test and verify success**

Run: `pytest tests/test_rank_reversal.py::test_bootstrap_reversal_statistics_is_reproducible_and_bounded -v`

Expected: PASS.

- [ ] **Step 5: Add a test for shared-weight behavior**

Construct two appearances from the same match and assert that changing the match weight affects both athletes through the same weight column rather than through independent athlete weights.

- [ ] **Step 6: Run the rank-reversal unit suite**

Run: `pytest tests/test_rank_reversal.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit bootstrap support**

```bash
git add ijsms/rank_reversal.py tests/test_rank_reversal.py
git commit -m "feat: bootstrap scorer rank reversal statistics"
```

### Task 4: Add the reproducible analysis runner and frozen-data integration checks

**Files:**
- Create: `code/run_rank_reversal_decomposition.py`
- Create: `tests/test_rank_reversal_integration.py`
- Create: `outputs/r25/rank_reversal_pairs.csv`
- Create: `outputs/r25/rank_reversal_athlete_displacements.csv`
- Create: `outputs/r25/rank_reversal_bootstrap_draws.csv`
- Create: `outputs/r25/rank_reversal_decomposition_table.csv`
- Create: `outputs/r25/rank_reversal_summary.json`

**Interfaces:**
- Consumes: `outputs/r24/observed_goal_scorer_metric_rankings.csv` and `base_archive/data/processed/all_player_match_outcomes_2022.csv`.
- Produces: the five stable R25 output files listed above.

- [ ] **Step 1: Write the failing integration test**

```python
from pathlib import Path
import pandas as pd

from code.run_rank_reversal_decomposition import run


def test_frozen_rank_decomposition_counts(tmp_path: Path):
    summary = run(output_dir=tmp_path, draws=50, seed=20260730)
    assert summary["unique_scorers"] == 117
    assert summary["total_pairs"] == 6786
    assert summary["comparable_pairs"] == 6780
    assert summary["tied_pairs"] == 6
    assert summary["observed_reversal_n"] == 3058
    assert summary["observed_concordant_n"] == 3722
    assert abs(summary["observed_reversal_rate"] - 0.4510324483775811) < 1e-12
    pairs = pd.read_csv(tmp_path / "rank_reversal_pairs.csv")
    assert len(pairs) == 6786
```

- [ ] **Step 2: Run the integration test and verify failure**

Run: `pytest tests/test_rank_reversal_integration.py -v`

Expected: FAIL because the runner is undefined.

- [ ] **Step 3: Implement the runner**

Expose:

```python
def run(
    ranking_path: Path = ROOT / "outputs/r24/observed_goal_scorer_metric_rankings.csv",
    events_path: Path = ROOT / "base_archive/data/processed/all_player_match_outcomes_2022.csv",
    output_dir: Path = ROOT / "outputs/r25",
    draws: int = 10000,
    seed: int = 20260730,
) -> dict[str, object]:
    ...
```

The runner must create the output directory, validate inputs, write UTF-8 CSV files without indices, write indented JSON, and return the exact summary written to disk.

- [ ] **Step 4: Define the compact table source**

Write `rank_reversal_decomposition_table.csv` with these rows:

```text
Total unordered athlete pairs
Comparable pairs
Observed rank reversals
Observed concordant pairs
Pairs tied on either observed estimand
Identity-observed agreement
Median absolute athlete rank displacement
Athletes moving at least 25 ranks
```

Columns must be `Statistic`, `Estimate`, `Percent_or_interval`, and `Definition`.

- [ ] **Step 5: Run the integration test with 50 bootstrap draws**

Run: `pytest tests/test_rank_reversal_integration.py -v`

Expected: PASS in under 30 seconds.

- [ ] **Step 6: Run the production analysis**

Run: `python code/run_rank_reversal_decomposition.py`

Expected: five files in `outputs/r25/`, `10000` rows in the bootstrap-draw file, and a printed summary ending without warnings or tracebacks.

- [ ] **Step 7: Verify production invariants**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/r25/rank_reversal_summary.json")
s = json.loads(p.read_text())
assert s["unique_scorers"] == 117
assert s["total_pairs"] == 6786
assert s["observed_reversal_n"] == 3058
assert s["tied_pairs"] == 6
print(s["observed_reversal_rate"])
PY
```

Expected printed value: `0.4510324483775811`.

- [ ] **Step 8: Run all public repository tests**

Run: `python -m pytest -q -W error::FutureWarning`

Expected: all tests PASS with no FutureWarning promoted to an error.

- [ ] **Step 9: Commit runner, tests and outputs**

```bash
git add code/run_rank_reversal_decomposition.py tests/test_rank_reversal_integration.py outputs/r25
git commit -m "feat: publish observed rank reversal results"
```

### Task 5: Document the method and refresh reproducibility metadata

**Files:**
- Create: `docs/RANK_REVERSAL_DECOMPOSITION.md`
- Modify: `README.md`
- Modify: `SHA256SUMS_PUBLIC.txt`

**Interfaces:**
- Consumes: verified R25 outputs from Task 4.
- Produces: public-facing method documentation and matching checksums.

- [ ] **Step 1: Write the method document**

The document must define the observed decomposition, denominator accounting diagnostic, aggregation residual, tie handling, bootstrap, and interpretation boundary. Include this sentence verbatim:

> The arithmetic identity is an accounting device, not the study's mathematical contribution; the empirical contribution is the magnitude and distribution of observed ordering changes in the athlete sample.

- [ ] **Step 2: Add the reproduction command to README**

Add:

```bash
python code/run_rank_reversal_decomposition.py
```

Describe `outputs/r25/` without mentioning or linking anonymous manuscript files.

- [ ] **Step 3: Regenerate checksums**

Run from repository root:

```bash
find . -type f ! -path './.git/*' ! -name 'SHA256SUMS_PUBLIC.txt' -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS_PUBLIC.txt
```

- [ ] **Step 4: Verify checksums and tests**

Run:

```bash
sha256sum -c SHA256SUMS_PUBLIC.txt
python -m pytest -q -W error::FutureWarning
```

Expected: every checksum reports `OK`; all tests PASS.

- [ ] **Step 5: Commit documentation and metadata**

```bash
git add README.md docs/RANK_REVERSAL_DECOMPOSITION.md SHA256SUMS_PUBLIC.txt
git commit -m "docs: explain rank reversal decomposition"
```

### Task 6: Open and review the GitHub pull request

**Files:**
- No new content files.

**Interfaces:**
- Consumes: completed feature branch.
- Produces: reviewed PR targeting `main`.

- [ ] **Step 1: Compare feature branch with main**

Use the GitHub compare action and confirm only the planned code, tests, outputs, documentation and checksum files changed.

- [ ] **Step 2: Open the pull request**

Title: `Add scorer rank-reversal decomposition`

Body must state:

```text
This PR quantifies pairwise ordering changes between observed proportional and additive athlete-attention estimands. It separates direct observed reversals from a denominator-accounting diagnostic and explicitly reports residual disagreement created by aggregation and winsorisation. It does not claim a new mathematical identity or causal effect.
```

- [ ] **Step 3: Perform code and specification review**

Check the PR against every Global Constraint and the approved design. Request changes for any denominator-only overclaim, unreported tie, non-deterministic output, or manuscript material accidentally added to the public repository.

- [ ] **Step 4: Merge only after tests and review pass**

Use squash merge with commit title: `Add scorer rank-reversal decomposition`.

### Task 7: Build a verified innovation-positioning matrix

**Files:**
- Create locally: `/mnt/data/IJSMS_contribution_strengthened/innovation_positioning_matrix.md`

**Interfaces:**
- Consumes: current manuscript references, verified web literature, and R25 results.
- Produces: a source-grounded matrix used to revise the introduction and discussion.

- [ ] **Step 1: Extract the manuscript's current contribution and novelty claims**

Record each claim, its current citation support, and whether it concerns arithmetic, observed ranking consequences, temporal boundaries, relational boundaries, or managerial decision design.

- [ ] **Step 2: Search current primary literature**

Search peer-reviewed and official primary sources for direct competitors on event leverage, athlete attention, online attention measurement, proportional-versus-absolute outcomes, ranking disagreement, and pre-activation resource identification. Do not rely on repository summaries, blogs, or generated citation lists as evidence.

- [ ] **Step 3: Verify every candidate source**

For each candidate, record DOI or stable publisher record, study setting, measured outcome, actual contribution, and the precise manuscript sentence it can support. Mark sources that were discovered but do not support the intended claim.

- [ ] **Step 4: Write the positioning matrix**

Use columns:

```text
Manuscript claim | Closest prior work | What prior work establishes | What this study adds | Evidence source | Revision action
```

- [ ] **Step 5: Run a novelty stress test**

Write separate answers to:

```text
What remains novel if the denominator identity is deleted?
What remains novel if the top-12 overlap statistic is deleted?
Which contribution requires the new full-ranking decomposition?
Which conclusions remain descriptive rather than causal?
```

Expected: each answer cites either verified literature or a named R25 output.

### Task 8: Revise the manuscript structure, hypotheses and contribution claims

**Files:**
- Create: `/mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx`

**Interfaces:**
- Consumes: final source manuscript, innovation-positioning matrix, and R25 outputs.
- Produces: revised manuscript with continuous hypotheses and a new rank-reversal subsection.

- [ ] **Step 1: Copy the source manuscript without overwriting it**

```bash
mkdir -p /mnt/data/IJSMS_contribution_strengthened
cp /mnt/data/IJSMS_exemplar_aligned_submission/01_Manuscript_IJSMS_final.docx \
   /mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx
```

- [ ] **Step 2: Replace the H2a/H2b block with the estimand contrast**

The replacement must state that proportional response answers a relative-movement question and additional pageviews answer an incremental-demand question. It must explicitly state that baseline dependence is partly built into the proportional measure and therefore is not offered as a mathematical innovation.

- [ ] **Step 3: Renumber the remaining hypotheses**

Change former H3 to H2 and former H4 to H3 in Sections 2, 3, 4, the abstract where applicable, and the discussion. Search the full document for `H2a`, `H2b`, `H3`, and `H4`; manually verify every match rather than using an unbounded global replacement.

- [ ] **Step 4: Add the decomposition method to Section 3.8**

Describe 117 scorers, 6,786 unordered pairs, direct observed classification, additive ties, denominator diagnostic, aggregation residual, and match-weighted Bayesian bootstrap. Do not include production results in the method section.

- [ ] **Step 5: Add a new Results subsection after current Section 4.2**

Heading: `4.3 Decomposing rank reversals across attention estimands`.

Report verified observed counts and percentages, rank displacement, denominator-diagnostic agreement, aggregation residual, and bootstrap intervals from `rank_reversal_summary.json`. Explain that the analysis quantifies the empirical reach of metric choice rather than proving the denominator identity.

- [ ] **Step 6: Renumber later Results subsections**

Current 4.3 becomes 4.4, current 4.4 becomes 4.5, and current 4.5 becomes 4.6. Update every internal cross-reference.

- [ ] **Step 7: Rewrite the contribution paragraphs**

Revise the abstract originality statement, Introduction contribution paragraph, Discussion 5.1, managerial implications, limitations and conclusion so that the contribution hierarchy is:

```text
measurement objective → observed selection-rule divergence → temporal boundary → relational boundary
```

State explicitly that the decomposition reveals how extensively real candidate ordering changes, not a new arithmetic law.

- [ ] **Step 8: Control manuscript length**

Add no more than 450 net words. Remove duplicated explanations of discovery costs, cumulative advantage, and top-list overlap so the revised manuscript remains within 10% of the original word count.

- [ ] **Step 9: Run textual consistency checks**

Extract document text and verify:

```bash
! grep -E "H2a|H2b|H4" revised_manuscript.txt
grep -F "Estimand contrast" revised_manuscript.txt
grep -F "Decomposing rank reversals across attention estimands" revised_manuscript.txt
```

Expected: first command succeeds because no obsolete labels remain; both required phrases are found.

### Task 9: Add the new submission table and renumber existing tables

**Files:**
- Create: `/mnt/data/IJSMS_contribution_strengthened/02_Tables_IJSMS_contribution_strengthened.docx`

**Interfaces:**
- Consumes: final source tables and `outputs/r25/rank_reversal_decomposition_table.csv`.
- Produces: five-table DOCX with consistent captions and manuscript references.

- [ ] **Step 1: Copy the source tables file**

```bash
cp /mnt/data/IJSMS_exemplar_aligned_submission/02_Tables_IJSMS_final.docx \
   /mnt/data/IJSMS_contribution_strengthened/02_Tables_IJSMS_contribution_strengthened.docx
```

- [ ] **Step 2: Insert new Table II after Table I**

Caption: `Table II. Pairwise decomposition of scorer rankings under proportional and additive attention estimands`.

The notes must define unordered pairs, comparable pairs, ties, identity-observed agreement, rank displacement, and the match-weighted Bayesian bootstrap interval. The source line remains `Source: Authors' own work.`

- [ ] **Step 3: Renumber existing tables**

Old Table II becomes Table III, old Table III becomes Table IV, and old Table IV becomes Table V. Preserve each table's existing contents, widths, notes, and source line.

- [ ] **Step 4: Update all manuscript table references**

Ensure the new decomposition section cites Table II, baseline robustness cites Table III, persistence cites Table IV, and shooter-goalkeeper results cite Table V.

- [ ] **Step 5: Verify numeric agreement**

Programmatically compare every numeric cell in new Table II with `rank_reversal_decomposition_table.csv`. Fail the build on any mismatch.

### Task 10: Render and visually verify both DOCX files

**Files:**
- Inspect: revised manuscript and tables DOCX files.
- Create internal QA renders under `/mnt/data/IJSMS_contribution_strengthened/_render_manuscript/` and `_render_tables/`.

**Interfaces:**
- Consumes: revised DOCX files from Tasks 8 and 9.
- Produces: visually verified DOCX files; PNG/PDF renders remain internal QA artifacts.

- [ ] **Step 1: Run style and structure audits**

```bash
python /home/oai/skills/docx/scripts/style_lint.py /mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx
python /home/oai/skills/docx/scripts/heading_audit.py /mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx
python /home/oai/skills/docx/scripts/section_audit.py /mnt/data/IJSMS_contribution_strengthened/02_Tables_IJSMS_contribution_strengthened.docx
```

Expected: no unexplained style drift, broken heading sequence, or section-layout error.

- [ ] **Step 2: Render manuscript and tables**

```bash
python /home/oai/skills/docx/render_docx.py \
  /mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx \
  --output_dir /mnt/data/IJSMS_contribution_strengthened/_render_manuscript --emit_pdf
python /home/oai/skills/docx/render_docx.py \
  /mnt/data/IJSMS_contribution_strengthened/02_Tables_IJSMS_contribution_strengthened.docx \
  --output_dir /mnt/data/IJSMS_contribution_strengthened/_render_tables --emit_pdf
```

- [ ] **Step 3: Inspect every page PNG at 100% zoom**

Check page breaks, headings, equations, table wrapping, caption placement, clipped text, missing glyphs, and orphaned source/notes lines. Record each defect by page number.

- [ ] **Step 4: Fix every recorded defect and re-render**

Repeat Steps 2 and 3 until no defects remain. A text-only inspection does not satisfy this step.

- [ ] **Step 5: Run metadata and accessibility audits**

```bash
python /home/oai/skills/docx/scripts/a11y_audit.py /mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx
python /home/oai/skills/docx/scripts/privacy_scrub.py /mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened.docx --out /mnt/data/IJSMS_contribution_strengthened/01_Manuscript_IJSMS_contribution_strengthened_scrubbed.docx
```

Replace the working manuscript with the scrubbed file only after confirming visible content and layout remain unchanged on a final render.

### Task 11: Run Academic Research Skills integrity and adversarial-review gates

**Files:**
- Create: `/mnt/data/IJSMS_contribution_strengthened/00_Revision_and_integrity_report.md`
- Modify revised DOCX files only when a verified defect requires correction.

**Interfaces:**
- Consumes: revised manuscript, tables, R25 outputs, current figures, verified references, and source matrix.
- Produces: final integrity report and corrected submission package.

- [ ] **Step 1: Complete the pre-review integrity gate**

For every new number in the manuscript and Table II, record the exact source file, field, and value. Verify that all cited references exist and support the attached claim. Mark any unsupported sentence as a blocking failure.

- [ ] **Step 2: Run five review perspectives**

Produce separate findings for:

```text
Contribution and literature positioning
Measurement and estimand logic
Statistical and reproducibility quality
Sport-management relevance
Devil's Advocate rejection case
```

Each finding must quote or locate the relevant manuscript passage and classify issues as Priority 1, Priority 2, or editorial.

- [ ] **Step 3: Resolve Priority 1 issues**

Revise the manuscript only when the critique is supported. Do not accept demands that would convert descriptive ranking evidence into causal language or reintroduce the arithmetic identity as the novelty claim.

- [ ] **Step 4: Run focused re-review**

Check that each Priority 1 response is visible in the revised text and did not create contradictory claims, incorrect numbering, or table mismatches.

- [ ] **Step 5: Complete the final integrity gate**

Require zero unresolved numerical, citation, hypothesis-numbering, table-numbering, or data-availability issues. Record test results, render status, and checksum status in `00_Revision_and_integrity_report.md`.

### Task 12: Assemble and verify the strengthened submission package

**Files:**
- Final contents under `/mnt/data/IJSMS_contribution_strengthened/`.

**Interfaces:**
- Consumes: final revised manuscript, tables, unchanged verified figures, and integrity report.
- Produces: complete non-ZIP deliverable directory and optional ZIP only when explicitly requested later.

- [ ] **Step 1: Copy verified figure files**

Copy Figure 1 and Figure 2 PDF, EPS, SVG, and 600-dpi TIFF versions from `/mnt/data/IJSMS_exemplar_aligned_submission/` without modifying their artwork.

- [ ] **Step 2: Update the package README**

Create `00_Submission_README.txt` listing the revised manuscript, five-table file, figure files, revision report, and public reproducibility repository URL.

- [ ] **Step 3: Calculate package checksums**

Create `SHA256SUMS_SUBMISSION.txt` covering deliverable files but excluding internal render directories.

- [ ] **Step 4: Perform final cross-file checks**

Verify manuscript references to Tables I–V and Figures 1–2; verify all listed files exist; verify the public GitHub URL resolves; verify no ZIP or anonymous source package was accidentally placed in the public repository.

- [ ] **Step 5: Deliver only verified final artifacts**

Provide links to the strengthened manuscript, tables, revision report, and submission directory contents. Do not deliver internal page PNGs unless requested.

## Plan Self-Review Result

- **Spec coverage:** Pairwise classification, exact denominator diagnostic, aggregation residual, athlete displacement, bootstrap, outputs, tests, documentation, hypothesis renumbering, new table, manuscript integration, review, integrity, and rendering are each mapped to a task.
- **Placeholder scan:** No placeholder markers or unspecified test steps remain.
- **Type consistency:** Function names and column names are consistent across Tasks 2–5. Submission numbering is consistently H1/estimand contrast/H2/H3 and Tables I–V.
- **Scope:** The public analysis and private manuscript are separate deliverables but share one validated result contract; keeping them in one plan prevents numerical drift between code, tables, and prose.
