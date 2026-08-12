# TEM COMPOT Two-Stage Extension Design

Date: 2026-08-11  
Branch: `tem-stage2-v5-run`  
Scope: Extend the validated realization pilot into a two-stage TEM design that uses COMPOT at the project level and as a boundary condition for partner realization.

## 1. Objective

Upgrade the empirical architecture from a single partner-realization analysis to a two-stage design:

1. **Stage A — Collaboration emergence:** among the 6,536 projects in the strict analytic sample, test whether project commercial potential (`compot`) is associated with whether a firm participates.
2. **Stage B — Partner realization:** among the 1,881 firm-participation projects, test how cognitive/technical fit and relational embeddedness jointly predict which firm becomes a realized partner, and whether COMPOT moderates those candidate-level relationships.

The 400-project realization pilot remains the validation sample for Stage B specification development. Full 1,881-project extraction is not launched until the COMPOT moderation specification is frozen.

## 2. Research questions

### Stage A

> Are higher-COMPOT scientific projects more likely to cross the boundary into firm participation?

This is a project-level extensive-margin question. The unit of analysis is one scientific project.

### Stage B

> Conditional on firm participation, how do cognitive/technical fit and relational embeddedness predict which firm becomes a realized partner, and does project commercial potential alter these matching logics?

This is a project × candidate-firm realization question. COMPOT is constant within each project, so its main effect is not identified in project-stratified conditional-logit models; only interactions with candidate-varying predictors are identified.

## 3. Sample architecture

### 3.1 Stage A population

Use all **6,536 unique projects** in `author_project_panel_v3_strict.csv`.

Project-level outcome:

- primary source: the strict panel's project-consistent `firm_participation` field;
- cross-check: `firm_participation = 1` must correspond exactly to membership in the validated 1,881-project Stage 2 universe.

The Stage A dataset must contain one row per `work_id`. If the strict panel contains multiple focal-author rows for a project, project-level fields must be checked for within-project consistency before deduplication.

**Selection boundary:** Stage A is conditional on entry into the V3 strict analytic sample. The 6,536 projects are not the universe of all scientific projects; they have already passed the upstream focal-author/score-matching eligibility process and have observed COMPOT. Stage A conclusions therefore describe variation in firm participation **within this strict eligible project population** and must not be generalized to projects excluded before the strict-sample stage without a separate selection analysis.

### 3.2 Stage B population

Use the validated **1,881 unique firm-participation projects** from `risk_set_full_firm_projects_input_v3.csv`.

Development/validation sample:

- deterministic 400-project pilot selected using the existing period × field × within-period COMPOT-quartile stratification and seed `20260804`.

Confirmatory sample:

- all 1,881 projects after the COMPOT interaction specification is frozen.

## 4. Stage A empirical design

### 4.1 Core variables

Outcome:

- `firm_participation`

Main predictor:

- continuous standardized `compot`

Descriptive transformations:

- COMPOT quartiles
- optional deciles for visualization only

Controls available from the validated panel should be restricted to clearly pre-project or contemporaneous project attributes that do not mechanically encode firm participation. At minimum, include publication-year and primary-field fixed effects where available.

### 4.2 Model sequence

Estimate the following prespecified sequence:

- **A0 — Descriptive:** firm-participation rate by COMPOT quartile; mean/median COMPOT by participation status.
- **A1 — Linear probability model:** `firm_participation ~ compot_z + year FE + field FE`.
- **A2 — Logit robustness:** same specification using binary logistic regression.
- **A3 — Nonlinearity robustness:** replace linear `compot_z` with COMPOT quartile indicators.

Use heteroskedasticity-robust standard errors. If a valid focal-author clustering identifier can be constructed without ambiguity for multi-author projects, report clustered standard errors as a robustness check; otherwise retain project-level robust standard errors and state why author clustering is not uniquely defined.

### 4.3 Stage A interpretation

Treat the results as associational and conditional on V3 strict-sample eligibility. Do not claim that COMPOT causally causes firm participation or that the estimated association represents projects excluded by the upstream sample-construction process.

Permitted interpretation:

> Among projects in the V3 strict eligible sample, projects with higher measured commercial potential are more/less likely to involve firm participation, conditional on observed year and field differences.

## 5. Stage B COMPOT moderation design

### 5.1 Existing baseline

Preserve the validated temporal ranking framework:

- technical model
- relational model
- combined model
- prepublication-only inputs
- train on projects through 2018
- evaluate partner ranking on 2019–2024 projects
- Top 50 main candidate specification
- Top 100 sensitivity specification

### 5.2 Required COMPOT features

Standardize COMPOT using **training-period projects only** and map the project-level scaling back to candidate rows.

Create:

- `compot_z`
- `fit_x_compot = cognitive_fit_z × compot_z`
- `author_x_compot = author_prior_partner × compot_z`
- `university_x_compot = university_prior_partner × compot_z`
- `strong_university_x_compot = strong_university_candidate × compot_z`

Do not include `compot_z` as a standalone term in project-stratified conditional logit because it has no within-project variation.

### 5.3 Model sequence

Preserve the current combined model as the baseline and add two prespecified moderation models:

- **B0 — Combined baseline:** technical + relational + existing fit × relationship interactions.
- **B1 — Relationship moderation:** B0 + author × COMPOT + university × COMPOT + strong-university × COMPOT.
- **B2 — Fit moderation:** B1 + cognitive fit × COMPOT.

A three-way `fit × relationship × COMPOT` specification is not part of the primary model set. It may be added only after B1/B2 are estimated and only if the two-way interaction pattern gives a specific theoretical reason for it.

### 5.4 Ranking evaluation

For B0/B1/B2, report out-of-sample:

- mean reciprocal rank
- Recall@5
- Recall@10
- average precision
- mean selected-partner rank / best selected rank

Use the same temporal split, project eligibility rules, tie-breaking, and bootstrap procedure as the existing pilot so model comparisons are apples-to-apples.

The COMPOT extension is considered predictively informative when B1 or B2 improves a ranking metric beyond B0 by a substantively nontrivial amount and the direction is consistent across Top 50 and Top 100.

Do not require predictive improvement for COMPOT to be theoretically useful if conflict heterogeneity is stable and interpretable.

## 6. COMPOT conflict-set analysis

Use the existing project-level realization-gap outputs and add COMPOT heterogeneity.

For each Top 50 and Top 100 specification, report by COMPOT quartile:

- number of projects
- natural project recall
- selected-firm cognitive-fit coverage
- relation–fit conflict share at thresholds 0.02, 0.05, and 0.10
- mean `fit_shortfall`
- mean selected-fit percentile
- share with no selected firm in the top fit decile

Primary theoretical contrast:

- **COMPOT weakens relational advantage:** higher-COMPOT projects are more likely to realize high-fit unembedded firms or show lower relationship-induced fit shortfall.
- **COMPOT strengthens relational advantage:** higher-COMPOT projects show stronger selection of embedded firms despite higher-fit unembedded alternatives, consistent with greater assurance/appropriation/coordination concerns.

These interpretations remain associational.

## 7. Validation gate before full 1,881 expansion

The 400-project COMPOT validation is complete when all of the following are true:

1. Stage A 6,536-project dataset passes uniqueness, missingness, and fixed-count checks, and the strict-panel `firm_participation` field yields exactly 1,881 positive projects matching the Stage 2 universe.
2. Stage A descriptive and regression outputs are generated with no unexplained sample loss.
3. Stage B candidate tables retain nonmissing COMPOT for all 400 projects and candidate rows.
4. B0/B1/B2 models run on the same eligible temporal train/test projects unless a model-specific mathematical redundancy is explicitly reported.
5. COMPOT quartile conflict tables are produced for Top 50 and Top 100.
6. Top 50/Top 100 moderation directions are documented, including null findings.
7. No post-publication evidence enters any Stage B predictor.

The gate does **not** require a statistically significant COMPOT coefficient or interaction. Null COMPOT moderation is an allowed result and implies that COMPOT primarily belongs to Stage A rather than Stage B.

## 8. Full 1,881 expansion rule

Only after the 400-project COMPOT validation is reviewed and the model set is frozen:

- run the exact same B0/B1/B2 specification on all 1,881 firm-participation projects;
- preserve Top 50 as main and Top 100 as sensitivity;
- use restartable OpenAlex caches;
- shard the full run rather than placing all 1,881 projects in one job;
- merge shard outputs before model estimation and final reporting;
- do not change interaction definitions after seeing the full-sample results except for clearly labeled robustness analyses.

## 9. Intended paper architecture

The empirical narrative becomes:

### Stage A — Collaboration emergence

`Strict-eligible scientific project → COMPOT → whether industry participates`

### Stage B — Partner realization

`Firm-involved scientific project → technical/cognitive fit + relational embeddedness + COMPOT moderation → which firm is realized`

The central paper contribution remains the distinction between potential matching and realized partnering, now with an explicit upstream extensive margin inside the validated strict-sample population.

## 10. Required implementation components

Create focused additions rather than rewriting the validated pilot pipeline:

1. A Stage A analysis module that collapses the strict panel to one project row, verifies within-project consistency, uses the panel's `firm_participation` outcome, cross-checks it against the 1,881 Stage 2 universe, audits counts, estimates A0–A3, and writes machine-readable outputs.
2. A Stage B COMPOT extension to the temporal ranking module that creates train-scaled COMPOT interactions and estimates B0–B2.
3. A COMPOT heterogeneity module or functions that summarize conflict/realization metrics by quartile for Top 50 and Top 100.
4. Regression tests covering:
   - project-level Stage A collapse and 6,536 count contract;
   - exact 1,881 positive outcomes and exact agreement with the V3 Stage 2 project universe;
   - COMPOT main effect excluded from conditional-logit terms;
   - interaction values vary across candidate firms when the candidate-level component varies;
   - scaling fitted on training projects only;
   - Top 50/Top 100 output schemas;
   - no mutation of the existing baseline B0 behavior.
5. A lightweight GitHub Actions validation workflow that reuses existing artifacts/candidate outputs where possible and does not re-fetch OpenAlex data solely to validate COMPOT interactions.

## 11. Output contract

Minimum new outputs:

### Stage A

- `stageA_project_panel.csv`
- `stageA_sample_audit.json`
- `stageA_compot_descriptives.csv`
- `stageA_compot_models.csv`
- `stageA_summary.json`

### Stage B COMPOT validation

For Top 50 and Top 100:

- `pilot_compot_ranking_coefficients_<spec>.csv`
- `pilot_compot_ranking_metrics_<spec>.csv`
- `pilot_compot_conflict_by_quartile_<spec>.csv`

Combined summary:

- `pilot_compot_validation_summary.json`

The summary must explicitly state whether COMPOT is supported as:

- Stage A only;
- Stage A + relationship moderator;
- Stage A + fit moderator;
- Stage A + both moderators;
- neither Stage A nor Stage B substantive signal.

A null classification is valid and must not be converted into a positive claim.

## 12. Error handling and reproducibility

- Fail on duplicate project IDs after Stage A collapse.
- Fail if project-level fields used in Stage A are inconsistent across duplicate focal-author rows.
- Fail if COMPOT is missing for any Stage A project used in primary models or any Stage B pilot project.
- Fail if the reconstructed Stage A project count is not 6,536, if positive Stage A outcomes are not 1,881, or if positive outcomes do not exactly match the Stage 2 universe.
- Preserve the existing 400-project sample seed and signature.
- Preserve time-provenance checks for every candidate-level predictor.
- Reuse cached/persisted pilot candidate outputs when schema and sample signature match.
- Treat software failure separately from substantive null results.

## 13. Claims and boundaries

Permitted:

- Within the V3 strict eligible population, COMPOT is associated with the emergence of firm participation.
- COMPOT conditions the predictive association between candidate-level fit/relationships and realized partnering.
- Technical compatibility and relational embeddedness provide distinct or interacting signals for realized partner choice.

Not permitted:

- COMPOT causally causes industry entry without an identification strategy.
- Stage A estimates automatically generalize to projects excluded before V3 strict-sample construction.
- A higher cognitive-fit firm is objectively the economically optimal partner.
- A relation–fit conflict is evidence of inefficiency.
- The constructed candidate set is the directly observed managerial consideration set.

## 14. Design decision

Recommended path:

1. implement and run Stage A on all 6,536 strict-eligible projects;
2. implement B0/B1/B2 COMPOT validation using the existing 400-project candidate/cognitive outputs without new OpenAlex extraction whenever valid cached artifacts can be reused;
3. review the 400-project moderation and conflict heterogeneity results;
4. freeze the full specification;
5. execute the sharded 1,881-project confirmatory expansion.

This sequence minimizes API cost, avoids specification drift, and makes the final 1,881-project run confirmatory rather than exploratory.
