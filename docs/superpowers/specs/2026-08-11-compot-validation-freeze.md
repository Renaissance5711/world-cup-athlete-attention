# TEM COMPOT Validation Freeze Decision

Date: 2026-08-11  
Branch: `tem-stage2-v5-run`  
Validation workflow: `TEM Stage 2 COMPOT Validation` run `31463205884`  
Validation artifact: `tem-stage2-compot-validation-31463205884-1`

## Decision

Freeze COMPOT as a **Stage A project-level predictor / upstream boundary condition**, not as a primary Stage B partner-realization moderator.

Classification: `STAGE_A_ONLY`.

This decision was made using the prespecified 6,536-project Stage A analysis and the existing 400-project Stage B pilot before the 1,881-project confirmatory expansion.

## Stage A result

Population: 6,536 V3 strict-eligible scientific projects, including exactly 1,881 projects with firm participation.

- Mean COMPOT, firm participation = 1: 0.44729.
- Mean COMPOT, firm participation = 0: 0.40940.
- Firm-participation rate by COMPOT quartile: Q1 27.48%, Q2 26.81%, Q3 27.36%, Q4 33.48%.
- A1 LPM: standardized COMPOT coefficient = 0.03471, HC1 SE = 0.00592, p = 4.51e-09.
- A2 logit: standardized COMPOT coefficient = 0.17498, HC1 SE = 0.02969, p = 3.80e-09; odds ratio approximately 1.191 per one-SD increase in COMPOT.
- A3 quartile LPM relative to Q1: Q2 = 0.01404 (p = 0.389), Q3 = 0.03430 (p = 0.0429), Q4 = 0.08886 (p = 1.26e-07), conditional on year and field controls.

Interpretation is associational and conditional on the V3 strict-eligible project population. It must not be generalized to all scientific projects and must not be described causally.

## Stage B moderation result

Population: validated deterministic 400-project realization pilot. Top 50 is main; Top 100 is sensitivity. Temporal training uses projects through 2018 and evaluates 2019–2024 projects.

### Relationship × COMPOT

No stable significant moderation term across Top 50 and Top 100.

- `author_x_compot`: Top50 -0.0238 (SE 0.2128); Top100 0.0108 (SE 0.2105).
- `university_x_compot`: Top50 0.3592 (SE 0.3752); Top100 0.3454 (SE 0.3696).
- `strong_university_x_compot`: Top50 -0.4120 (SE 0.2716); Top100 -0.3905 (SE 0.2713).

The strong-university interaction is directionally negative in both candidate definitions but does not meet the prespecified stability/significance gate.

### Cognitive fit × COMPOT

No stable significant moderation.

- `fit_x_compot`: Top50 -0.0613 (SE 0.0707); Top100 -0.0250 (SE 0.0622).

### Predictive comparison

Adding COMPOT interactions does not improve the validated combined ranking baseline.

Top50:
- B0 MRR 0.80821.
- B1 MRR 0.78744, delta -0.02077.
- B2 MRR 0.79002, delta from B1 +0.00258 and still below B0.
- B0 Recall@10 0.85290; B1 0.84785; B2 0.84533.

Top100:
- B0 MRR 0.79248.
- B1 MRR 0.78820, delta -0.00428.
- B2 MRR 0.78109, delta from B1 -0.00711.
- B0 Recall@10 0.85732; B1 0.85480; B2 0.85480.

Therefore COMPOT interactions are not promoted into the primary Stage B ranking model.

## Conflict heterogeneity

Conflict shares show some descriptive variation by COMPOT quartile but not a clean monotonic moderation pattern.

At the primary 0.05 threshold:
- Top50: Q1 0.14, Q2 0.17, Q3 0.13, Q4 0.23.
- Top100: Q1 0.17, Q2 0.21, Q3 0.18, Q4 0.25.

Q4 is descriptively higher than Q1 in both candidate definitions, but the middle quartiles are nonmonotonic. Treat this as descriptive heterogeneity / robustness rather than primary evidence of moderation.

## Frozen full-sample specification

### Stage A main analysis

Use all 6,536 V3 strict-eligible projects:

1. Descriptive firm-participation rates by COMPOT quartile.
2. A1 LPM: `firm_participation ~ compot_z + year FE + field FE`, HC1 robust SE.
3. A2 logit robustness with the same covariates.
4. A3 COMPOT quartile robustness.

### Stage B main analysis

Use all 1,881 firm-participation projects with the validated realization architecture:

1. Technical model.
2. Relational model.
3. Combined B0 model with technical + relational predictors and the already-validated fit × relationship terms.
4. Top50 primary candidate definition; Top100 sensitivity.
5. Temporal out-of-sample evaluation and project-stratified conditional logit.
6. Realization-gap and relation-fit conflict analyses.

COMPOT is not added as a standalone Stage B term and COMPOT interactions are not part of the primary Stage B model.

### Stage B secondary prespecified robustness

Retain B1/B2 COMPOT interactions as a clearly labeled secondary full-sample robustness test because they were prespecified before the 1,881-project run. Do not change interaction definitions after observing full-sample results. A larger-sample signal may be reported as secondary evidence, but it cannot retroactively replace B0 as the primary Stage B specification without a separate theoretical justification.

## Next execution rule

Proceed to the sharded 1,881-project confirmatory expansion. The full run must preserve:

- prepublication-only evidence;
- the validated Top50/Top100 candidate definitions;
- B0 as the primary realization model;
- B1/B2 only as prespecified secondary robustness;
- restartable OpenAlex cache/checkpoints;
- deterministic shard assignment and deterministic final merge;
- a final global provenance audit after shard merge.

No further exploratory changes to COMPOT are permitted before the full-sample confirmatory results are produced.
