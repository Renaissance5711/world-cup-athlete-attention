# TEM Full 1,881-Project Sharded Expansion Plan

**Goal:** Execute the frozen TEM Stage 2 specification on all 1,881 firm-participation projects without exceeding a single GitHub-hosted job limit, while preserving deterministic sampling, restartability, prepublication provenance, and global model estimation.

## Architecture

Use five extraction shards of sizes 376/376/376/377/376. Shards are balanced within the existing period × primary-field × within-period COMPOT-quartile strata. Each shard runs the validated realization extraction logic but does not define the final empirical model. A merge job concatenates all shard candidate-level outputs and estimates the global technical, relational, combined B0, and secondary COMPOT B1/B2 models on the complete 1,881-project panel.

The old monolithic V5 workflow remains retired. The successful 400-project pilot workflow becomes manual-only.

## Task 1 — Deterministic full-sample shard assignment

Create:
- `.github/overlays/stage2_full_sharding.py`
- `.github/overlays/test_stage2_full_sharding.py`

Requirements:
- exactly five shards;
- exactly 1,881 unique projects in full production input;
- shard sizes differ by at most one;
- within each period × field × COMPOT-quartile stratum, shard counts differ by at most one;
- deterministic under input row reordering;
- every shard contains both pre-2019 and 2019+ projects when the full V3 population does;
- emit `full_shard_assignment.csv` and audit JSON.

## Task 2 — Extraction-only shard runner

Create:
- `.github/overlays/run_stage2_realization_shard.py`
- `.github/overlays/test_stage2_realization_shard.py`

Import validated helper functions from `run_stage2_realization_pilot.py` and the UTC/singleton fixes from `run_stage2_realization_pilot_v2.py`. Do not rewrite OpenAlex extraction logic.

Inputs:
- full 1,881 project file;
- full strict panel;
- shard index 0–4;
- cache directory;
- output directory.

Outputs per shard:
- shard project list/config/audit;
- project details CSV;
- Top50/Top100 natural candidate audit;
- Top50/Top100 cognitive candidate long CSV;
- project realization metrics Top50/Top100;
- time-provenance audit;
- compact shard summary.

The raw candidate text-history CSV may be retained locally for restartability during the job but is excluded from the uploaded result artifact after successful cognitive-fit construction.

## Task 3 — Global merge and confirmatory models

Create:
- `.github/overlays/run_stage2_full_merge.py`
- `.github/overlays/test_stage2_full_merge.py`

Merge five shard outputs and fail unless:
- union is exactly 1,881 projects for both Top50 and Top100;
- shard project sets are disjoint;
- project-company keys are unique after merge;
- Top50/Top100 project sets match;
- all shard provenance audits report zero violations.

Primary Stage B outputs:
- full cognitive candidate long Top50/Top100;
- full project realization metrics Top50/Top100;
- technical/relational/combined B0 coefficients and temporal ranking metrics;
- full gate/summary diagnostics;
- full relation-fit conflict summaries.

Secondary prespecified robustness:
- B1 relationship × COMPOT;
- B2 fit × COMPOT;
- COMPOT conflict quartile tables.

Stage A:
- recompute the frozen 6,536-project Stage A A0–A3 results directly from V3 strict input in the merge job.

## Task 4 — GitHub Actions workflow

Replace paused `.github/workflows/tem-stage2-v5.yml` with a sharded workflow.

Jobs:
1. `prepare`: reconstruct code, download/verify V3, create and upload shard assignment/input bundle, run all software tests.
2. `extract`: matrix shard `[0,1,2,3,4]`, `max-parallel: 2`, timeout 350 minutes each; restore the newest verified realization cache seed; run extraction-only shard; always save a shard-specific restartable cache checkpoint and upload compact/partial outputs.
3. `merge`: depends on all five extraction jobs; download all shard artifacts, verify success/exit codes, merge, estimate global models, run final provenance audit, upload final full-results artifact.

No shard is allowed to silently succeed when its extraction subprocess fails.

## Task 5 — Workflow isolation

Change `.github/workflows/tem-stage2-realization-pilot.yml` to `workflow_dispatch` only. The 400-project pilot has completed and should no longer rerun on every PR synchronization.

## Task 6 — Verification before full launch

TDD sequence:
- RED: sharding/runner/merge tests fail because modules do not exist.
- GREEN: focused tests pass.
- Run legacy pilot/model tests to prove no regression.
- Run workflow prepare/tests before any OpenAlex extraction.
- Only then allow the five shard matrix jobs to start.

Final success condition:
- 5/5 shard extraction jobs succeed;
- exactly 1,881 projects merged;
- final provenance violations = 0;
- global Stage B primary and COMPOT robustness outputs generated;
- final artifact uploaded;
- no claim of full-sample result until merge job succeeds.
