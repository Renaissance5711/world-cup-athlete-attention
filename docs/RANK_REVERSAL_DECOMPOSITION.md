# Rank-reversal decomposition across athlete-attention estimands

## Purpose

This analysis quantifies how often the primary observed proportional and additive attention estimands order the same 117 World Cup scorers differently. It separates three objects that should not be conflated:

1. **Observed ordering changes** between the two athlete-level estimands used in the study.
2. **A denominator accounting diagnostic** based on a reconstructed ratio score.
3. **Aggregation residual disagreement** between the reconstructed ratio ordering and the primary proportional ordering.

The arithmetic identity is an accounting device, not the study's mathematical contribution; the empirical contribution is the magnitude and distribution of observed ordering changes in the athlete sample.

## Inputs and population

The runner uses:

- `outputs/r24/observed_goal_scorer_metric_rankings.csv` for the frozen athlete-level rankings;
- `base_archive/data/processed/all_player_match_outcomes_2022.csv` for scoring appearances used in the bootstrap.

The population contains 117 unique scorers. Athlete-level values retain the existing aggregation rule:

- proportional response: mean `immediate_attention_log_lift` across scoring appearances;
- additive response: mean `winsorized_additional_pageviews` across scoring appearances;
- baseline: the stored athlete-level `baseline_views` value.

No athlete is removed to improve agreement or interpretability.

## Observed pairwise decomposition

For every unordered athlete pair, the analysis compares the sign of the difference in the observed proportional score with the sign of the difference in the observed additive score.

A pair is:

- **concordant** when both estimands order the athletes in the same direction;
- **reversed** when the estimands order them in opposite directions;
- **tied** when either observed estimand has an exact stored-precision tie.

Tied pairs remain in the total accounting and are excluded only from the concordant-versus-reversed rate denominator.

The frozen sample contains 6,786 unordered pairs. Six pairs are tied, leaving 6,780 comparable pairs. Of these, 3,058 are reversed and 3,722 are concordant, producing an observed reversal rate of 45.1%.

## Denominator accounting diagnostic

For athlete `i`, the diagnostic reconstructs a proportional score as:

```text
R_i = log1p(D_i / B_i)
```

where `D_i` is the athlete's observed additive response and `B_i` is the positive baseline. For a pair oriented so that `D_i > D_j`, the reconstructed ordering reverses the additive ordering exactly when:

```text
log(B_i / B_j) > log(D_i / D_j)
```

The difference between these two log gaps is stored as `dominance_margin`. A positive margin identifies a denominator-diagnostic reversal.

This condition is not presented as a new theorem. It makes the baseline dependence of the reconstructed ratio ordering transparent.

## Aggregation residual

The primary proportional score is a mean of scoring-appearance log lifts. The additive score is a mean of winsorized additional pageviews. Consequently, the primary proportional ordering is not algebraically identical to the athlete-level reconstructed score `log1p(D/B)`.

The denominator diagnostic agrees with the observed concordant-versus-reversed classification for 6,441 of 6,780 comparable pairs (95.0%). The remaining 339 pairs (5.0%) are reported as aggregation residual disagreement. They reflect the combined consequences of appearance-level aggregation, nonlinear transformation, and winsorization; they are not forced into a denominator-only explanation.

## Athlete-level displacement

For each athlete:

```text
rank_displacement = additive_rank - proportional_rank
```

Negative values mean that the athlete ranks higher on the additive scale; positive values mean that the athlete ranks higher on the proportional scale. The absolute displacement has a median of 34 ranks, and 75 of 117 athletes move at least 25 positions.

## Match-weighted Bayesian bootstrap

Uncertainty is assessed with 10,000 draws using seed `20260730`. Each draw assigns Dirichlet weights to matches. The same match weight is applied to every scoring appearance in that match, after which athlete-level proportional and additive means are recomputed and all unordered pairs are reclassified.

The bootstrap is conditional on the recorded scoring appearances. It does not represent a superpopulation causal design.

Key intervals are:

- observed reversal rate: bootstrap 95% interval 43.9%-45.7%;
- identity-observed agreement: bootstrap 95% interval 94.7%-95.8%.

## Interpretation boundary

The results support a measurement and decision-design claim: the two estimands act as materially different athlete-selection rules in the observed sample. They do not establish that choosing a metric causes attention, sponsorship value, or activation effectiveness. Temporal persistence and shooter-goalkeeper relational patterns are separate empirical boundaries and are not implied by the denominator identity.

## Reproduction

From the repository root:

```bash
python code/run_rank_reversal_decomposition.py
```

This writes:

- `outputs/r25/rank_reversal_pairs.csv` (generated locally; not versioned because it is a complete pair-level derivative);
- `outputs/r25/rank_reversal_athlete_displacements.csv` (versioned compact audit output);
- `outputs/r25/rank_reversal_bootstrap_draws.csv` (generated locally; not versioned because all 10,000 draws can be reproduced from the fixed seed);
- `outputs/r25/rank_reversal_decomposition_table.csv` (versioned submission-table source);
- `outputs/r25/rank_reversal_summary.json` (versioned machine-readable summary).

Run the focused tests with:

```bash
python -m pytest tests/test_rank_reversal.py tests/test_rank_reversal_integration.py -q
```
