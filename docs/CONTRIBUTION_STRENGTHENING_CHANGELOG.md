# Contribution-strengthening change log

## Purpose

This change set adds a full-ranking decomposition of disagreement between the primary observed proportional and additive athlete-attention estimands. It supports a manuscript revision that treats the estimands as different decision targets rather than presenting their arithmetic relationship as a mathematical innovation.

## Analysis additions

- Pairwise classification of all 6,786 unordered pairs among 117 observed scorers.
- Separate accounting for concordant, reversed and tied pairs.
- Athlete-level signed and absolute rank displacement.
- A denominator diagnostic based on `log1p(additional_pageviews / baseline_views)`.
- Explicit reporting of residual disagreement attributable to aggregation, nonlinear transformation and winsorisation.
- Match-weighted Bayesian bootstrap using shared Dirichlet match weights, seed `20260730`, and 10,000 draws.

## Verified frozen results

- 6,780 comparable pairs and six tied pairs.
- 3,058 observed reversals: 45.1% of comparable pairs.
- Bootstrap median reversal rate 44.8%, 95% interval 43.9%-45.7%.
- Denominator-diagnostic agreement with the primary observed pair classification: 95.0%.
- Median absolute athlete rank displacement: 34 positions.
- 75 of 117 athletes move at least 25 positions.

## Interpretation boundaries

- The arithmetic denominator relationship is an accounting identity, not a new theorem.
- The analysis is descriptive and does not estimate the causal effect of metric choice.
- Pageviews screen candidates for further evaluation; they do not establish brand equity or activation return.
- Full pair-level and draw-level derivatives are deterministic and generated locally rather than versioned.

## Verification

```bash
python -m pytest -q -W error::FutureWarning
sha256sum -c outputs/r25/SHA256SUMS_R25.txt
```
