# Rank-reversal decomposition and manuscript contribution redesign

Date: 2026-08-01
Status: approved design
Repository: `Renaissance5711/world-cup-athlete-attention`
Target manuscript: IJSMS submission on athlete attention following the 2022 FIFA World Cup

## 1. Purpose

This design strengthens the manuscript without presenting the arithmetic distinction between proportional and additive change as a new mathematical result. The revised contribution is empirical and managerial: different attention estimands operate as different athlete-selection rules, and the study quantifies how often, for whom and by how much those rules reverse the ordering of real athletes.

The new analysis explains the weak agreement already observed between the proportional and additive scorer rankings. It does not attempt to establish a causal effect of metric choice, nor does it treat the denominator effect as a theoretical discovery.

## 2. Approved manuscript framing

The hypothesis structure will be revised as follows:

- **H1** remains the immediate-attention hypothesis associated with scoring.
- **Estimand contrast** replaces the former H2a and H2b. It states that proportional response and additional pageview volume answer different decision questions and therefore need not identify the same activation candidates. This is a measurement and decision-design proposition, not a directional hypothesis.
- **H2** is the former H3 and concerns the temporal persistence or decay of attention.
- **H3** is the former H4 and concerns the relational comparison between shooter and goalkeeper attention.

All hypothesis references in the introduction, methods, results, tables, figure captions and discussion will be renumbered consistently.

The manuscript will distinguish three levels of claim:

1. **Arithmetic identity:** for a given absolute increment, proportional change depends on the baseline denominator.
2. **Empirical ranking consequence:** in the observed scorer sample, the two estimands produce a measurable number of pairwise rank reversals and large athlete-level rank displacements.
3. **Non-arithmetic boundaries:** persistence over time and shooter-goalkeeper relational patterns are empirical results that are not implied by the denominator identity.

## 3. Primary analysis population

The primary population is the same set of unique scorers used in the existing observed-ranking analysis. The current pipeline reports 117 unique scorers. The aggregation rule must remain identical to the existing ranking analysis, currently described as the mean across scoring appearances.

No athlete may be added or removed solely to improve agreement, statistical significance or interpretability. Any exclusion required by the mathematical decomposition must be reported explicitly.

## 4. Estimands and notation

For athlete `i`:

- `B_i` is the pre-event attention baseline used by the existing analysis.
- `D_i` is the additive attention response, measured using the same additional-pageview definition used in the manuscript.
- `P_i` is the proportional attention response used in the existing observed ranking.

When the proportional response is represented as `D_i / B_i`, or by a monotone transformation such as `log(1 + D_i / B_i)`, the ordering is the same for positive baselines. The implementation must use the exact manuscript variable rather than silently substituting a new metric.

For an unordered athlete pair `(i, j)`, the pair is:

- **concordant** when additive and proportional estimands order the two athletes in the same direction;
- **reversed** when the two estimands order them in opposite directions;
- **tied** when either estimand does not distinguish the pair under the exact stored precision.

The primary reversal indicator is calculated directly from the observed estimands:

```text
reversal_ij = sign(D_i - D_j) != sign(P_i - P_j)
```

Pairs involving a tie are retained in the accounting but excluded from the simple concordant-versus-reversed denominator. Tie counts must be reported.

## 5. Exact decomposition for positive observations

For positive baselines and positive additive responses, orient each pair so that `D_i > D_j`. Define:

```text
increment_gap_ij = log(D_i / D_j)
baseline_gap_ij  = log(B_i / B_j)
dominance_margin_ij = baseline_gap_ij - increment_gap_ij
```

A proportional-order reversal occurs exactly when:

```text
baseline_gap_ij > increment_gap_ij
```

Equivalently, the larger additive response belongs to an athlete whose baseline advantage is sufficiently large to outweigh the athlete's additive-response advantage on the proportional scale.

This inequality is not presented as a novel theorem. It is used as an accounting identity that makes the empirical source of each observed reversal transparent.

## 6. Primary reported quantities

The decomposition will report:

1. total unordered athlete pairs;
2. valid comparison pairs;
3. additive ties, proportional ties and double ties;
4. concordant pairs;
5. reversed pairs;
6. reversal share among non-tied valid pairs;
7. Kendall-style pairwise agreement, computed from concordant and reversed pair counts and reported with the exact tie convention;
8. the distribution of `baseline_gap`, `increment_gap` and `dominance_margin` among reversed pairs;
9. the proportion of reversed pairs in which the baseline ratio exceeds specified descriptive thresholds, chosen before viewing the final result table;
10. athlete-level rank displacement between additive and proportional rankings;
11. athlete-level reversal burden, defined as the number and share of pairwise comparisons whose ordering changes for that athlete.

The main text will emphasize reversal share and athlete-level rank displacement. Detailed pairwise distributions may be placed in supplementary or repository outputs if space is constrained.

## 7. Zero, negative and non-finite values

The existing pipeline contains structural-zero handling. The decomposition must not hide these cases.

- The direct reversal indicator can be calculated whenever both stored estimands are finite, regardless of whether the log-gap identity is available.
- The exact log-gap decomposition is restricted to pairs for which the required baselines and additive responses are strictly positive.
- Athletes with a structural-zero baseline, non-positive additive response or non-finite value are reported as boundary cases.
- The primary manuscript table will show how many athletes and pairs enter the direct ranking comparison and how many also enter the positive-value exact decomposition.
- A sensitivity analysis will apply the manuscript's existing zero-coding or continuity convention, if one exists, without introducing an undocumented replacement rule.

If only one scorer is affected by structural-zero coding, that athlete must still be described explicitly in the diagnostic output because its pairwise influence can be large.

## 8. Uncertainty analysis

The decomposition is descriptive, but sampling uncertainty will be assessed using the same match-weighted Bayesian bootstrap framework already used for the observed ranking analysis.

For each bootstrap draw:

1. draw shared match weights using the existing procedure;
2. recompute athlete-level additive and proportional aggregates using the established aggregation rule;
3. recompute valid pair counts, reversal share, pairwise agreement and selected rank-displacement summaries;
4. retain the same zero and tie rules as the observed analysis.

Report the bootstrap median and 95% interval for reversal share and pairwise agreement. The observed value remains the primary estimate. The bootstrap is conditional on the recorded scoring appearances and must not be described as population-level causal uncertainty.

## 9. Counterfactual benchmark

A common-baseline benchmark may be included as a short explanatory sensitivity check. When every athlete is assigned the same positive baseline, proportional and additive ordering coincide by construction.

This benchmark is an identity demonstration, not evidence. It should be used only to clarify that the empirical question is the magnitude and distribution of rank reversal under actual heterogeneous baselines.

## 10. Software design

### 10.1 New analysis module

Create a focused module, provisionally:

```text
ijsms/rank_reversal.py
```

Recommended public functions:

```text
classify_pairwise_order(...)
compute_pairwise_decomposition(...)
summarize_rank_displacement(...)
bootstrap_rank_reversal(...)
```

Each function must have one responsibility and return tidy data or a documented summary object.

### 10.2 Runner

Create:

```text
code/run_rank_reversal_decomposition.py
```

The runner will:

- load the same scorer-appearance input used by the existing observed-ranking code;
- apply the same athlete aggregation;
- call the decomposition module;
- write deterministic outputs;
- print a concise reproducibility summary.

### 10.3 Outputs

Write outputs under:

```text
outputs/rank_reversal/
```

Minimum files:

```text
rank_reversal_summary.json
rank_reversal_pair_counts.csv
rank_reversal_gap_summary.csv
athlete_rank_displacement.csv
rank_reversal_bootstrap_summary.json
```

A full athlete-pair file may be generated locally but should not be committed if it is unnecessarily large or creates disclosure concerns. The summary outputs must be sufficient to reproduce every number reported in the manuscript.

### 10.4 Documentation

Create:

```text
docs/analysis/rank_reversal_decomposition.md
```

The documentation will define all variables, pair orientation, tie handling, zero handling, bootstrap procedure and manuscript mapping.

## 11. Testing

Create focused tests, provisionally:

```text
tests/test_rank_reversal.py
```

Required test cases:

1. a concordant positive pair;
2. an exact reversed pair satisfying `baseline_gap > increment_gap`;
3. a boundary case where the two gaps are equal;
4. additive and proportional ties;
5. a structural-zero baseline handled without an unreported deletion;
6. non-positive additive responses handled by direct ordering but excluded from the log-gap identity;
7. invariance of proportional ordering under the monotone transformation used in the manuscript;
8. deterministic output under a fixed bootstrap seed;
9. agreement between pair counts and athlete-level reversal burden totals;
10. a synthetic complete ranking whose reversal count is known exactly.

All existing tests must continue to pass.

## 12. Manuscript integration

### Introduction and theory

- Remove any wording that frames denominator sensitivity as mathematical innovation.
- Replace H2a/H2b with a titled `Estimand contrast` subsection.
- State the two decision questions explicitly:
  - proportional response identifies athletes experiencing the largest relative departure from prior visibility;
  - additive response identifies athletes generating the largest additional attention volume.
- Explain that neither estimand is universally superior; appropriateness depends on the activation objective.

### Methods

Add a subsection titled:

```text
Pairwise decomposition of estimand-dependent rank reversals
```

It will state the population, aggregation, direct reversal rule, positive-value identity, tie and zero handling, and bootstrap method.

### Results

Add a subsection titled:

```text
Decomposing rank reversals across attention estimands
```

The results must first report observed full-ranking disagreement, then the pairwise reversal share, then the baseline-versus-increment decomposition. Mathematical interpretation must follow, not precede, the empirical quantities.

One compact main-text table is permitted. The preferred table contains pair counts, reversal share, agreement, bootstrap intervals and selected dominance-margin summaries. No new main-text figure will be added unless the analysis reveals a pattern that cannot be communicated clearly in the table.

### Discussion

The revised contribution statement will separate:

- the known arithmetic dependency of proportional change on baseline;
- the new empirical quantification of how that dependency changes athlete ordering in a real mega-event setting;
- the decision implication that candidate identification must follow the activation objective;
- the temporal and relational findings that extend beyond the arithmetic relationship.

The discussion must avoid claiming that the decomposition proves causality or that it introduces a new mathematical method.

## 13. Table and hypothesis renumbering

The final manuscript will use consecutive hypotheses:

```text
H1 -> Estimand contrast -> H2 -> H3
```

The exact table placement will be determined after results are available. If a new main-text table is inserted, all table numbers and all in-text references must be updated mechanically and verified. Figure numbering is expected to remain unchanged.

## 14. Integrity and quality gates

Before the revised manuscript is finalized:

1. every reported decomposition number must trace to a committed summary output;
2. formulas in the manuscript, code and documentation must use the same orientation and notation;
3. all athlete and pair counts must reconcile across JSON, CSV and manuscript text;
4. all hypothesis and table references must be continuous and correct;
5. no citation may be added without source verification;
6. claims of novelty must be restricted to what the evidence supports;
7. the revised DOCX must be rendered and inspected page by page;
8. a focused re-review must test whether the denominator-effect objection remains a major weakness.

## 15. Non-goals

This work will not:

- claim a new mathematical theorem;
- replace the primary full-ranking analysis with a top-K-only analysis;
- add new data sources unless implementation reveals a blocking data gap;
- change the existing attention windows or scorer inclusion rules without a separately approved amendment;
- make causal claims about sponsorship returns, consumer behavior or commercial value;
- upload the anonymous manuscript or submission figures to the public repository.

## 16. Success criteria

The redesign succeeds when a skeptical reviewer can see, from one transparent analysis, all of the following:

1. the denominator relationship is openly acknowledged as arithmetic;
2. the empirical magnitude of real-sample rank reversal is quantified;
3. the athletes most affected by estimand choice are identifiable;
4. the result leads to a precise decision rule for sport managers;
5. the manuscript's temporal and shooter-goalkeeper contributions remain logically independent of the arithmetic identity;
6. the analysis is reproducible from committed code, tests and compact outputs.

## 17. Academic Research Skills pipeline mapping

This project is a mid-entry manuscript workflow. The approved sequence is:

1. implementation planning for the approved decomposition;
2. analysis implementation and reproducibility validation;
3. contribution-focused manuscript revision;
4. full reviewer simulation including Devil's Advocate review;
5. focused re-revision if required;
6. final citation, data and document integrity check;
7. DOCX finalization and visual inspection.

Mandatory user checkpoints will be retained before material changes to the study design, after the first reviewer simulation and before finalization.