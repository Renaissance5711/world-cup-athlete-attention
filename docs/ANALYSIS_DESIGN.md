# IJSMS analysis design: attention scale, opportunity and role allocation

## Objective

The archive evaluates how 2022 FIFA World Cup performance is associated with English-Wikipedia information seeking. The design distinguishes four quantities that should not be treated as interchangeable: proportional attention change, additional pageview volume, persistence and role-relative visibility within shooter-goalkeeper relationships.

## Primary pageview analysis

The original study contains 680 tournament participants. Player-match and player-day fixed-effects models estimate immediate proportional and additive attention responses and their persistence through 30 days. The design treats scoring as an observed sporting outcome rather than a randomized treatment; estimates are interpreted as conditional associations.

## Shot-opportunity extension

### Unit and comparison

The near-miss unit is player-match. Scorers are compared with non-scorers who took at least one shot. Opportunity and context variables include non-penalty xG, shot count, maximum-shot xG, minutes, starter status, baseline attention, position, tournament stage, score state and team/match context.

### Estimation boundary

Overlap-weighted and sensitivity specifications improve balance on recorded opportunity. They do not eliminate differences in finishing skill, goalkeeper performance, defensive reactions or unrecorded context and therefore do not make scoring random.

## Shooter-goalkeeper role-allocation extension

### Shot construction

StatsBomb events and lineups are used to identify all on-target shots ending in a goal or goalkeeper save. Exact StatsBomb-to-Fjelstul identifiers connect shooters and goalkeepers to match records and pageview outcomes. The shot file contains 510 observations from all 64 matches, 248 shooters and 40 goalkeepers. Aggregation produces 386 shooter-goalkeeper-match pairs.

### Expected-save prediction

A regularized logistic model predicts save probability from recorded shot quality, geometry, trajectory and match context. Predictions use eight-fold cross-fitting grouped by match, so the focal match never enters the training data used to predict its shots. Predictive gates require zero match leakage, held-out AUC of at least 0.65 and a Brier score below the intercept-only benchmark.

Expected-save probability is a conditional predictive benchmark. It does not measure public responsibility, moral blame, definitive goalkeeper fault or audience sentiment.

### Primary unit and model

The primary bilateral sample contains the 303 pairs with exactly one on-target shot, including 94 goals and 209 saves. This restriction aligns the observed player-match attention outcome with one joint shooting episode.

For each role-relative attention outcome, the primary model includes:

1. a realised-goal indicator, with save as the reference outcome;
2. expected-save probability centered at its single-shot-sample mean;
3. the interaction between the goal indicator and centered expected-save probability;
4. shooter-minus-goalkeeper baseline log visibility;
5. match fixed effects; and
6. standard errors clustered by match.

The realised-goal coefficient estimates the goal-versus-save difference at the mean expected-save probability. The expected-save coefficient estimates the gradient among saves. The linear combination of the expected-save coefficient and its interaction estimates the gradient among goals.

The confirmatory interpretation is the realised-outcome direction: saves are associated with greater goalkeeper-relative visibility and goals with greater shooter-relative visibility. A claim that model-implied surprise adds an incremental gradient requires a statistically distinguishable within-save or within-goal slope. The present data do not meet that condition.

### All-pair sensitivity

The 386-pair sensitivity model replaces the binary goal outcome with the share of the pair's shots ending in goals and includes shot count. It is an aggregation check, not the primary single-episode estimand. Forty-three of the 83 multi-shot pairs contain both goals and saves.

### Descriptive categories

A 0.50 predicted-save threshold labels shots as heroic saves, routine saves, understandable goals or unexpected goals. These classes provide transparent descriptive summaries. Direct heroic-versus-routine and unexpected-versus-understandable contrasts are reported, but category names do not assign fault and are not the principal inferential design.

### Exploratory headline audit

Google News-indexed English-language headline metadata are filtered to a 48-hour post-match window, explicit goalkeeper identification and match-consistent timing/opponent information. Duplicate, pre-match and ambiguous records are removed. Praise, blame and neutral dictionaries provide an exploratory valence audit. The retained sample is too small and selection-dependent to identify audience attitudes or validate public responsibility attribution.

## Historical audience-data feasibility audit

A stratified pilot evaluated whether comparable 2022 pre/post follower or audience counts could be recovered from public archives. The pre-specified inclusion gate was 70% coverage. The pilot failed that gate, so current follower counts and incomplete historical estimates are excluded from the manuscript outcomes.

## Error handling and auditability

Source retrievals, mappings, cross-fitting folds, sample counts and output hashes are recorded. Unmatched active participants trigger failure rather than silent deletion. Raw and processed files remain separate. The workflow uses archive-contained paths and is tested in a clean directory.

## Verification

1. Automated tests cover parsing, own-goal exclusion, mapping, cross-fitting, leakage, expected-save diagnostics, near-miss balance, primary single-shot restriction, outcome-gradient decomposition, all-pair shot-count adjustment, headline filtering and workflow outputs.
2. The goalkeeper pipeline regenerates every reported v1.4.0 goalkeeper result.
3. Clean-directory tests and pipeline reruns must pass.
4. Locked outputs from the release and clean replay must match byte for byte.
5. Every page of the final R15 DOCX files is rendered and visually inspected.
