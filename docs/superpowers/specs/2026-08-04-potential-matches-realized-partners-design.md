# Potential Matches, Realized Partners: Analysis Design

Date: 2026-08-04  
Target outlet: IEEE Transactions on Engineering Management  
Status: Design approved at the topic level; empirical core awaits a corrected Stage 2 candidate and cognitive-fit run.

## 1. Core research problem

Data-driven partner-search methods can identify firms that appear technically or cognitively compatible with a scientific project. Yet a potential match does not automatically become a realized partner. The study asks:

> How do relational embeddedness and project–firm cognitive fit jointly determine whether a potential industry match becomes a realized partner, and does project commercial potential alter this conversion?

The paper does not treat prior relationships as a new theory. It extends the project-level open-innovation partner-search and university–industry matching dialogue by distinguishing **potential matching** from **partnership realization**.

## 2. Central theoretical contrast

Existing partner-search tools emphasize identification and evaluation of suitable firms. The proposed paper adds a realization stage.

- Cognitive fit indicates whether a firm is technically relevant to the project.
- Relational embeddedness may enable realization by supplying visibility, contactability, trust, and coordination experience.
- Relational embeddedness may also constrain realization by favoring familiar firms when unfamiliar firms have higher observed cognitive fit.
- COMPOT is a project-level boundary condition. High commercial potential may attract firms outside the existing network, or it may increase appropriation and coordination risks and strengthen reliance on trusted partners.

The main contribution is not “prior ties matter.” It is the distinction between:

1. **Potential-match quality**, and
2. **The probability that a potential match becomes a realized partnership.**

## 3. Required unit of analysis

Primary unit:

`scientific project i × candidate firm j`

Outcome:

`selected_ij = 1` when firm j is an observed industry partner on project i.

Projects may select more than one firm. Project-stratified conditional logit is interpreted as subset inclusion conditional on the number of selected firms in each project, not as a single-choice multinomial model.

## 4. Candidate-set redesign

### 4.1 Problem discovered in the current V5 code

The current candidate union includes every firm that had any university-level relationship during the previous five years. This produces implausibly large relationship-generated candidate sets:

- Across all 1,881 projects, the median author-plus-any-university set is 220 firms.
- Among the 1,087 projects with university information, the median is approximately 788 firms.
- The maximum is 3,881 firms.

This should not be interpreted as a managerial consideration set.

### 4.2 Main natural discovery set

For each project, construct the natural set before observing actual selection:

1. Top 100 firms active in the project’s primary subfield during the exact five-year prepublication window.
2. All firms with an author-level prior relationship during that window.
3. Strong university-network firms, defined in the main specification as at least five distinct university–firm joint works during that window.

Any university relationship remains measurable for every candidate, but unrestricted university relationships do not generate candidates in the main specification.

### 4.3 Sensitivity candidate sets

- Top 50, 100, and 200 field-active firms.
- University threshold of at least two works.
- At least two university works in the previous two years.
- No university-generated candidates.
- Unrestricted university candidates only as an explicit upper-bound robustness test.

### 4.4 Natural recall and selected-firm injection

The workflow must separate discovery performance from estimation completeness.

1. Calculate natural project recall and firm recall before adding actual firms.
2. Create an estimation set by appending all actual selected firms not already captured.
3. Flag appended rows with `forced_selected_candidate = 1`.
4. Fetch cognitive history for appended selected firms as well.
5. Report results both including and excluding forced selected rows/projects.

This prevents the analysis from silently dropping the hardest realization cases.

## 5. Core constructs

### 5.1 Cognitive fit

Main measure:

TF–IDF cosine similarity between the project title/abstract and the candidate firm’s strictly prepublication scientific-document profile in the same subfield and five-year window.

Required variants:

- Raw cosine.
- Within-project percentile rank.
- Top-decile fit indicator.
- Publication-count-weighted or shrinkage-adjusted fit.
- Minimum evidence threshold for firm profiles.
- Topic-distribution similarity as a robustness measure if available.

Use the term “observed cognitive fit,” not “true partner quality.”

### 5.2 Relational embeddedness

Four-state categorical measure:

- Neither author nor university prior relationship.
- University-only.
- Author-only.
- Both.

Additional measures:

- Author relationship strength.
- University relationship strength.
- Relationship recency.
- Immediate prior-project partner.
- Multiplexity: both author and university routes.

Missing university information must not be coded as no university relationship.

### 5.3 Realization gap

Use multiple outcomes rather than one fragile metric.

1. `fit_shortfall = max natural-candidate fit – mean selected-firm fit`
2. Selected firm’s within-project fit percentile.
3. Indicator that no selected firm is in the project’s top fit decile.
4. Technical-model rank of the realized partner.
5. Difference between technical-only and combined-model rank of the realized partner.

Avoid calling these efficiency losses. They are observed matching deviations.

### 5.4 Conflict projects

A project is theoretically informative when relationship and cognitive fit point toward different firms.

Primary conflict condition:

- At least one unselected firm has no author relationship and materially higher fit than a selected embedded firm.

Report:

- Frequency of conflict projects.
- Which alternative wins.
- Fit difference and relationship difference.
- Variation by COMPOT.

## 6. Empirical sequence

### Analysis 1: Candidate-set validity

Report for every candidate definition:

- Mean, median, p90, p95, and maximum size.
- Natural project recall.
- Natural firm-instance recall.
- Recall by year, field, COMPOT quartile, and university-data coverage.
- Share of selected firms requiring forced injection.

The main candidate definition must be chosen before examining model significance.

### Analysis 2: Predictive realization benchmark

Compare three models.

**Technical model**
- Cognitive fit.
- Subfield activity/capability.
- Scientific-profile evidence volume.

**Relational model**
- Four-state relationship category.
- Strength and recency.
- Immediate prior-project partner.

**Combined model**
- Technical and relational variables.
- Fit × relationship interactions.
- COMPOT interactions.

Use temporal out-of-sample evaluation, preferably rolling or blocked by publication year. Report:

- Mean reciprocal rank.
- Recall@1, Recall@5, Recall@10.
- Mean selected-partner rank.
- Project-level average precision for multi-firm projects.
- Log loss or ranking loss where estimable.

The central predictive result is not merely whether the combined model performs better. It is whether technical-model errors are systematically concentrated among high-fit firms without relational routes.

### Analysis 3: Project-stratified choice models

Model sequence:

- M1: capability controls.
- M2: cognitive fit.
- M3: four-state relational embeddedness.
- M4: fit and relationships jointly.
- M5: fit × relationship state.
- M6: relationship state × COMPOT.
- M7: fit × COMPOT.
- M8: selected three-way interactions only when theoretically justified and statistically supported.

Because COMPOT is constant within a project, its main effect is absorbed by project strata. Only interactions with candidate-varying variables are identified.

### Analysis 4: Conflict-set analysis

Restrict to projects where a selected embedded firm competes against a higher-fit unembedded candidate.

Estimate:

- Probability that the embedded firm wins.
- How large a fit advantage an unembedded firm needs to offset relationship disadvantage.
- Whether university-only routes narrow the disadvantage.
- Whether COMPOT changes the trade-off.

This is the strongest test of the paper’s theoretical contrast.

### Analysis 5: Realization-gap outcomes

At project level, estimate fit shortfall and selected-fit percentile as functions of:

- Selected-firm relational state.
- Author relationship strength.
- Immediate prior-project repetition.
- COMPOT.
- Relationship × COMPOT.

Include year and field fixed effects and cluster by author. Interpret associationally.

### Analysis 6: Existing relationship findings as mechanism evidence

The completed relation-only analyses should move to a secondary role:

- Strong and recent author relationships predict selection among known firms.
- The immediately previous project partner has an additional selection advantage.
- These findings explain how realized selections may become self-reinforcing.

They do not by themselves establish search restriction or matching deviation.

## 7. Result-contingent theoretical interpretations

### Pattern A: Fit and relationships are complements

High cognitive fit has a stronger effect when a relational route exists.

Contribution:

Relationships operate as realization infrastructure that allows technical compatibility to be recognized and implemented.

### Pattern B: Fit substitutes for relationships

High fit has its largest effect for unembedded firms.

Contribution:

Strong technical signals can help firms overcome relational barriers.

### Pattern C: Embedded low-fit firms defeat higher-fit outsiders

Contribution:

Relational embeddedness generates a realization constraint and observed cognitive-fit deviation.

### Pattern D: University-only routes matter only for high-fit firms

Contribution:

Organization-level networks do not indiscriminately favor all connected firms; they activate technically relevant latent matches.

### Pattern E: COMPOT weakens relational advantage

Contribution:

Commercial attraction expands realization beyond existing networks.

### Pattern F: COMPOT strengthens relational advantage

Contribution:

Higher-value projects increase assurance needs and reliance on trusted partners.

### Pattern G: No fit–relationship interaction

The paper becomes weaker. It can still document independent technical and relational dimensions, but should not claim a realization mechanism unless predictive-error or conflict analyses reveal systematic patterns.

## 8. Go/no-go criteria

Proceed with the main paper when all of the following are satisfied:

- Natural project recall is at least 80%.
- Natural firm-instance recall is at least 75%.
- At least 500 projects have captured selected firms and at least two credible alternatives.
- Cognitive fit is nonmissing or substantively measurable for at least 80% of selected firms and natural candidates.
- At least 10% of projects contain a meaningful relation–fit conflict.
- Candidate-set conclusions are stable across top-50/100/200 field definitions.
- The combined model improves partner ranking or reveals a systematic technical-model error pattern.

Reconsider the paper when:

- Candidate recall is low and highly selective.
- Cognitive profiles are missing mainly for unembedded firms.
- Conflict projects are rare.
- Results disappear when forced selected firms or large universities are handled correctly.

## 9. Claims that are not permitted

Do not claim:

- Causal effects of relationships.
- That the highest cosine firm is objectively the best partner.
- That matching deviation equals economic inefficiency.
- That university relationships represent active introductions.
- That the candidate set is the manager’s directly observed consideration set.
- That COMPOT has a main effect in project-stratified models.

## 10. Required workflow changes

Before another full Stage 2 run:

1. Replace unrestricted university candidate generation with the prespecified strong-university rule.
2. Preserve unrestricted university-tie indicators as candidate attributes.
3. Add natural-candidate and forced-selected flags.
4. Calculate natural recall before selected-firm injection.
5. Fetch cognitive history for all estimation candidates, including forced selected firms.
6. Add realization-gap and conflict-set outputs.
7. Add temporal predictive comparison for technical, relational, and combined models.
8. Add tests for time leakage, candidate recall, selected-firm injection, and university-missingness handling.

## 11. Intended contribution statement

The study advances project-level open-innovation partner search by showing that identifying a technically compatible firm and realizing a partnership are distinct management problems. It theorizes relational embeddedness as a mechanism that can either enable the conversion of cognitive fit into collaboration or constrain realization in favor of familiar firms. The empirical design identifies when these two selection logics reinforce one another and when they conflict.
