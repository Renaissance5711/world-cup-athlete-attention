# Review and reproducibility note

The public repository intentionally excludes the anonymous manuscript, submission tables and submission figures. The code and compact outputs added in this branch support audit of the new full-ranking analysis without exposing the review package.

The primary decomposition directly compares the two observed athlete-level ranking variables. The reconstructed ratio score is used only as a denominator accounting diagnostic. Because the primary proportional score averages appearance-level log lifts while the additive score averages winsorised additional pageviews, the diagnostic is not expected to reproduce every observed ordering. The residual is reported rather than suppressed.

All uncertainty calculations reweight matches, not pair rows. Pair rows share athletes and are not treated as independent observations.
