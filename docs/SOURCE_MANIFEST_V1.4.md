# Source manifest for v1.4.0 extension

## StatsBomb Open Data

- Source: official StatsBomb/Hudl Open Data GitHub repository.
- Competition: FIFA World Cup, competition ID 43, season ID 106.
- Files used: one match metadata file, 64 event JSON files and 64 lineup JSON files.
- Retrieval audit: every event and lineup request returned HTTP 200; batch hashes are recorded in `data/raw/statsbomb/manifest.json`.
- Near-miss use: shot quality, volume, context, match state and participation measures.
- Goalkeeper use: exact shooter-goalkeeper pairing; realised shot outcome; expected goals; shot origin and available trajectory/end coordinates; technique; body part; pressure, open-goal, one-on-one and deflection indicators; goalkeeper outcome; game minute; score state and tournament context.
- Derived expected-save predictions: eight-fold cross-predictions grouped by match. A shot is never predicted by a model trained on its own match.
- Attribution: the manuscript identifies StatsBomb as the data source. Users redistributing or publishing analyses based on these files must review and follow the current StatsBomb Open Data terms.

## Google News-indexed headline metadata

- Source interface: Google News RSS search results, US English edition.
- Retrieval scope: historical searches covering 19 November through 21 December 2022 for the 40 goalkeepers observed in the StatsBomb on-target-shot panel.
- Raw archived fields: goalkeeper identifier, query name, headline title, indexed link, publication date and source name.
- Raw indexed records: 1,714.
- Strict retained records: 59 headlines covering 31 goalkeeper-match observations and 26 matches.
- Retention rules: after the match and within 48 hours; explicit goalkeeper identity/context; conservative opponent and temporal checks; duplicate removal; English-language indexed results.
- Use in study: exploratory audit of praise, blame and neutral framing following goalkeeper match performances.
- Boundary: search results are ranked and may be incomplete. The archive redistributes metadata and links only, not article body text. Results are not a census of coverage, a direct measure of audience attitudes or a validation of public responsibility attribution.

## Existing sources

Fjelstul World Cup Database and Wikimedia Analytics API inputs are documented in `base_archive/docs/SOURCE_MANIFEST.md`.
