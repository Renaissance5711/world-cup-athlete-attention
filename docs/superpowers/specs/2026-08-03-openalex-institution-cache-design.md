# OpenAlex Institution Metadata Cache Optimization

## Goal

Reduce paid OpenAlex list/filter calls during Stage 2 subfield-company extraction while preserving exact output semantics and restartability.

## Root cause

The current metadata cache key is a hash of each 50-ID batch. The same institution can appear in many subfield/date units and in different batch combinations, so its metadata is repeatedly fetched and charged. The field-count loop also resolves metadata independently for every unit rather than deduplicating IDs across the complete manifest.

## Design

1. Replace combination-based metadata caching with a SQLite cache keyed by normalized `institution_id`.
2. Migrate all existing `institution_metadata/batch_*.json` records into the SQLite cache once, so previously paid results are retained.
3. Normalize and globally deduplicate all institution IDs across the 1,798 subfield/date group files before requesting missing metadata.
4. Fetch missing IDs using OpenAlex OR filters in batches of 100 with `per_page=100` and `select=id,display_name,type`.
5. For IDs omitted by a batch response, use the free singleton endpoint `/institutions/{id}` sequentially, with a 0.02-second delay between calls to stay below 100 requests per second.
6. Negative-cache singleton 404s so invalid or removed IDs are not retried in later runs.
7. Read group files twice: the first pass collects globally unique institution IDs; the second pass builds each unit's top-company rows. This avoids holding all group payloads in memory.

## Data flow

`field_groups_v2/*.json` → collect unique normalized institution IDs → migrate/read SQLite metadata cache → fetch only globally missing IDs → singleton fallback for omitted IDs → reread each group file → select company groups → write the existing six-column field-count output.

## Compatibility

The public function signatures and output columns remain compatible. Existing legacy batch cache files remain untouched after migration. The SQLite file is stored under `institution_metadata/by_id.sqlite3` and is included in the existing GitHub Actions cache path.

## Error handling

Paid batch failures and non-404 singleton failures propagate. Singleton 404s are stored as missing records. Cache writes use SQLite transactions. Empty results retain the established six-column schema.

## Tests

Regression tests must prove:

- duplicate and reordered IDs are fetched once globally;
- default batch size is 100;
- a second overlapping call uses the per-ID cache without network calls;
- legacy batch JSON is migrated without network calls;
- IDs omitted by a list response use singleton fallback and are cached;
- field extraction performs one global metadata-resolution pass rather than one pass per unit;
- existing float subfield normalization and empty-schema behavior remain intact.
