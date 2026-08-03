# OpenAlex Institution Metadata Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make institution metadata resolution globally deduplicated, restartable by institution ID, and substantially cheaper in paid OpenAlex credits.

**Architecture:** Store normalized institution records in one SQLite database keyed by `institution_id`. Migrate legacy batch JSON into that database, resolve only globally missing IDs in 100-ID list calls, and use rate-limited free singleton calls only for list omissions. `fetch_exact_field_counts` performs a disk-backed two-pass scan so metadata is resolved once for the complete field manifest.

**Tech Stack:** Python 3.11, pandas, sqlite3, pytest, GitHub Actions cache.

## Global Constraints

- Preserve the existing `fetch_institution_metadata` and `fetch_exact_field_counts` return schemas.
- Preserve all previously saved OpenAlex evidence and group caches.
- Do not expose API keys in source, tests, logs, or artifacts.
- OpenAlex OR filters contain at most 100 values.
- Singleton fallback runs sequentially with a default 0.02-second delay.

---

### Task 1: Institution-ID cache behavior

**Files:**
- Modify: `.github/overlays/test_cache_compaction.py`
- Modify: `.github/overlays/stage2_openalex_extract.py`

**Interfaces:**
- Consumes: `OpenAlexClient.get_json(path, params)` and normalized institution IDs.
- Produces: `fetch_institution_metadata(client, institution_ids, cache_dir, *, batch_size=100, singleton_delay_seconds=0.02) -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests**

Add tests that request duplicate/reordered IDs, verify one 100-ID list request, verify an overlapping second call performs no network request, verify migration from `institution_metadata/batch_*.json`, and verify omitted list IDs use `/institutions/{id}` once then remain cached.

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest -q tests_v5/test_cache_compaction.py`

Expected: failures because the current cache is batch-signature based, defaults to 50, and has no singleton fallback.

- [ ] **Step 3: Implement the SQLite cache**

Create `institution_metadata/by_id.sqlite3` with:

```sql
CREATE TABLE IF NOT EXISTS institutions (
  institution_id TEXT PRIMARY KEY,
  institution_name TEXT NOT NULL,
  institution_type TEXT NOT NULL,
  is_missing INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cache_state (
  cache_key TEXT PRIMARY KEY,
  cache_value TEXT NOT NULL
);
```

Migrate legacy `batch_*.json` once, query cached IDs in chunks, fetch globally missing IDs in batches of 100, singleton-fetch batch omissions, negative-cache 404s, and return rows in normalized ID order.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `pytest -q tests_v5/test_cache_compaction.py`

Expected: all focused tests pass.

### Task 2: Global metadata deduplication in field extraction

**Files:**
- Modify: `.github/overlays/test_cache_compaction.py`
- Modify: `.github/overlays/stage2_openalex_extract.py`

**Interfaces:**
- Consumes: cached/fetched `field_groups_v2/<unit_key>.json` and Task 1 metadata cache.
- Produces: unchanged six-column field-count DataFrame.

- [ ] **Step 1: Write a failing two-unit test**

Create two field units whose groups overlap on an institution ID. Assert that metadata resolution receives/fetches the globally deduplicated ID set once, not once per unit, while both output units are retained.

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest -q tests_v5/test_cache_compaction.py -k global`

Expected: failure because the current implementation calls `fetch_institution_metadata` inside the per-unit loop.

- [ ] **Step 3: Implement a two-pass field scan**

First pass: fetch/load each group file, collect normalized IDs globally, and retain only unit metadata plus cache paths. Resolve metadata once. Second pass: reread each group file, select companies, and append the existing output rows.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
pytest -q tests_v5/test_cache_compaction.py
pytest -q tests_v5
```

Expected: focused tests and all Stage 2 tests pass.

### Task 3: Workflow checkpoint and verification

**Files:**
- Modify: `.github/workflows/tem-stage2-v5.yml`

**Interfaces:**
- Consumes: latest cache `tem-stage2-v5-openalex-30780601294-1`.
- Produces: a new run that validates the optimization and restores the newest checkpoint.

- [ ] **Step 1: Update the preferred restore key**

Place `tem-stage2-v5-openalex-30780601294-1` first in `restore-keys`.

- [ ] **Step 2: Commit implementation files**

Commit tests and implementation together only after local RED/GREEN evidence is recorded.

- [ ] **Step 3: Verify the triggered workflow**

Confirm 21 existing tests plus new tests pass, input validation reports 1,881 projects, and cache restore logs show `tem-stage2-v5-openalex-30780601294-1`.

- [ ] **Step 4: Inspect the next completed run**

Verify the new SQLite cache exists in the checkpoint, paid metadata calls are sharply reduced, Stage 2 exit code is authoritative, and any remaining failure preserves the newest cache key.
