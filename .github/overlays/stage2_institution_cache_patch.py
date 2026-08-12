# Runtime overlay: globally deduplicated, per-institution OpenAlex metadata cache.
import sqlite3
import time

INSTITUTION_METADATA_COLUMNS = [
    "institution_id",
    "institution_name",
    "institution_type",
]


def _institution_metadata_row(item: dict[str, Any]) -> tuple[str, str, str] | None:
    institution_id = normalize_openalex_id(item.get("id"))
    if not institution_id:
        return None
    return (
        institution_id,
        str(item.get("display_name") or ""),
        str(item.get("type") or ""),
    )


def _open_institution_metadata_cache(cache_dir: Path) -> sqlite3.Connection:
    cache_root = cache_dir / "institution_metadata"
    cache_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_root / "by_id.sqlite3", timeout=60)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS institutions (
            institution_id TEXT PRIMARY KEY,
            institution_name TEXT NOT NULL,
            institution_type TEXT NOT NULL,
            is_missing INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_state (
            cache_key TEXT PRIMARY KEY,
            cache_value TEXT NOT NULL
        )
        """
    )
    return connection


def _upsert_institution_rows(
    connection: sqlite3.Connection,
    rows: Iterable[tuple[str, str, str]],
) -> None:
    connection.executemany(
        """
        INSERT INTO institutions (
            institution_id, institution_name, institution_type, is_missing
        ) VALUES (?, ?, ?, 0)
        ON CONFLICT(institution_id) DO UPDATE SET
            institution_name=excluded.institution_name,
            institution_type=excluded.institution_type,
            is_missing=0
        """,
        list(rows),
    )


def _migrate_legacy_institution_metadata(
    connection: sqlite3.Connection,
    cache_dir: Path,
) -> None:
    migration_key = "legacy_batch_json_v1"
    migrated = connection.execute(
        "SELECT 1 FROM cache_state WHERE cache_key=?", (migration_key,)
    ).fetchone()
    if migrated:
        return

    cache_root = cache_dir / "institution_metadata"
    for index, cache_path in enumerate(sorted(cache_root.glob("batch_*.json")), start=1):
        payload = _read_json(cache_path)
        rows = [
            parsed
            for item in payload.get("results", [])
            if (parsed := _institution_metadata_row(item)) is not None
        ]
        _upsert_institution_rows(connection, rows)
        if index % 250 == 0:
            connection.commit()
    connection.execute(
        """
        INSERT INTO cache_state (cache_key, cache_value) VALUES (?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET cache_value=excluded.cache_value
        """,
        (migration_key, "complete"),
    )
    connection.commit()


def _cached_institution_rows(
    connection: sqlite3.Connection,
    institution_ids: list[str],
) -> dict[str, tuple[str, str, int]]:
    cached: dict[str, tuple[str, str, int]] = {}
    for index in range(0, len(institution_ids), 900):
        batch = institution_ids[index : index + 900]
        if not batch:
            continue
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT institution_id, institution_name, institution_type, is_missing "
            f"FROM institutions WHERE institution_id IN ({placeholders})"
        )
        for institution_id, name, institution_type, is_missing in connection.execute(query, batch):
            cached[str(institution_id)] = (str(name), str(institution_type), int(is_missing))
    return cached


def _is_not_found_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "(404)" in message or "not found" in message


def fetch_institution_metadata(
    client: OpenAlexClient,
    institution_ids: list[str],
    cache_dir: Path,
    *,
    batch_size: int = 100,
    singleton_delay_seconds: float = 0.02,
) -> pd.DataFrame:
    """Resolve institution metadata once per normalized ID with restartable caching."""
    if not 1 <= batch_size <= 100:
        raise ValueError("Institution metadata batch_size must be between 1 and 100")
    if singleton_delay_seconds < 0:
        raise ValueError("singleton_delay_seconds must be non-negative")

    normalized_ids = sorted(
        {
            normalized
            for value in institution_ids
            if (normalized := normalize_openalex_id(value))
        }
    )
    if not normalized_ids:
        return pd.DataFrame(columns=INSTITUTION_METADATA_COLUMNS)

    connection = _open_institution_metadata_cache(cache_dir)
    try:
        _migrate_legacy_institution_metadata(connection, cache_dir)
        cached = _cached_institution_rows(connection, normalized_ids)
        missing_ids = [institution_id for institution_id in normalized_ids if institution_id not in cached]

        sleeper = getattr(client, "sleep", time.sleep)
        for index in range(0, len(missing_ids), batch_size):
            batch = missing_ids[index : index + batch_size]
            payload = client.get_json(
                "/institutions",
                {
                    "filter": f"openalex_id:{'|'.join(batch)}",
                    "per_page": "100",
                    "select": "id,display_name,type",
                },
            )
            parsed_rows = [
                parsed
                for item in payload.get("results", [])
                if (parsed := _institution_metadata_row(item)) is not None
            ]
            _upsert_institution_rows(connection, parsed_rows)
            returned_ids = {row[0] for row in parsed_rows}

            for institution_id in sorted(set(batch) - returned_ids):
                if singleton_delay_seconds:
                    sleeper(singleton_delay_seconds)
                try:
                    singleton = client.get_json(
                        f"/institutions/{institution_id}",
                        {"select": "id,display_name,type"},
                    )
                except RuntimeError as error:
                    if not _is_not_found_error(error):
                        raise
                    connection.execute(
                        """
                        INSERT INTO institutions (
                            institution_id, institution_name, institution_type, is_missing
                        ) VALUES (?, '', '', 1)
                        ON CONFLICT(institution_id) DO UPDATE SET
                            institution_name='', institution_type='', is_missing=1
                        """,
                        (institution_id,),
                    )
                    continue
                parsed = _institution_metadata_row(singleton)
                if parsed is None:
                    parsed = (institution_id, "", "")
                _upsert_institution_rows(connection, [parsed])
            connection.commit()

        cached = _cached_institution_rows(connection, normalized_ids)
    finally:
        connection.close()

    rows = [
        {
            "institution_id": institution_id,
            "institution_name": cached[institution_id][0],
            "institution_type": cached[institution_id][1],
        }
        for institution_id in normalized_ids
        if institution_id in cached and cached[institution_id][2] == 0
    ]
    if not rows:
        return pd.DataFrame(columns=INSTITUTION_METADATA_COLUMNS)
    return pd.DataFrame(rows, columns=INSTITUTION_METADATA_COLUMNS)


def fetch_exact_field_counts(
    client: OpenAlexClient,
    units: pd.DataFrame,
    cache_dir: Path,
    *,
    top_n: int = 100,
) -> pd.DataFrame:
    """Fetch grouped field activity and resolve institution metadata globally once."""
    columns = [
        "subfield_id",
        "as_of_date",
        "company_id",
        "prior_subfield_publication_count",
        "query_from_date",
        "query_to_date",
    ]
    unit_cache_records: list[dict[str, Any]] = []
    institution_ids: set[str] = set()

    for unit in units.itertuples(index=False):
        start, end = exact_five_year_window(str(unit.publication_date))
        subfield_filter_id = _canonical_subfield_filter_id(unit.primary_subfield_id)
        cache_path = cache_dir / "field_groups_v2" / f"{unit.unit_key}.json"
        if cache_path.exists():
            groups = _read_json(cache_path)
        else:
            filters = ",".join(
                [
                    f"primary_topic.subfield.id:{subfield_filter_id}",
                    f"from_publication_date:{start}",
                    f"to_publication_date:{end}",
                    "authorships.institutions.type:company",
                ]
            )
            groups = list(
                client.iter_groups(
                    "/works",
                    {
                        "filter": filters,
                        "group_by": "authorships.institutions.id",
                    },
                )
            )
            _write_json_atomic(cache_path, groups)

        institution_ids.update(
            normalized
            for group in groups
            if group.get("key")
            if (normalized := normalize_openalex_id(group.get("key")))
        )
        unit_cache_records.append(
            {
                "subfield_id": unit.primary_subfield_id,
                "as_of_date": unit.publication_date,
                "query_from_date": start,
                "query_to_date": end,
                "cache_path": cache_path,
            }
        )

    metadata = fetch_institution_metadata(client, sorted(institution_ids), cache_dir)
    rows: list[dict[str, Any]] = []
    for record in unit_cache_records:
        groups = _read_json(record["cache_path"])
        selected = select_top_company_groups(groups, metadata, top_n=top_n)
        for item in selected.itertuples(index=False):
            rows.append(
                {
                    "subfield_id": record["subfield_id"],
                    "as_of_date": record["as_of_date"],
                    "company_id": item.company_id,
                    "prior_subfield_publication_count": int(item.prior_subfield_publication_count),
                    "query_from_date": record["query_from_date"],
                    "query_to_date": record["query_to_date"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)
