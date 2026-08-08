"""OpenAlex extraction primitives for the full Stage 2 risk set."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.openalex_stage2 import OpenAlexClient
from src.stage2_risk_set import normalize_openalex_id, parse_project_detail


def exact_five_year_window(publication_date: str) -> tuple[str, str]:
    """Return exact five-year start and inclusive day-before-publication end."""
    publication = pd.Timestamp(publication_date).tz_localize(None)
    start = publication - pd.DateOffset(years=5)
    end = publication - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def extract_company_evidence_from_works(
    works: Iterable[dict[str, Any]],
    *,
    entity_id: str,
    entity_column: str,
) -> list[dict[str, str]]:
    """Convert work records into dated entity-company collaboration evidence."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for work in works:
        work_id = normalize_openalex_id(work.get("id"))
        evidence_date = str(work.get("publication_date") or "")
        if not work_id or not evidence_date:
            continue
        companies: set[str] = set()
        for authorship in work.get("authorships", []) or []:
            for institution in authorship.get("institutions", []) or []:
                if institution.get("type") == "company":
                    institution_id = normalize_openalex_id(institution.get("id"))
                    if institution_id:
                        companies.add(institution_id)
        for company_id in sorted(companies):
            key = (company_id, evidence_date, work_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    entity_column: str(entity_id),
                    "company_id": company_id,
                    "evidence_date": evidence_date,
                    "prior_work_id": work_id,
                }
            )
    return rows


def select_top_company_groups(
    groups: Iterable[dict[str, Any]],
    institution_metadata: pd.DataFrame,
    *,
    top_n: int = 100,
) -> pd.DataFrame:
    """Filter grouped institutions to companies, then rank locally by count."""
    required = {"institution_id", "institution_type"}
    missing = sorted(required - set(institution_metadata.columns))
    if missing:
        raise ValueError(f"Institution metadata is missing: {', '.join(missing)}")
    company_ids = set(
        institution_metadata.loc[
            institution_metadata["institution_type"].eq("company"), "institution_id"
        ].astype(str)
    )
    rows = [
        {
            "company_id": normalize_openalex_id(group.get("key")),
            "prior_subfield_publication_count": int(group.get("count", 0)),
        }
        for group in groups
        if normalize_openalex_id(group.get("key")) in company_ids
    ]
    if not rows:
        return pd.DataFrame(columns=["company_id", "prior_subfield_publication_count"])
    return (
        pd.DataFrame(rows)
        .sort_values(["prior_subfield_publication_count", "company_id"], ascending=[False, True])
        .drop_duplicates("company_id")
        .head(top_n)
        .reset_index(drop=True)
    )


def _read_json(path: Path) -> Any:
    """Read plain or gzip-compressed JSON cache content."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write compact JSON, using gzip when the name ends in .gz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    else:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_project_details(
    client: OpenAlexClient,
    projects: pd.DataFrame,
    cache_dir: Path,
    *,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Fetch works in restartable batches and parse Stage 2 project details."""
    work_to_authors = {
        str(row.work_id): {value for value in str(row.focal_author_ids).split("|") if value}
        for row in projects.itertuples(index=False)
    }
    work_ids = sorted(work_to_authors)
    rows: list[dict[str, Any]] = []
    for batch_index in range(0, len(work_ids), batch_size):
        batch = work_ids[batch_index : batch_index + batch_size]
        cache_path = cache_dir / "project_details" / f"batch_{batch_index // batch_size:04d}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = client.get_json(
                "/works",
                {
                    "filter": f"openalex_id:{'|'.join(batch)}",
                    "per_page": "100",
                    "select": "id,title,abstract_inverted_index,publication_date,authorships",
                },
            )
            _write_json_atomic(cache_path, payload)
        fetched = {normalize_openalex_id(work.get("id")): work for work in payload.get("results", [])}
        missing = sorted(set(batch) - set(fetched))
        if missing:
            raise ValueError(f"OpenAlex project-detail batch omitted works: {', '.join(missing[:5])}")
        for work_id in batch:
            rows.append(parse_project_detail(fetched[work_id], work_to_authors[work_id]))
    return pd.DataFrame(rows)


def fetch_entity_year_history(
    client: OpenAlexClient,
    units: pd.DataFrame,
    cache_dir: Path,
    *,
    entity_column: str,
    openalex_filter_field: str,
) -> pd.DataFrame:
    """Fetch annual history and retain only compact entity-company evidence."""
    output: list[dict[str, str]] = []
    for row in units.itertuples(index=False):
        entity_id = str(getattr(row, entity_column))
        cache_key = str(row.cache_key)
        cache_root = cache_dir / f"{entity_column}_history"
        compact_path = cache_root / f"{cache_key}.evidence.json.gz"
        legacy_path = cache_root / f"{cache_key}.json"
        if compact_path.exists():
            evidence_rows = _read_json(compact_path)
        else:
            if legacy_path.exists():
                works = _read_json(legacy_path)
            else:
                filters = ",".join(
                    [
                        f"{openalex_filter_field}:{entity_id}",
                        f"from_publication_date:{row.query_from_date}",
                        f"to_publication_date:{row.query_to_date}",
                        "authorships.institutions.type:company",
                    ]
                )
                works = list(
                    client.iter_results(
                        "/works",
                        {
                            "filter": filters,
                            "select": "id,publication_date,authorships",
                        },
                    )
                )
            evidence_rows = extract_company_evidence_from_works(
                works,
                entity_id=entity_id,
                entity_column=entity_column,
            )
            _write_json_atomic(compact_path, evidence_rows)
            if legacy_path.exists():
                legacy_path.unlink()
        output.extend(evidence_rows)
    columns = [entity_column, "company_id", "evidence_date", "prior_work_id"]
    if not output:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(output).drop_duplicates(columns).reset_index(drop=True)


def fetch_institution_metadata(
    client: OpenAlexClient,
    institution_ids: list[str],
    cache_dir: Path,
    *,
    batch_size: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for index in range(0, len(institution_ids), batch_size):
        batch = sorted(institution_ids[index : index + batch_size])
        signature = hashlib.sha256("|".join(batch).encode()).hexdigest()[:16]
        cache_path = cache_dir / "institution_metadata" / f"batch_{signature}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = client.get_json(
                "/institutions",
                {
                    "filter": f"openalex_id:{'|'.join(batch)}",
                    "per_page": "100",
                    "select": "id,display_name,type",
                },
            )
            _write_json_atomic(cache_path, payload)
        for item in payload.get("results", []):
            rows.append(
                {
                    "institution_id": normalize_openalex_id(item.get("id")),
                    "institution_name": str(item.get("display_name") or ""),
                    "institution_type": str(item.get("type") or ""),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["institution_id", "institution_name", "institution_type"])
    return pd.DataFrame(rows).drop_duplicates("institution_id")


def _canonical_subfield_filter_id(value: Any) -> str:
    """Return the integer OpenAlex subfield ID used by API filters."""
    if value is None or pd.isna(value):
        raise ValueError("Missing primary_subfield_id")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    if not text:
        raise ValueError("Missing primary_subfield_id")
    return text


def fetch_exact_field_counts(
    client: OpenAlexClient,
    units: pd.DataFrame,
    cache_dir: Path,
    *,
    top_n: int = 100,
) -> pd.DataFrame:
    """Fetch exact-date grouped field activity and retain the top company groups."""
    columns = [
        "subfield_id",
        "as_of_date",
        "company_id",
        "prior_subfield_publication_count",
        "query_from_date",
        "query_to_date",
    ]
    rows: list[dict[str, Any]] = []
    for unit in units.itertuples(index=False):
        start, end = exact_five_year_window(str(unit.publication_date))
        subfield_filter_id = _canonical_subfield_filter_id(unit.primary_subfield_id)
        cache_path = cache_dir / "field_groups_v2" / f"{unit.unit_key}.json"
        if cache_path.exists():
            groups = json.loads(cache_path.read_text(encoding="utf-8"))
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
        institution_ids = sorted(
            {normalize_openalex_id(group.get("key")) for group in groups if group.get("key")}
        )
        metadata = fetch_institution_metadata(client, institution_ids, cache_dir)
        selected = select_top_company_groups(groups, metadata, top_n=top_n)
        for item in selected.itertuples(index=False):
            rows.append(
                {
                    "subfield_id": unit.primary_subfield_id,
                    "as_of_date": unit.publication_date,
                    "company_id": item.company_id,
                    "prior_subfield_publication_count": int(item.prior_subfield_publication_count),
                    "query_from_date": start,
                    "query_to_date": end,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def fetch_candidate_firm_text_history(
    client: OpenAlexClient,
    projects: pd.DataFrame,
    candidate_long: pd.DataFrame,
    cache_dir: Path,
    *,
    batch_size: int = 20,
) -> pd.DataFrame:
    """Fetch exact-window title/abstract histories for candidate companies by project."""
    from src.cognitive_fit import work_document_text
    required_projects={"work_id","publication_date","primary_subfield_id"}
    required_candidates={"work_id","company_id"}
    if required_projects-set(projects.columns):
        raise ValueError("projects missing cognitive-history columns")
    if required_candidates-set(candidate_long.columns):
        raise ValueError("candidate table missing cognitive-history columns")
    pmap=projects.drop_duplicates("work_id").set_index("work_id")
    output=[]
    for work_id, group in candidate_long.groupby("work_id", observed=True):
        project=pmap.loc[work_id]
        start,end=exact_five_year_window(str(project.publication_date))
        companies=sorted(group["company_id"].astype(str).unique())
        for batch_index in range(0,len(companies),batch_size):
            batch=companies[batch_index:batch_index+batch_size]
            signature=hashlib.sha256("|".join(batch).encode()).hexdigest()[:12]
            cache_root=cache_dir/"cognitive_history"
            cache_stem=f"{work_id}_{batch_index//batch_size:03d}_{signature}"
            compact_path=cache_root/f"{cache_stem}.rows.json.gz"
            legacy_path=cache_root/f"{cache_stem}.json"
            if compact_path.exists():
                batch_rows=_read_json(compact_path)
            else:
                if legacy_path.exists():
                    works=_read_json(legacy_path)
                else:
                    filters=",".join([
                        f"authorships.institutions.id:{'|'.join(batch)}",
                        f"primary_topic.subfield.id:{_canonical_subfield_filter_id(project.primary_subfield_id)}",
                        f"from_publication_date:{start}",
                        f"to_publication_date:{end}",
                    ])
                    works=list(client.iter_results("/works",{
                        "filter":filters,
                        "select":"id,title,abstract_inverted_index,publication_date,authorships",
                    }))
                batch_set=set(batch)
                batch_rows=[]
                for work in works:
                    document=work_document_text(work)
                    if not document:
                        continue
                    observed=set()
                    for authorship in work.get("authorships",[]) or []:
                        for institution in authorship.get("institutions",[]) or []:
                            iid=normalize_openalex_id(institution.get("id"))
                            if iid in batch_set:
                                observed.add(iid)
                    for company_id in sorted(observed):
                        batch_rows.append({
                            "focal_work_id":str(work_id),
                            "work_id":normalize_openalex_id(work.get("id")),
                            "company_id":company_id,
                            "publication_date":str(work.get("publication_date") or ""),
                            "document_text":document,
                        })
                _write_json_atomic(compact_path,batch_rows)
                if legacy_path.exists():
                    legacy_path.unlink()
            output.extend(batch_rows)
    columns=["focal_work_id","work_id","company_id","publication_date","document_text"]
    if not output:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(output).drop_duplicates(["focal_work_id","work_id","company_id"]).reset_index(drop=True)
