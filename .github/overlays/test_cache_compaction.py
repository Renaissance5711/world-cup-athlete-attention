import gzip
import json

import pandas as pd

from src.stage2_openalex_extract import (
    _read_json,
    _write_json_atomic,
    fetch_candidate_firm_text_history,
    fetch_entity_year_history,
)


class NoNetworkClient:
    def iter_results(self, *args, **kwargs):
        raise AssertionError("network should not be called")


class StaticClient:
    def __init__(self, works):
        self.works = works
        self.calls = 0

    def iter_results(self, *args, **kwargs):
        self.calls += 1
        yield from self.works


def company_work(work_id="W1", company_id="I9", title="A useful title"):
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "publication_date": "2020-01-02",
        "abstract_inverted_index": {"useful": [0], "research": [1]},
        "authorships": [
            {
                "institutions": [
                    {
                        "id": f"https://openalex.org/{company_id}",
                        "type": "company",
                        "display_name": "Repeated company metadata " * 100,
                    }
                ]
            }
        ],
    }


def test_atomic_gzip_json_round_trip(tmp_path):
    path = tmp_path / "cache" / "rows.json.gz"
    payload = [{"a": 1}, {"text": "数据"}]
    _write_json_atomic(path, payload)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    assert _read_json(path) == payload
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == payload


def test_entity_history_migrates_legacy_raw_json(tmp_path):
    units = pd.DataFrame(
        [{
            "author_id": "A1",
            "cache_key": "A1_2020",
            "query_from_date": "2015-01-01",
            "query_to_date": "2020-12-31",
        }]
    )
    legacy = tmp_path / "author_id_history" / "A1_2020.json"
    _write_json_atomic(legacy, [company_work()])
    legacy_size = legacy.stat().st_size

    frame = fetch_entity_year_history(
        NoNetworkClient(),
        units,
        tmp_path,
        entity_column="author_id",
        openalex_filter_field="authorships.author.id",
    )

    compact = tmp_path / "author_id_history" / "A1_2020.evidence.json.gz"
    assert not legacy.exists()
    assert compact.exists()
    assert compact.stat().st_size < legacy_size
    assert frame.to_dict("records") == [{
        "author_id": "A1",
        "company_id": "I9",
        "evidence_date": "2020-01-02",
        "prior_work_id": "W1",
    }]


def test_entity_history_new_query_writes_only_compact_evidence(tmp_path):
    units = pd.DataFrame(
        [{
            "university_id": "I1",
            "cache_key": "I1_2020",
            "query_from_date": "2015-01-01",
            "query_to_date": "2020-12-31",
        }]
    )
    client = StaticClient([company_work()])
    fetch_entity_year_history(
        client,
        units,
        tmp_path,
        entity_column="university_id",
        openalex_filter_field="authorships.institutions.id",
    )
    assert client.calls == 1
    root = tmp_path / "university_id_history"
    assert (root / "I1_2020.evidence.json.gz").exists()
    assert not (root / "I1_2020.json").exists()


def test_cognitive_history_migrates_raw_works_to_compact_rows(tmp_path):
    projects = pd.DataFrame(
        [{"work_id": "P1", "publication_date": "2021-01-10", "primary_subfield_id": "S1"}]
    )
    candidates = pd.DataFrame([{"work_id": "P1", "company_id": "I9"}])
    signature = __import__("hashlib").sha256(b"I9").hexdigest()[:12]
    root = tmp_path / "cognitive_history"
    legacy = root / f"P1_000_{signature}.json"
    _write_json_atomic(legacy, [company_work()])

    result = fetch_candidate_firm_text_history(
        NoNetworkClient(), projects, candidates, tmp_path, batch_size=20
    )

    compact = root / f"P1_000_{signature}.rows.json.gz"
    assert compact.exists()
    assert not legacy.exists()
    assert len(result) == 1
    assert result.loc[0, "focal_work_id"] == "P1"
    assert result.loc[0, "company_id"] == "I9"
    assert "useful research" in result.loc[0, "document_text"]
