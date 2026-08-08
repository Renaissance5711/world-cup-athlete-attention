import pandas as pd

from src.stage2_openalex_extract import (
    _write_json_atomic,
    fetch_exact_field_counts,
    fetch_institution_metadata,
)


class NoNetworkClient:
    def get_json(self, *args, **kwargs):
        raise AssertionError("network should not be called")


class InstitutionMetadataClient:
    def __init__(self, omitted_from_lists=None):
        self.omitted_from_lists = set(omitted_from_lists or [])
        self.list_calls = []
        self.singleton_calls = []
        self.sleep_calls = []

    def get_json(self, path, params):
        if path == "/institutions":
            self.list_calls.append((path, dict(params)))
            ids = params["filter"].removeprefix("openalex_id:").split("|")
            return {
                "results": [
                    {
                        "id": f"https://openalex.org/{institution_id}",
                        "display_name": f"Institution {institution_id}",
                        "type": "company" if institution_id.endswith("1") else "education",
                    }
                    for institution_id in ids
                    if institution_id not in self.omitted_from_lists
                ]
            }
        if path.startswith("/institutions/"):
            institution_id = path.rsplit("/", 1)[-1]
            self.singleton_calls.append((path, dict(params)))
            return {
                "id": f"https://openalex.org/{institution_id}",
                "display_name": f"Institution {institution_id}",
                "type": "company" if institution_id.endswith("1") else "education",
            }
        raise AssertionError(f"unexpected path: {path}")

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)


class FieldGroupClient:
    def __init__(self, groups):
        self.groups = list(groups)
        self.group_calls = []
        self.metadata_calls = []

    def iter_groups(self, path, params):
        self.group_calls.append((path, dict(params)))
        yield from self.groups

    def get_json(self, path, params):
        self.metadata_calls.append((path, dict(params)))
        ids = params["filter"].removeprefix("openalex_id:").split("|")
        return {
            "results": [
                {
                    "id": f"https://openalex.org/{institution_id}",
                    "display_name": f"Institution {institution_id}",
                    "type": "company",
                }
                for institution_id in ids
            ]
        }


def test_institution_metadata_deduplicates_batches_of_100_and_reuses_cache(tmp_path):
    client = InstitutionMetadataClient()
    ids = [f"I{index}" for index in range(1, 202)] + ["I1", "I100"]

    first = fetch_institution_metadata(client, ids, tmp_path)
    second = fetch_institution_metadata(client, ["I200", "I1", "I50"], tmp_path)

    assert len(first) == 201
    assert set(second["institution_id"]) == {"I1", "I50", "I200"}
    assert len(client.list_calls) == 3
    assert all(
        len(call[1]["filter"].removeprefix("openalex_id:").split("|")) <= 100
        for call in client.list_calls
    )
    assert (tmp_path / "institution_metadata" / "by_id.sqlite3").exists()


def test_institution_metadata_migrates_legacy_batch_cache_without_network(tmp_path):
    legacy = tmp_path / "institution_metadata" / "batch_legacy.json"
    _write_json_atomic(
        legacy,
        {
            "results": [
                {
                    "id": "https://openalex.org/I9",
                    "display_name": "Legacy Company",
                    "type": "company",
                }
            ]
        },
    )

    result = fetch_institution_metadata(NoNetworkClient(), ["I9"], tmp_path)

    assert result.to_dict("records") == [
        {
            "institution_id": "I9",
            "institution_name": "Legacy Company",
            "institution_type": "company",
        }
    ]


def test_institution_metadata_uses_rate_limited_singleton_for_list_omissions(tmp_path):
    client = InstitutionMetadataClient(omitted_from_lists={"I2"})

    first = fetch_institution_metadata(client, ["I1", "I2"], tmp_path)
    second = fetch_institution_metadata(client, ["I2"], tmp_path)

    assert set(first["institution_id"]) == {"I1", "I2"}
    assert second.loc[0, "institution_id"] == "I2"
    assert len(client.list_calls) == 1
    assert client.singleton_calls == [("/institutions/I2", {"select": "id,display_name,type"})]
    assert client.sleep_calls == [0.02]


def test_field_counts_resolves_global_institution_metadata_once(tmp_path):
    units = pd.DataFrame(
        [
            {
                "primary_subfield_id": 1102.0,
                "publication_date": "2012-09-26",
                "unit_key": "unit_a",
            },
            {
                "primary_subfield_id": 1102.0,
                "publication_date": "2013-09-26",
                "unit_key": "unit_b",
            },
        ]
    )
    client = FieldGroupClient(
        [
            {"key": "https://openalex.org/I1", "count": 7},
            {"key": "https://openalex.org/I2", "count": 5},
        ]
    )

    result = fetch_exact_field_counts(client, units, tmp_path)

    assert len(result) == 4
    assert len(client.group_calls) == 2
    assert len(client.metadata_calls) == 1
    assert client.metadata_calls[0][1]["filter"] == "openalex_id:I1|I2"
