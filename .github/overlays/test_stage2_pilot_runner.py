import pandas as pd
from run_stage2_realization_pilot import (
    add_deterministic_field_rank,
    required_output_filenames,
)


def test_runner_declares_all_gate_outputs():
    names = set(required_output_filenames())
    assert {
        "pilot_gate_decision_top50.json",
        "pilot_gate_decision_top100.json",
        "pilot_summary.json",
    }.issubset(names)


def test_field_rank_is_deterministic_with_company_tiebreak():
    frame = pd.DataFrame({
        "subfield_id": [1, 1, 1],
        "as_of_date": ["2020-01-01"] * 3,
        "company_id": ["C2", "C1", "C3"],
        "prior_subfield_publication_count": [10, 10, 5],
        "query_from_date": ["2015-01-01"] * 3,
        "query_to_date": ["2019-12-31"] * 3,
    })
    out = add_deterministic_field_rank(frame)
    assert out.sort_values("field_rank")["company_id"].tolist() == ["C1", "C2", "C3"]
    assert out.sort_values("field_rank")["field_rank"].tolist() == [1, 2, 3]


def test_singleton_project_details_use_free_per_work_requests(tmp_path):
    from run_stage2_realization_pilot_v2 import fetch_project_details_singletons

    class FakeClient:
        def __init__(self):
            self.calls = []

        def get_json(self, path, params):
            self.calls.append((path, params))
            return {
                "id": "https://openalex.org/W1",
                "title": "Pilot title",
                "abstract_inverted_index": {"pilot": [0]},
                "publication_date": "2020-01-01",
                "authorships": [
                    {
                        "author": {"id": "https://openalex.org/A1"},
                        "institutions": [
                            {
                                "id": "https://openalex.org/I1",
                                "display_name": "Firm",
                                "type": "company",
                            },
                            {
                                "id": "https://openalex.org/U1",
                                "display_name": "University",
                                "type": "education",
                            },
                        ],
                    }
                ],
            }

    projects = pd.DataFrame({"work_id": ["W1"], "focal_author_ids": ["A1"]})
    client = FakeClient()
    details = fetch_project_details_singletons(client, projects, tmp_path)
    assert client.calls[0][0] == "/works/W1"
    assert details.loc[0, "actual_company_ids"] == "I1"
    assert details.loc[0, "focal_education_ids"] == "U1"

    second = fetch_project_details_singletons(client, projects, tmp_path)
    assert len(client.calls) == 1
    pd.testing.assert_frame_equal(details, second)


def test_cognitive_datetime_normalizer_makes_dates_utc():
    import run_stage2_realization_pilot as runner

    normalizer = getattr(runner, "_normalize_cognitive_datetime_columns", None)
    assert callable(normalizer), "runner must normalize cognitive-fit dates before comparison"

    project_texts = pd.DataFrame({
        "work_id": ["W1"],
        "publication_date": ["2000-05-18"],
        "project_text": ["project text"],
    })
    text_history = pd.DataFrame({
        "focal_work_id": ["W1"],
        "company_id": ["I1"],
        "publication_date": pd.to_datetime(["1999-05-18"]),
        "title": ["prior work"],
        "abstract_inverted_index": [{"prior": [0], "work": [1]}],
    })

    normalized_projects, normalized_history = normalizer(project_texts, text_history)

    assert str(normalized_projects["publication_date"].dtype).endswith(", UTC]")
    assert str(normalized_history["publication_date"].dtype).endswith(", UTC]")
