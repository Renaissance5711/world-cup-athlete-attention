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
