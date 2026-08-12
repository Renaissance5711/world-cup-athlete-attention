import pandas as pd

from run_stage2_full_merge import (
    build_full_realized_company_audit,
    merge_shard_candidate_frames,
    required_full_output_filenames,
)


def _candidate(work_id: str, company: str = "C1") -> pd.DataFrame:
    return pd.DataFrame({
        "work_id": [work_id],
        "company_id": [company],
        "selected": [1],
    })


def test_merge_requires_disjoint_project_company_keys_and_complete_project_count():
    merged = merge_shard_candidate_frames(
        [_candidate("W1"), _candidate("W2")], expected_projects=2
    )
    assert set(merged["work_id"]) == {"W1", "W2"}
    try:
        merge_shard_candidate_frames(
            [_candidate("W1"), _candidate("W1")], expected_projects=1
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate project-company rejection")


def test_full_realized_company_audit_tracks_input_and_analyzable_universes():
    shard0 = pd.DataFrame({
        "work_id": ["W1", "W2"],
        "realized_company_resolved": [True, False],
        "stageB_exclusion_reason": ["", "NO_REALIZED_COMPANY_IN_OPENALEX"],
    })
    shard1 = pd.DataFrame({
        "work_id": ["W3"],
        "realized_company_resolved": [True],
        "stageB_exclusion_reason": [""],
    })

    audit, summary = build_full_realized_company_audit(
        [shard0, shard1], expected_input_projects=3
    )

    assert len(audit) == 3
    assert summary == {
        "input_projects": 3,
        "stageB_analyzable_projects": 2,
        "stageB_excluded_projects": 1,
        "stageB_excluded_share": 1 / 3,
    }


def test_full_merge_declares_primary_and_compot_robustness_outputs():
    names = set(required_full_output_filenames())
    assert {
        "full_cognitive_candidate_long_top50.csv",
        "full_cognitive_candidate_long_top100.csv",
        "full_project_realization_metrics_top50.csv",
        "full_project_realization_metrics_top100.csv",
        "full_ranking_coefficients_top50.csv",
        "full_ranking_metrics_top50.csv",
        "full_ranking_coefficients_top100.csv",
        "full_ranking_metrics_top100.csv",
        "full_compot_ranking_coefficients_top50.csv",
        "full_compot_ranking_metrics_top50.csv",
        "full_compot_ranking_coefficients_top100.csv",
        "full_compot_ranking_metrics_top100.csv",
        "full_realized_company_audit.csv",
        "full_time_provenance_audit.json",
        "full_stage2_summary.json",
    }.issubset(names)
