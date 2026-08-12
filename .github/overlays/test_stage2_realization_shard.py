import pandas as pd

from run_stage2_realization_shard import (
    partition_stageb_realized_company_coverage,
    required_shard_output_filenames,
)


def test_realization_shard_declares_compact_output_contract():
    names = set(required_shard_output_filenames())
    assert {
        "shard_projects.csv",
        "shard_config.json",
        "shard_summary.json",
        "shard_realized_company_audit.csv",
        "natural_candidate_audit_top50.csv",
        "natural_candidate_audit_top100.csv",
        "cognitive_candidate_long_top50.csv",
        "cognitive_candidate_long_top100.csv",
        "project_realization_metrics_top50.csv",
        "project_realization_metrics_top100.csv",
        "shard_time_provenance_audit.json",
    }.issubset(names)
    assert "candidate_firm_prepublication_text_history.csv" not in names


def test_stageb_partitions_unresolved_realized_company_with_explicit_audit():
    projects = pd.DataFrame({"work_id": ["W1", "W2"]})
    strict = pd.DataFrame({
        "work_id": ["W1", "W2", "W2"],
        "author_id": ["A1", "A2", "A3"],
    })
    details = pd.DataFrame({
        "work_id": ["W1", "W2"],
        "actual_company_count": [1, 0],
        "actual_company_ids": ["I1", ""],
        "actual_company_names": ["Firm 1", ""],
    })

    analysis_projects, analysis_strict, analysis_details, audit = (
        partition_stageb_realized_company_coverage(
            projects,
            strict,
            details,
            max_unresolved_share=0.60,
        )
    )

    assert analysis_projects["work_id"].tolist() == ["W1"]
    assert analysis_strict["work_id"].tolist() == ["W1"]
    assert analysis_details["work_id"].tolist() == ["W1"]
    assert set(audit["work_id"]) == {"W1", "W2"}
    unresolved = audit.loc[audit["work_id"].eq("W2")].iloc[0]
    assert not bool(unresolved["realized_company_resolved"])
    assert unresolved["stageB_exclusion_reason"] == "NO_REALIZED_COMPANY_IN_OPENALEX"
