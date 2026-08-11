from run_stage2_realization_shard import required_shard_output_filenames


def test_realization_shard_declares_compact_output_contract():
    names = set(required_shard_output_filenames())
    assert {
        "shard_projects.csv",
        "shard_config.json",
        "shard_summary.json",
        "natural_candidate_audit_top50.csv",
        "natural_candidate_audit_top100.csv",
        "cognitive_candidate_long_top50.csv",
        "cognitive_candidate_long_top100.csv",
        "project_realization_metrics_top50.csv",
        "project_realization_metrics_top100.csv",
        "shard_time_provenance_audit.json",
    }.issubset(names)
    assert "candidate_firm_prepublication_text_history.csv" not in names
