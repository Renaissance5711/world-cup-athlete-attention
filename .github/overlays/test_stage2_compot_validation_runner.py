from run_stage2_compot_validation import required_compot_output_filenames


def test_compot_validation_declares_full_output_contract():
    names = set(required_compot_output_filenames())
    assert {
        "stageA_project_panel.csv",
        "stageA_sample_audit.json",
        "stageA_compot_descriptives.csv",
        "stageA_compot_models.csv",
        "stageA_summary.json",
        "pilot_compot_ranking_coefficients_top50.csv",
        "pilot_compot_ranking_metrics_top50.csv",
        "pilot_compot_conflict_by_quartile_top50.csv",
        "pilot_compot_ranking_coefficients_top100.csv",
        "pilot_compot_ranking_metrics_top100.csv",
        "pilot_compot_conflict_by_quartile_top100.csv",
        "pilot_compot_validation_summary.json",
    }.issubset(names)
