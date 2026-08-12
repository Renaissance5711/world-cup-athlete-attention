import pandas as pd

from stage2_compot_conflict import summarize_compot_conflict


def make_conflict_panel() -> pd.DataFrame:
    rows = []
    for project_index, compot in enumerate([0.1, 0.3, 0.7, 0.9], start=1):
        work_id = f"W{project_index}"
        selected_fit = 0.40 + project_index * 0.02
        alternative_fit = selected_fit + (0.12 if project_index in {2, 4} else 0.01)
        rows.extend([
            {
                "work_id": work_id,
                "company_id": f"S{project_index}",
                "selected": 1,
                "compot": compot,
                "cognitive_fit_cosine": selected_fit,
                "cognitive_fit_publication_count": 3,
                "author_prior_partner": 1,
                "university_prior_partner": 0,
                "forced_selected_candidate": 0,
            },
            {
                "work_id": work_id,
                "company_id": f"A{project_index}",
                "selected": 0,
                "compot": compot,
                "cognitive_fit_cosine": alternative_fit,
                "cognitive_fit_publication_count": 3,
                "author_prior_partner": 0,
                "university_prior_partner": 0,
                "forced_selected_candidate": 0,
            },
            {
                "work_id": work_id,
                "company_id": f"B{project_index}",
                "selected": 0,
                "compot": compot,
                "cognitive_fit_cosine": 0.15,
                "cognitive_fit_publication_count": 2,
                "author_prior_partner": 0,
                "university_prior_partner": 0,
                "forced_selected_candidate": 0,
            },
        ])
    return pd.DataFrame(rows)


def test_conflict_quartiles_are_project_level_and_thresholds_monotone():
    summary = summarize_compot_conflict(make_conflict_panel())
    assert set(summary["compot_quartile"]) == {1, 2, 3, 4}
    assert summary["projects"].tolist() == [1, 1, 1, 1]
    for _, row in summary.iterrows():
        assert row["relation_fit_conflict_share_0_02"] >= row["relation_fit_conflict_share_0_05"]
        assert row["relation_fit_conflict_share_0_05"] >= row["relation_fit_conflict_share_0_10"]


def test_conflict_summary_rejects_missing_compot():
    frame = make_conflict_panel().drop(columns=["compot"])
    try:
        summarize_compot_conflict(frame)
    except ValueError as exc:
        assert "compot" in str(exc).lower()
    else:
        raise AssertionError("Expected missing COMPOT error")
