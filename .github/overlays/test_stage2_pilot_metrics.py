import pandas as pd
from stage2_pilot_metrics import build_project_realization_metrics, evaluate_pilot_gates


def test_conflict_requires_embedded_selected_and_higher_fit_unembedded_candidate():
    frame = pd.DataFrame([
        {
            "work_id": "W1", "company_id": "Cselected", "selected": 1,
            "natural_candidate": 1, "forced_selected_candidate": 0,
            "author_prior_partner": 1, "strong_university_candidate": 0,
            "cognitive_fit_cosine": 0.30, "cognitive_fit_publication_count": 4,
        },
        {
            "work_id": "W1", "company_id": "Coutside", "selected": 0,
            "natural_candidate": 1, "forced_selected_candidate": 0,
            "author_prior_partner": 0, "strong_university_candidate": 0,
            "cognitive_fit_cosine": 0.42, "cognitive_fit_publication_count": 3,
        },
    ])
    out = build_project_realization_metrics(frame, conflict_threshold=0.05)
    row = out.iloc[0]
    assert row["relation_fit_conflict"] == 1
    assert abs(row["fit_shortfall"] - 0.12) < 1e-9
    assert row["selected_fit_percentile"] == 0.5


def test_missing_fit_stays_missing_not_zero():
    frame = pd.DataFrame([
        {"work_id": "W1", "company_id": "C1", "selected": 1, "natural_candidate": 1,
         "forced_selected_candidate": 0, "author_prior_partner": 1,
         "strong_university_candidate": 0, "cognitive_fit_cosine": 0.0,
         "cognitive_fit_publication_count": 0},
        {"work_id": "W1", "company_id": "C2", "selected": 0, "natural_candidate": 1,
         "forced_selected_candidate": 0, "author_prior_partner": 0,
         "strong_university_candidate": 0, "cognitive_fit_cosine": 0.0,
         "cognitive_fit_publication_count": 0},
    ])
    row = build_project_realization_metrics(frame).iloc[0]
    assert pd.isna(row["fit_shortfall"])
    assert pd.isna(row["relation_fit_conflict"])


def test_gate_decision_requires_three_of_four_passes():
    candidate_long = pd.DataFrame({
        "work_id": ["W1", "W2"], "selected": [1, 1],
        "natural_candidate": [1, 1], "cognitive_fit_publication_count": [2, 2],
    })
    ranking = pd.DataFrame({
        "model": ["technical", "combined"],
        "mean_reciprocal_rank": [0.50, 0.60],
        "mean_recall_at_10": [0.60, 0.61],
    })
    projects = pd.DataFrame({
        "work_id": ["W1", "W2"], "relation_fit_conflict": [1, 0],
    })
    decision = evaluate_pilot_gates(candidate_long, ranking, projects)
    assert decision["passed_gate_count"] == 4
    assert decision["recommendation"] == "GO_FULL_STAGE2"
