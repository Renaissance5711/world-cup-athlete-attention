import pandas as pd
from stage2_pilot_models import fit_temporal_ranking_models


def make_candidate_panel() -> pd.DataFrame:
    rows = []
    for year in range(2015, 2024):
        for project_index in range(8):
            work_id = f"W{year}_{project_index}"
            for company_index in range(5):
                fit = (5 - company_index) / 5
                relation = int(company_index == 1)
                selected = int(company_index == (0 if year >= 2019 else 1))
                rows.append(
                    {
                        "work_id": work_id,
                        "publication_year": year,
                        "company_id": f"C{company_index}",
                        "selected": selected,
                        "cognitive_fit_cosine": fit,
                        "cognitive_fit_publication_count": 2,
                        "prior_subfield_publication_count": 10 - company_index,
                        "author_prior_partner": relation,
                        "university_prior_partner": relation,
                        "strong_university_candidate": relation,
                        "author_relationship_strength": 3 * relation,
                        "author_recency_years": 1.0 if relation else 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_models_emit_temporal_test_ranking_metrics():
    coefficients, metrics = fit_temporal_ranking_models(
        make_candidate_panel(), train_end_year=2018, bootstrap_reps=30
    )
    assert set(coefficients["model"]) == {"technical", "relational", "combined"}
    assert set(metrics["model"]) == {"technical", "relational", "combined"}
    project_metrics = metrics[~metrics["is_aggregate"]]
    assert project_metrics["publication_year"].min() >= 2019
    assert {
        "selected_best_rank",
        "reciprocal_rank",
        "recall_at_5",
        "recall_at_10",
        "average_precision",
    }.issubset(metrics.columns)
    aggregate = metrics[metrics["is_aggregate"]]
    assert len(aggregate) == 3
    assert aggregate["mean_reciprocal_rank"].notna().all()


def test_temporal_model_rejects_missing_recency_inputs():
    frame = make_candidate_panel().drop(columns=["author_recency_years"])
    try:
        fit_temporal_ranking_models(frame, train_end_year=2018, bootstrap_reps=5)
    except ValueError as exc:
        assert "recency" in str(exc).lower()
    else:
        raise AssertionError("Expected a clear recency-input error")
