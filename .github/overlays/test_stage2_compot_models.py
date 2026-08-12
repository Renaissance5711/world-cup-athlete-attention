import numpy as np
import pandas as pd

from stage2_compot_models import (
    fit_compot_temporal_ranking_models,
    prepare_compot_features,
)
from stage2_pilot_models import fit_temporal_ranking_models


def make_candidate_panel() -> pd.DataFrame:
    rows = []
    for year in range(2015, 2024):
        for project_index in range(8):
            work_id = f"W{year}_{project_index}"
            compot = (project_index + 1) / 10 + (year - 2015) * 0.01
            for company_index in range(5):
                fit = (5 - company_index) / 5
                relation = int(company_index == 1)
                selected = int(company_index == (0 if year >= 2019 else 1))
                rows.append({
                    "work_id": work_id,
                    "publication_year": year,
                    "company_id": f"C{company_index}",
                    "selected": selected,
                    "compot": compot,
                    "cognitive_fit_cosine": fit,
                    "cognitive_fit_publication_count": 2,
                    "prior_subfield_publication_count": 10 - company_index,
                    "author_prior_partner": relation,
                    "university_prior_partner": relation,
                    "strong_university_candidate": relation,
                    "author_relationship_strength": 3 * relation,
                    "author_recency_years": 1.0 if relation else 5.0,
                })
    return pd.DataFrame(rows)


def test_compot_scaling_uses_training_projects_only():
    frame = make_candidate_panel()
    frame.loc[frame["publication_year"].le(2018), "compot"] = frame.loc[
        frame["publication_year"].le(2018), "work_id"
    ].str.extract(r"_(\d+)$")[0].astype(float).to_numpy()
    frame.loc[frame["publication_year"].gt(2018), "compot"] = 100.0
    prepared = prepare_compot_features(frame, train_end_year=2018)
    train_projects = (
        prepared.loc[prepared["publication_year"].le(2018), ["work_id", "compot_z"]]
        .drop_duplicates("work_id")
    )
    assert abs(float(train_projects["compot_z"].mean())) < 1e-9
    assert prepared.loc[prepared["publication_year"].gt(2018), "compot_z"].min() > 10


def test_compot_models_never_include_compot_main_effect():
    coefficients, _ = fit_compot_temporal_ranking_models(
        make_candidate_panel(), train_end_year=2018, bootstrap_reps=5
    )
    assert "compot_z" not in set(coefficients["term"])
    assert set(coefficients["model"]) == {
        "B0_combined", "B1_relationship_compot", "B2_fit_compot"
    }


def test_b0_matches_existing_combined_baseline_metrics():
    frame = make_candidate_panel()
    _, baseline = fit_temporal_ranking_models(
        frame, train_end_year=2018, bootstrap_reps=0
    )
    _, extended = fit_compot_temporal_ranking_models(
        frame, train_end_year=2018, bootstrap_reps=0
    )
    base = baseline[(baseline["model"] == "combined") & baseline["is_aggregate"]].iloc[0]
    b0 = extended[(extended["model"] == "B0_combined") & extended["is_aggregate"]].iloc[0]
    for metric in [
        "mean_selected_best_rank", "mean_reciprocal_rank",
        "mean_recall_at_5", "mean_recall_at_10", "mean_average_precision",
    ]:
        assert np.isclose(float(base[metric]), float(b0[metric]))
