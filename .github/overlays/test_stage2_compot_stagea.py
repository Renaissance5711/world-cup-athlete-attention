import pandas as pd

from stage2_compot_stagea import (
    build_stagea_descriptives,
    build_stagea_project_panel,
    fit_stagea_compot_models,
)


def _small_stagea_panel() -> pd.DataFrame:
    rows = []
    for i in range(24):
        rows.append({
            "work_id": f"W{i}",
            "compot": i / 23,
            "firm_participation": int(i >= 12),
            "publication_year": 2018 + (i % 3),
            "primary_field_id": f"F{i % 2}",
        })
    return pd.DataFrame(rows)


def test_stagea_collapses_duplicate_author_rows_without_changing_project_fields():
    strict = pd.DataFrame({
        "work_id": ["W1", "W1", "W2"],
        "compot": [0.2, 0.2, 0.8],
        "firm_participation": [0, 0, 1],
        "publication_year": [2019, 2019, 2020],
        "primary_field_id": ["F1", "F1", "F2"],
    })
    panel, audit = build_stagea_project_panel(
        strict, expected_projects=None, expected_positive=None
    )
    assert panel["work_id"].tolist() == ["W1", "W2"]
    assert audit["projects"] == 2
    assert audit["positive_projects"] == 1


def test_stagea_rejects_within_project_compot_disagreement():
    strict = pd.DataFrame({
        "work_id": ["W1", "W1"],
        "compot": [0.2, 0.3],
        "firm_participation": [1, 1],
        "publication_year": [2019, 2019],
        "primary_field_id": ["F1", "F1"],
    })
    try:
        build_stagea_project_panel(strict, expected_projects=None, expected_positive=None)
    except ValueError as exc:
        assert "within-project" in str(exc).lower()
    else:
        raise AssertionError("Expected a within-project consistency error")


def test_stagea_outputs_quartile_descriptives_and_compot_models():
    panel, _ = build_stagea_project_panel(
        _small_stagea_panel(), expected_projects=None, expected_positive=None
    )
    descriptives = build_stagea_descriptives(panel)
    models = fit_stagea_compot_models(panel)
    assert set(descriptives["compot_quartile"]) == {1, 2, 3, 4}
    assert {"A1_lpm", "A2_logit", "A3_quartile_lpm"}.issubset(set(models["model"]))
    assert ((models["model"] == "A1_lpm") & (models["term"] == "compot_z")).any()
    assert ((models["model"] == "A2_logit") & (models["term"] == "compot_z")).any()
