import pandas as pd
from stage2_pilot_sampling import select_stratified_pilot_projects


def make_projects() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2025):
        for field in [10, 20, 30, 40]:
            for index in range(8):
                rows.append({
                    "work_id": f"W{year}_{field}_{index}",
                    "publication_year": year,
                    "primary_field_id": field,
                    "compot": (index + 1) / 9,
                })
    return pd.DataFrame(rows)


def test_stratified_sample_is_deterministic_and_unique():
    projects = make_projects()
    first, first_audit = select_stratified_pilot_projects(projects, sample_size=400, seed=20260804)
    second, second_audit = select_stratified_pilot_projects(projects, sample_size=400, seed=20260804)
    assert first["work_id"].tolist() == second["work_id"].tolist()
    assert first["work_id"].is_unique
    assert len(first) == 400
    pd.testing.assert_frame_equal(first_audit, second_audit)


def test_stratified_sample_preserves_all_nonempty_macro_periods():
    projects = make_projects()
    sample, audit = select_stratified_pilot_projects(projects, sample_size=400)
    assert set(sample["pilot_period"]) == {"2000-2006", "2007-2012", "2013-2018", "2019-2024"}
    assert audit["sampled_n"].sum() == 400
    assert audit.loc[audit["population_n"].gt(0), "sampled_n"].gt(0).all()
