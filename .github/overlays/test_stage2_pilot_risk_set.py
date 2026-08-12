import pandas as pd
from stage2_pilot_risk_set import assemble_pilot_candidates, inject_realized_firms


def test_weak_university_tie_is_attribute_not_candidate_source():
    projects = pd.DataFrame({
        "work_id": ["W1"],
        "publication_year": [2020],
        "publication_date": ["2020-06-01"],
        "primary_subfield_id": [100.0],
        "primary_field_id": [10],
        "compot": [0.5],
        "focal_author_ids": ["A1"],
    })
    details = pd.DataFrame({
        "work_id": ["W1"],
        "actual_company_ids": ["C9"],
        "focal_education_ids": ["U1"],
    })
    author_history = pd.DataFrame(columns=["author_id", "company_id", "evidence_date", "prior_work_id"])
    university_history = pd.DataFrame({
        "university_id": ["U1"] * 6,
        "company_id": ["Cweak"] + ["Cstrong"] * 5,
        "evidence_date": ["2019-01-01"] * 6,
        "prior_work_id": ["P0", "P1", "P2", "P3", "P4", "P5"],
    })
    field_counts = pd.DataFrame({
        "subfield_id": [100],
        "as_of_date": ["2020-06-01"],
        "company_id": ["Cfield"],
        "prior_subfield_publication_count": [20],
        "field_rank": [1],
        "query_from_date": ["2015-06-01"],
        "query_to_date": ["2020-05-31"],
    })

    natural, audit = assemble_pilot_candidates(
        projects, details, author_history, university_history, field_counts,
        field_top_n=50, university_min_works=5,
    )

    assert set(natural["company_id"]) == {"Cstrong", "Cfield"}
    strong = natural.set_index("company_id").loc["Cstrong"]
    assert strong["university_prior_partner"] == 1
    assert strong["university_prior_work_count"] == 5
    assert strong["strong_university_candidate"] == 1
    assert audit.loc[0, "natural_project_recall"] == 0


def test_weak_university_tie_is_retained_as_attribute_for_field_candidate():
    projects = pd.DataFrame({
        "work_id": ["W1"], "publication_year": [2020],
        "publication_date": ["2020-06-01"], "primary_subfield_id": [100],
        "primary_field_id": [10], "compot": [0.5], "focal_author_ids": ["A1"],
    })
    details = pd.DataFrame({"work_id": ["W1"], "actual_company_ids": ["Cweak"], "focal_education_ids": ["U1"]})
    author_history = pd.DataFrame(columns=["author_id", "company_id", "evidence_date", "prior_work_id"])
    university_history = pd.DataFrame({
        "university_id": ["U1"], "company_id": ["Cweak"],
        "evidence_date": ["2019-01-01"], "prior_work_id": ["P0"],
    })
    field_counts = pd.DataFrame({
        "subfield_id": [100], "as_of_date": ["2020-06-01"], "company_id": ["Cweak"],
        "prior_subfield_publication_count": [20], "field_rank": [1],
        "query_from_date": ["2015-06-01"], "query_to_date": ["2020-05-31"],
    })
    natural, _ = assemble_pilot_candidates(projects, details, author_history, university_history, field_counts, field_top_n=50, university_min_works=5)
    row = natural.iloc[0]
    assert row["university_prior_partner"] == 1
    assert row["university_prior_work_count"] == 1
    assert row["strong_university_candidate"] == 0
    assert row["subfield_active_company"] == 1


def test_realized_firm_is_injected_after_natural_recall():
    natural = pd.DataFrame({
        "work_id": ["W1", "W1"], "company_id": ["C1", "C2"],
        "selected": [0, 0], "natural_candidate": [1, 1],
        "forced_selected_candidate": [0, 0],
    })
    projects = pd.DataFrame({
        "work_id": ["W1"], "publication_year": [2020],
        "publication_date": ["2020-06-01"], "primary_subfield_id": [100],
        "primary_field_id": [10], "compot": [0.5], "focal_author_ids": ["A1"],
    })
    details = pd.DataFrame({"work_id": ["W1"], "actual_company_ids": ["C9"], "focal_education_ids": ["U1"]})
    estimation = inject_realized_firms(natural, projects, details)
    injected = estimation.set_index("company_id").loc["C9"]
    assert injected["selected"] == 1
    assert injected["natural_candidate"] == 0
    assert injected["forced_selected_candidate"] == 1
    assert estimation.groupby(["work_id", "company_id"]).size().eq(1).all()


def test_injected_realized_firm_relationships_are_enriched():
    from stage2_pilot_risk_set import enrich_candidate_relationships

    projects = pd.DataFrame({
        "work_id": ["W1"], "publication_year": [2020],
        "publication_date": ["2020-06-01"], "primary_subfield_id": [100],
        "primary_field_id": [10], "compot": [0.5], "focal_author_ids": ["A1"],
    })
    details = pd.DataFrame({"work_id": ["W1"], "actual_company_ids": ["C9"], "focal_education_ids": ["U1"]})
    injected = inject_realized_firms(pd.DataFrame(columns=["work_id", "company_id"]), projects, details)
    author_history = pd.DataFrame({
        "author_id": ["A1"], "company_id": ["C9"],
        "evidence_date": ["2019-01-01"], "prior_work_id": ["P1"],
    })
    university_history = pd.DataFrame({
        "university_id": ["U1"], "company_id": ["C9"],
        "evidence_date": ["2018-01-01"], "prior_work_id": ["P2"],
    })
    enriched = enrich_candidate_relationships(
        injected, projects, details, author_history, university_history,
        university_min_works=5,
    )
    row = enriched.iloc[0]
    assert row["forced_selected_candidate"] == 1
    assert row["author_prior_partner"] == 1
    assert row["author_relationship_strength"] == 1
    assert row["university_prior_partner"] == 1
    assert row["university_prior_work_count"] == 1
    assert row["strong_university_candidate"] == 0
