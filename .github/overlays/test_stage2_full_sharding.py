import pandas as pd

from stage2_full_sharding import assign_full_shards


def _projects(n=120):
    rows = []
    years = [2001, 2005, 2008, 2011, 2014, 2017, 2020, 2023]
    for i in range(n):
        rows.append({
            "work_id": f"W{i:04d}",
            "publication_year": years[i % len(years)],
            "primary_field_id": str(20 + (i % 4)),
            "compot": ((i * 37) % 997) / 997,
        })
    return pd.DataFrame(rows)


def test_full_shards_are_deterministic_balanced_and_stratified():
    projects = _projects()
    assignment, audit = assign_full_shards(
        projects, shard_count=5, seed=20260811, expected_projects=None
    )
    shuffled, shuffled_audit = assign_full_shards(
        projects.sample(frac=1, random_state=7),
        shard_count=5,
        seed=20260811,
        expected_projects=None,
    )
    left = assignment.sort_values("work_id").reset_index(drop=True)
    right = shuffled.sort_values("work_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert audit == shuffled_audit
    sizes = assignment.groupby("shard_index").size()
    assert sizes.max() - sizes.min() <= 1
    spread = (
        assignment.groupby(["shard_stratum", "shard_index"]).size()
        .unstack(fill_value=0)
    )
    assert (spread.max(axis=1) - spread.min(axis=1)).max() <= 1
    assert audit["projects"] == len(projects)
    assert audit["shards"] == 5


def test_full_shards_reject_duplicate_project_ids():
    projects = pd.concat([_projects(8), _projects(1)], ignore_index=True)
    try:
        assign_full_shards(projects, expected_projects=None)
    except ValueError as exc:
        assert "unique" in str(exc).lower() or "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate work_id rejection")
