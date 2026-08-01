from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_rank_reversal_decomposition.py"
spec = spec_from_file_location("rank_reversal_runner", SCRIPT)
assert spec is not None and spec.loader is not None
runner = module_from_spec(spec)
spec.loader.exec_module(runner)
run = runner.run


def test_frozen_rank_decomposition_counts(tmp_path: Path):
    summary = run(output_dir=tmp_path, draws=50, seed=20260730)
    assert summary["unique_scorers"] == 117
    assert summary["total_pairs"] == 6786
    assert summary["comparable_pairs"] == 6780
    assert summary["tied_pairs"] == 6
    assert summary["observed_reversal_n"] == 3058
    assert summary["observed_concordant_n"] == 3722
    assert abs(summary["observed_reversal_rate"] - 0.4510324483775811) < 1e-12
    pairs = pd.read_csv(tmp_path / "rank_reversal_pairs.csv")
    assert len(pairs) == 6786
