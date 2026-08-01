from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ijsms.rank_reversal import (  # noqa: E402
    bootstrap_reversal_statistics,
    build_athlete_displacements,
    build_pairwise_decomposition,
    summarize_decomposition,
    validate_rank_input,
)

DEFAULT_RANKING = ROOT / "outputs/r24/observed_goal_scorer_metric_rankings.csv"
DEFAULT_EVENTS = ROOT / "base_archive/data/processed/all_player_match_outcomes_2022.csv"
DEFAULT_OUTPUT = ROOT / "outputs/r25"


def _interval(values: pd.Series) -> list[float]:
    return [float(values.quantile(0.025)), float(values.quantile(0.975))]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _compact_table(summary: dict[str, Any]) -> pd.DataFrame:
    reversal_ci = summary["bootstrap_observed_reversal_rate_ci_95"]
    agreement_ci = summary["bootstrap_identity_agreement_rate_ci_95"]
    return pd.DataFrame(
        [
            {
                "Statistic": "Total unordered athlete pairs",
                "Estimate": summary["total_pairs"],
                "Percent_or_interval": "100% of possible unordered pairs",
                "Definition": "All unique pairs among the 117 observed scorers.",
            },
            {
                "Statistic": "Comparable pairs",
                "Estimate": summary["comparable_pairs"],
                "Percent_or_interval": f'{100 * summary["comparable_pairs"] / summary["total_pairs"]:.1f}%',
                "Definition": "Pairs without a tie on either observed estimand.",
            },
            {
                "Statistic": "Observed rank reversals",
                "Estimate": summary["observed_reversal_n"],
                "Percent_or_interval": (
                    f'{100 * summary["observed_reversal_rate"]:.1f}%; '
                    f'bootstrap 95% CI {100 * reversal_ci[0]:.1f}-{100 * reversal_ci[1]:.1f}%'
                ),
                "Definition": "Comparable pairs ordered in opposite directions by observed proportional and additive scores.",
            },
            {
                "Statistic": "Observed concordant pairs",
                "Estimate": summary["observed_concordant_n"],
                "Percent_or_interval": f'{100 * summary["observed_concordant_n"] / summary["comparable_pairs"]:.1f}%',
                "Definition": "Comparable pairs ordered in the same direction by both observed estimands.",
            },
            {
                "Statistic": "Pairs tied on either observed estimand",
                "Estimate": summary["tied_pairs"],
                "Percent_or_interval": f'{100 * summary["tied_pairs"] / summary["total_pairs"]:.2f}%',
                "Definition": "Pairs excluded only from the concordant-versus-reversed denominator.",
            },
            {
                "Statistic": "Identity-observed agreement",
                "Estimate": summary["identity_observed_agreement_n"],
                "Percent_or_interval": (
                    f'{100 * summary["identity_observed_agreement_rate"]:.1f}%; '
                    f'bootstrap 95% CI {100 * agreement_ci[0]:.1f}-{100 * agreement_ci[1]:.1f}%'
                ),
                "Definition": "Comparable pairs for which the denominator diagnostic and primary observed classification agree.",
            },
            {
                "Statistic": "Median absolute athlete rank displacement",
                "Estimate": summary["median_absolute_rank_displacement"],
                "Percent_or_interval": "ranks",
                "Definition": "Median absolute difference between additive and proportional rank positions.",
            },
            {
                "Statistic": "Athletes moving at least 25 ranks",
                "Estimate": summary["athletes_moving_at_least_25_ranks"],
                "Percent_or_interval": f'{100 * summary["athletes_moving_at_least_25_ranks"] / summary["unique_scorers"]:.1f}%',
                "Definition": "Athletes whose absolute rank displacement is 25 positions or more.",
            },
        ]
    )


def run(
    ranking_path: Path = DEFAULT_RANKING,
    events_path: Path = DEFAULT_EVENTS,
    output_dir: Path = DEFAULT_OUTPUT,
    draws: int = 10000,
    seed: int = 20260730,
) -> dict[str, Any]:
    ranking_path = Path(ranking_path)
    events_path = Path(events_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = pd.read_csv(ranking_path)
    validate_rank_input(rankings)
    pairs = build_pairwise_decomposition(rankings)
    athletes = build_athlete_displacements(rankings)

    events = pd.read_csv(events_path)
    scoring = events.loc[
        events["scored_int"].eq(1),
        [
            "player_id",
            "match_id",
            "immediate_attention_log_lift",
            "winsorized_additional_pageviews",
            "baseline_views",
        ],
    ].copy()
    bootstrap = bootstrap_reversal_statistics(
        scoring=scoring, draws=draws, seed=seed, batch_size=250
    )

    summary: dict[str, Any] = summarize_decomposition(pairs, athletes)
    summary.update(
        {
            "unique_scorers": int(len(rankings)),
            "seed": int(seed),
            "bootstrap_draws": int(draws),
            "observed_estimands": {
                "proportional": "mean immediate_attention_log_lift across scoring appearances",
                "additive": "mean winsorized_additional_pageviews across scoring appearances",
            },
            "denominator_diagnostic": "log1p(observed_additional_pageviews / baseline_views)",
            "tie_rule": "Exact stored-precision ties are reported separately and excluded only from the concordant-versus-reversed denominator.",
            "bootstrap_method": "Match-weighted Bayesian bootstrap over scoring appearances with shared Dirichlet match weights.",
            "bootstrap_observed_reversal_rate_median": float(
                bootstrap["observed_reversal_rate"].median()
            ),
            "bootstrap_observed_reversal_rate_ci_95": _interval(
                bootstrap["observed_reversal_rate"]
            ),
            "bootstrap_identity_reversal_rate_median": float(
                bootstrap["identity_reversal_rate"].median()
            ),
            "bootstrap_identity_reversal_rate_ci_95": _interval(
                bootstrap["identity_reversal_rate"]
            ),
            "bootstrap_identity_agreement_rate_median": float(
                bootstrap["identity_agreement_rate"].median()
            ),
            "bootstrap_identity_agreement_rate_ci_95": _interval(
                bootstrap["identity_agreement_rate"]
            ),
            "interpretation": (
                "The observed decomposition quantifies ordering changes between primary observed estimands. "
                "The denominator diagnostic is an accounting comparison, while residual disagreement "
                "reflects aggregation, transformation, and winsorisation differences."
            ),
        }
    )
    summary = _to_builtin(summary)

    pairs.to_csv(output_dir / "rank_reversal_pairs.csv", index=False, encoding="utf-8")
    athletes.to_csv(
        output_dir / "rank_reversal_athlete_displacements.csv",
        index=False,
        encoding="utf-8",
    )
    bootstrap.to_csv(
        output_dir / "rank_reversal_bootstrap_draws.csv", index=False, encoding="utf-8"
    )
    _compact_table(summary).to_csv(
        output_dir / "rank_reversal_decomposition_table.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "rank_reversal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
