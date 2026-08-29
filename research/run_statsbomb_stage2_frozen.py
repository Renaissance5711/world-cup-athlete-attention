"""Frozen Stage-2 launcher enforcing organization-specific historical reference H0."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import research.run_statsbomb_stage2 as core
from research.historical_marginality_own import historical_marginality_targets


def _arg_value(flag: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Missing required argument {flag}") from exc


def main() -> None:
    # Replace only M. J, temporal folds, radicality, recovery, and Figure-5 logic
    # remain exactly as implemented in the already-tested Stage-2 runner.
    core.historical_marginality_targets = historical_marginality_targets
    core.main()

    out = Path(_arg_value("--output-dir"))
    manifest_path = out / "stage2_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["M_definition"] = (
        "strict-prior focal-team own-history componentwise two-sided tail "
        "marginality percentile; H0 uses only the focal organization's earlier games"
    )
    manifest["M_reference_freeze"] = "organization-specific own strict-prior H0"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = out / "STAGE2_RESULTS_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    note = (
        "\n\n## Historical-reference freeze\n\n"
        "Historical configuration position M is calculated against the focal team's "
        "own strict-prior configuration history H0. Other teams are not used as the "
        "reference distribution for M. Because B and C are in the same match, they "
        "share the same frozen strict-prior H0.\n"
    )
    report_path.write_text(report + note, encoding="utf-8")


if __name__ == "__main__":
    main()
