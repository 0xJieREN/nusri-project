from __future__ import annotations

import argparse
from pathlib import Path

from nusri_project.reporting.html_reports import build_index_html, generate_experiment_report


DEFAULT_EXPERIMENTS = [
    "phase2_2024_round2_fast_fixed",
    "phase2_2025_candidates",
    "label_optimization_round1",
    "cost_aware_label_round1_2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML experiment reports.")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--output-root", default="reports/html")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_root = Path(args.reports_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    generated = []
    missing = []
    for name in args.experiments:
        experiment_dir = reports_root / name
        if not experiment_dir.exists():
            missing.append(name)
            continue
        html_path = generate_experiment_report(experiment_dir, output_root / name)
        rel = html_path.relative_to(output_root)
        from nusri_project.reporting.html_reports import _extract_experiment_summary

        generated.append(
            {
                "name": name,
                "href": rel.as_posix(),
                "summary": _extract_experiment_summary(experiment_dir),
            }
        )

    (output_root / "index.html").write_text(build_index_html(generated, missing), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
