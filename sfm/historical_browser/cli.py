"""CLI for listing, validating, and serving preserved E1-E10 point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .catalog import CatalogError, HistoricalCatalog, SUBJECTS, validate_result
from .combined import CombinedBrowserCatalog
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only selector and browser for preserved SfM E1-E10 results"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="absolute data root (default: CV802_DATA_ROOT or the permanent /l path)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="list E1-E10 and subject availability")
    listing.add_argument("--subject", choices=SUBJECTS)
    listing.add_argument("--json", action="store_true")

    show = commands.add_parser("show", help="show complete metadata for one experiment")
    show.add_argument("experiment")
    show.add_argument("--json", action="store_true")

    path = commands.add_parser("path", help="print the exact preserved display PLY path")
    path.add_argument("experiment")
    path.add_argument("--subject", choices=SUBJECTS, required=True)

    validate = commands.add_parser("validate", help="validate existence and PLY point counts")
    validate.add_argument("experiment", nargs="?")
    validate.add_argument("--subject", choices=SUBJECTS)
    validate.add_argument("--json", action="store_true")

    web = commands.add_parser("serve", help="serve the local interactive WebGL viewer")
    web.add_argument("--bind", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=False))


def _list_rows(
    catalog: CombinedBrowserCatalog, subject_filter: str | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subjects = (subject_filter,) if subject_filter else SUBJECTS
    for experiment in catalog.experiments():
        for subject in subjects:
            result = catalog.result(experiment["id"], subject, require_available=False)
            rows.append(
                {
                    "experiment": experiment["id"],
                    "stage_type": experiment["stage_type"],
                    "subject": subject,
                    "available": result.available,
                    "points": result.point_count,
                    "ply_path": str(result.ply_path) if result.ply_path else None,
                    "on_disk": bool(result.ply_path and result.ply_path.is_file()),
                    "title": experiment["title"],
                    "reason": result.reason,
                }
            )
    return rows


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print(f"{'ID':<4} {'STAGE':<17} {'SUBJECT':<7} {'POINTS':>7}  STATUS  TITLE")
    for row in rows:
        points = f"{row['points']:,}" if row["points"] is not None else "-"
        if not row["available"]:
            status = "not-run"
        elif row["on_disk"]:
            status = "ready"
        else:
            status = "missing"
        print(
            f"{row['experiment']:<4} {row['stage_type']:<17} {row['subject']:<7} "
            f"{points:>7}  {status:<7} {row['title']}"
        )


def _validation_targets(
    catalog: CombinedBrowserCatalog, experiment: str | None, subject: str | None
) -> list[tuple[str, str]]:
    if subject and not experiment:
        raise CatalogError("--subject requires an experiment for validate")
    experiment_ids = [catalog.experiment(experiment)["id"]] if experiment else [
        item["id"] for item in catalog.experiments()
    ]
    subjects = [subject] if subject else list(SUBJECTS)
    return [(experiment_id, name) for experiment_id in experiment_ids for name in subjects]


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        historical = HistoricalCatalog(data_root=arguments.data_root)
        catalog = CombinedBrowserCatalog(historical)
        if arguments.command == "list":
            rows = _list_rows(catalog, arguments.subject)
            _print_json(rows) if arguments.json else _print_rows(rows)
            return 0
        if arguments.command == "show":
            details = catalog.describe(arguments.experiment)
            if arguments.json:
                _print_json(details)
            else:
                print(f"{details['id']} — {details['title']}")
                print(f"Stage: {details['stage_type']}")
                print(f"Method: {details['method']}")
                print(f"Cameras: {details['camera_policy']}")
                print(f"Display: {details['display_variant']}")
                print("Exact portable rerun: no; this command inspects the saved result")
                for subject in SUBJECTS:
                    result = details["subjects"][subject]
                    if result["available"]:
                        print(
                            f"{subject}: {result['point_count']:,} points — {result['ply_path']}"
                        )
                    else:
                        print(f"{subject}: unavailable — {result['reason']}")
            return 0
        if arguments.command == "path":
            print(catalog.result(arguments.experiment, arguments.subject).ply_path)
            return 0
        if arguments.command == "validate":
            reports = [
                validate_result(catalog.result(experiment, subject, require_available=False))
                for experiment, subject in _validation_targets(
                    catalog, arguments.experiment, arguments.subject
                )
            ]
            if arguments.json:
                _print_json(reports)
            else:
                for report in reports:
                    state = "PASS" if report["valid"] else (
                        "N/A" if not report["available"] else "FAIL"
                    )
                    detail = (
                        f"{report['point_count']:,} vertices"
                        if report["valid"]
                        else report.get("error", "unknown error")
                    )
                    print(f"{state:<4} {report['experiment_id']} {report['subject']}: {detail}")
            return 0 if all(report["valid"] or not report["available"] for report in reports) else 1
        if arguments.command == "serve":
            if not 1 <= arguments.port <= 65535:
                raise CatalogError("port must be in the range 1-65535")
            serve(catalog, arguments.bind, arguments.port)
            return 0
    except (CatalogError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
