#!/usr/bin/env python3
"""Extract and plot actor/entropy against training step from verl logs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import parse_step_records


DEFAULT_LOG_DIR = ROOT / "experiment_data" / "logs"
DEFAULT_OUTPUT = ROOT / "experiment_data" / "plots" / "optimized_actor_entropy.png"


def extract_entropy(log_path: Path, max_step: int | None = None) -> list[tuple[int, float]]:
    """Return sorted (step, actor/entropy) pairs from one log file."""
    records = parse_step_records(log_path)
    return [
        (step, values["actor/entropy"])
        for step, values in records.items()
        if "actor/entropy" in values and (max_step is None or step <= max_step)
    ]


def write_csv(
    output_path: Path,
    series: list[tuple[Path, list[tuple[int, float]]]],
) -> None:
    """Write all extracted values as a long-format CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["log", "step", "actor/entropy"])
        for log_path, points in series:
            for step, entropy in points:
                writer.writerow([log_path.name, step, entropy])


def plot_entropy(
    output_path: Path,
    series: list[tuple[Path, list[tuple[int, float]]]],
    title: str,
    dpi: int,
) -> None:
    """Render all entropy series to one image using a headless backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to draw the figure; install it with "
            "`python3 -m pip install matplotlib`"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(12, 7))
    for log_path, points in series:
        steps, entropies = zip(*points)
        axes.plot(
            steps,
            entropies,
            linewidth=1.6,
            marker="o",
            markersize=2.5,
            label=log_path.stem,
        )

    axes.set_title(title)
    axes.set_xlabel("Step")
    axes.set_ylabel("actor/entropy")
    axes.grid(True, linestyle="--", alpha=0.35)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan optimized*.log files and plot actor/entropy against training step."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"directory containing log files (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--pattern",
        default="optimized*.log",
        help="glob pattern used inside --log-dir (default: optimized*.log)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="extracted CSV path (default: same name as --output with .csv suffix)",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        help="only include records whose step is at most this value",
    )
    parser.add_argument(
        "--title",
        default="Actor Entropy by Training Step",
        help="figure title",
    )
    parser.add_argument("--dpi", type=int, default=180, help="output image DPI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    csv_output = (
        args.csv_output.expanduser().resolve()
        if args.csv_output
        else output_path.with_suffix(".csv")
    )

    if not log_dir.is_dir():
        raise SystemExit(f"log directory does not exist: {log_dir}")

    log_paths = sorted(path for path in log_dir.glob(args.pattern) if path.is_file())
    if not log_paths:
        raise SystemExit(f"no files matched {args.pattern!r} in {log_dir}")

    series: list[tuple[Path, list[tuple[int, float]]]] = []
    for log_path in log_paths:
        points = extract_entropy(log_path, args.max_step)
        if points:
            series.append((log_path, points))
        else:
            print(f"warning: no actor/entropy records in {log_path}", file=sys.stderr)

    if not series:
        raise SystemExit("no actor/entropy records found in matched log files")

    write_csv(csv_output, series)
    try:
        plot_entropy(output_path, series, args.title, args.dpi)
    except RuntimeError as exc:
        print(f"CSV written: {csv_output}")
        raise SystemExit(str(exc)) from exc

    point_count = sum(len(points) for _, points in series)
    print(f"logs: {len(series)}, points: {point_count}")
    print(f"figure: {output_path}")
    print(f"CSV: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
