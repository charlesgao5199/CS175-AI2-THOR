"""Generate tables and plots from ObjectNav evaluation results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


def _load_rows(results_path: Path) -> List[Dict[str, str]]:
    with results_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _is_success(row: Dict[str, str]) -> bool:
    return row.get("success", "").lower() == "true"


def _has_error(row: Dict[str, str]) -> bool:
    return bool(row.get("error", "").strip())


def _step_value(row: Dict[str, str]) -> int | None:
    value = row.get("steps_taken", "")
    if not value:
        return None
    return int(value)


def _average(values: Iterable[int]) -> float | None:
    values = list(values)
    if not values:
        return None
    return mean(values)


def _metrics(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    episodes = len(rows)
    successes = sum(1 for row in rows if _is_success(row))
    errors = sum(1 for row in rows if _has_error(row))
    timeouts = sum(1 for row in rows if row.get("stop_reason") == "max_steps_reached")
    step_values = [_step_value(row) for row in rows]
    valid_steps = [value for value in step_values if value is not None]
    success_steps = [
        value
        for row, value in zip(rows, step_values)
        if value is not None and _is_success(row)
    ]
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "errors": errors,
        "timeouts": timeouts,
        "timeout_rate": timeouts / episodes if episodes else 0.0,
        "average_steps_all": _average(valid_steps),
        "average_steps_successes": _average(success_steps),
    }


def _group_rows(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(sorted(grouped.items()))


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _metrics_table(metrics: Dict[str, Any]) -> str:
    return _markdown_table(
        ["metric", "value"],
        [
            ["episodes", str(metrics["episodes"])],
            ["successes", str(metrics["successes"])],
            ["success rate", _format_percent(metrics["success_rate"])],
            ["timeouts", str(metrics["timeouts"])],
            ["timeout rate", _format_percent(metrics["timeout_rate"])],
            ["errors", str(metrics["errors"])],
            ["avg steps all", _format_float(metrics["average_steps_all"])],
            ["avg steps successes", _format_float(metrics["average_steps_successes"])],
        ],
    )


def _group_table(rows: List[Dict[str, str]], key: str, label: str) -> str:
    table_rows: List[List[str]] = []
    for group_name, group_rows in _group_rows(rows, key).items():
        metrics = _metrics(group_rows)
        table_rows.append(
            [
                group_name,
                str(metrics["episodes"]),
                str(metrics["successes"]),
                _format_percent(metrics["success_rate"]),
                _format_float(metrics["average_steps_successes"]),
                str(metrics["timeouts"]),
                str(metrics["errors"]),
            ]
        )
    return _markdown_table(
        [label, "episodes", "successes", "success rate", "avg success steps", "timeouts", "errors"],
        table_rows,
    )


def _combo_table(rows: List[Dict[str, str]]) -> str:
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["target_object_type"])].append(row)

    table_rows: List[List[str]] = []
    for (scene, target), group_rows in sorted(grouped.items()):
        metrics = _metrics(group_rows)
        table_rows.append(
            [
                scene,
                target,
                str(metrics["episodes"]),
                str(metrics["successes"]),
                _format_percent(metrics["success_rate"]),
                _format_float(metrics["average_steps_successes"]),
            ]
        )
    return _markdown_table(
        ["scene", "target", "episodes", "successes", "success rate", "avg success steps"],
        table_rows,
    )


def _bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1 if percent else max(values + [1]) * 1.15)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)

    for bar, value in zip(bars, values):
        label = _format_percent(value) if percent else _format_float(value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _success_heatmap(rows: List[Dict[str, str]], output_path: Path) -> None:
    scenes = sorted({row["scene"] for row in rows})
    targets = sorted({row["target_object_type"] for row in rows})
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["target_object_type"])].append(row)

    matrix = [
        [_metrics(grouped[(scene, target)])["success_rate"] for target in targets]
        for scene in scenes
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Success Rate by Scene and Target")
    ax.set_xticks(range(len(targets)), labels=targets)
    ax.set_yticks(range(len(scenes)), labels=scenes)

    for scene_index, scene in enumerate(scenes):
        for target_index, target in enumerate(targets):
            value = _metrics(grouped[(scene, target)])["success_rate"]
            color = "white" if value >= 0.55 else "black"
            ax.text(
                target_index,
                scene_index,
                _format_percent(value),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_plots(output_dir: Path, rows: List[Dict[str, str]]) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = [
        output_dir / "success_by_scene.png",
        output_dir / "success_by_target.png",
        output_dir / "steps_by_target.png",
        output_dir / "success_by_scene_target.png",
    ]

    by_scene = _group_rows(rows, "scene")
    _bar_chart(
        labels=list(by_scene.keys()),
        values=[_metrics(group_rows)["success_rate"] for group_rows in by_scene.values()],
        title="Success Rate by Scene",
        ylabel="success rate",
        output_path=plot_paths[0],
        percent=True,
    )

    by_target = _group_rows(rows, "target_object_type")
    _bar_chart(
        labels=list(by_target.keys()),
        values=[_metrics(group_rows)["success_rate"] for group_rows in by_target.values()],
        title="Success Rate by Target",
        ylabel="success rate",
        output_path=plot_paths[1],
        percent=True,
    )
    _bar_chart(
        labels=list(by_target.keys()),
        values=[
            _metrics(group_rows)["average_steps_successes"] or 0
            for group_rows in by_target.values()
        ],
        title="Average Steps for Successful Episodes by Target",
        ylabel="steps",
        output_path=plot_paths[2],
    )
    _success_heatmap(rows, output_path=plot_paths[3])
    return plot_paths


def _failure_samples(rows: List[Dict[str, str]], limit: int = 10) -> str:
    failures = [row for row in rows if not _is_success(row)]
    if not failures:
        return "No failed episodes."

    table_rows = [
        [
            row["episode_id"],
            row["scene"],
            row["target_object_type"],
            row["seed"],
            row["stop_reason"],
            row["steps_taken"],
            row.get("error", ""),
        ]
        for row in failures[:limit]
    ]
    return _markdown_table(
        ["episode", "scene", "target", "seed", "stop reason", "steps", "error"],
        table_rows,
    )


def _write_report(
    evaluation_dir: Path,
    rows: List[Dict[str, str]],
    plot_paths: List[Path],
) -> Path:
    metrics = _metrics(rows)
    report_path = evaluation_dir / "analysis.md"
    relative_plots = [path.name for path in plot_paths]
    report = "\n\n".join(
        [
            "# Random Baseline Evaluation Analysis",
            f"Source: `{evaluation_dir / 'results.csv'}`",
            "## Overall",
            _metrics_table(metrics),
            "## By Scene",
            _group_table(rows, "scene", "scene"),
            "## By Target",
            _group_table(rows, "target_object_type", "target"),
            "## By Scene And Target",
            _combo_table(rows),
            "## Plots",
            "\n".join(f"- `{plot}`" for plot in relative_plots),
            "## Failure Samples",
            _failure_samples(rows),
            "",
        ]
    )
    report_path.write_text(report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", help="Directory containing results.csv.")
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    results_path = evaluation_dir / "results.csv"
    rows = _load_rows(results_path)
    plot_paths = _write_plots(output_dir=evaluation_dir, rows=rows)
    report_path = _write_report(
        evaluation_dir=evaluation_dir,
        rows=rows,
        plot_paths=plot_paths,
    )
    print(report_path)
    for plot_path in plot_paths:
        print(plot_path)


if __name__ == "__main__":
    main()
