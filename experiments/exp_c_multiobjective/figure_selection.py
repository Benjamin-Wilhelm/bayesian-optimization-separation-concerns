"""Presentation-only selection and reversible migration of existing figures."""
from __future__ import annotations

import re
from pathlib import Path

from common.serialization import load_json, save_json


def figure_directory(fig_dir: Path, name: str) -> Path:
    """Keep query maps, regrets, coverage and paired comparisons in the main set."""
    if name.endswith("_paired_bootstrap_differences"):
        # Legacy plots mixed metrics with different units on a single axis.
        return fig_dir / "archive"
    supplementary = (
        re.fullmatch(r"fixed_.+_angle[^_]+_final_.+", name)
        or re.fullmatch(r"fixed_.+_angle[^_]+_incumbents_seed\d+", name)
        or re.fullmatch(r"fixed_.+_preference_map_seed\d+", name)
        or re.fullmatch(r"fixed_.+_trajectory_summary_seed\d+", name)
        or name in {"pareto_hypervolume", "pareto_bayes_regret",
                    "pareto_average_dominance_regret"}
        or name.startswith(("pareto_decision_space_", "pareto_objective_space_"))
        or name in {
            f"{prefix}{metric}"
            for prefix in ("pareto_final_", "pareto_front_paired_bootstrap_")
            for metric in ("empirical_bayes_regret", "average_dominance_regret")
        }
    )
    return fig_dir / "supplementary" if supplementary else fig_dir


def _preserve_move(source: Path, destination: Path) -> Path:
    """Move without overwriting any existing artifact, including on retries."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = destination
    index = 1
    while candidate.exists():
        candidate = destination.with_name(f"{destination.stem}_{index}{destination.suffix}")
        index += 1
    source.rename(candidate)
    return candidate


def organize_existing_figures(fig_dir: Path) -> None:
    """Reclassify known PNG/PDF files; preserve old fixed three-panel versions.

    Unknown figures, raw data and processed metrics are left alone. The manifest
    records relative paths and makes the migration idempotent after completion.
    """
    manifest_path = fig_dir / "figure_selection.json"
    manifest = (load_json(manifest_path) if manifest_path.exists()
                else {"layout_version": 1, "moves": []})
    first_migration = not manifest_path.exists()
    for source in sorted(fig_dir.glob("*")):
        if not source.is_file() or source.suffix not in (".png", ".pdf"):
            continue
        destination_dir = figure_directory(fig_dir, source.stem)
        if (first_migration
                and re.fullmatch(r"fixed_.+_angle\d+", source.stem)):
            destination = fig_dir / "archive" / f"{source.stem}_with_best_so_far{source.suffix}"
        elif destination_dir != fig_dir:
            destination = destination_dir / source.name
        else:
            continue
        moved = _preserve_move(source, destination)
        manifest["moves"].append({"from": source.name,
                                  "to": str(moved.relative_to(fig_dir))})
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, manifest_path)
