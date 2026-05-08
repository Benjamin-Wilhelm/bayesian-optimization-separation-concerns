"""Run directory management and result logging for BO experiments.

Manages the full lifecycle of a run directory:

    results/<experiment_name>/runs/<run_id>/
        config.yaml
        metadata.json
        raw/<benchmark>/<seed>.json
        processed/
        figures/
        logs/
            run.log

Raw results for Experiment A use a **paired seed** structure: one JSON file
per benchmark and seed, containing all methods evaluated under that seed.
This guarantees that WRONG vs RIGHT comparisons are paired (same Sobol
initial design, same random seed).

Usage::

    from common.logging import RunLogger

    logger = RunLogger.create(
        experiment="exp_a_target_matching",
        tag="branin_full",
        results_root=Path("results"),
    )
    logger.save_config(config_dict)
    logger.save_metadata(metadata_dict)
    logger.save_seed_result("branin_2d", seed=0, result=seed_data_dict)
    logger.save_processed("summary.csv", rows)
    logger.get_log_file()   # returns Path to logs/run.log
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from common.serialization import load_json, save_csv, save_json

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_commit_hash(repo_root: Optional[Path] = None) -> Optional[str]:
    """Return the current HEAD commit hash, or ``None`` if unavailable."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def _git_is_dirty(repo_root: Optional[Path] = None) -> Optional[bool]:
    """Return ``True`` if the working tree has uncommitted changes, else ``False``."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def make_run_id(tag: str) -> str:
    """Build a sortable run identifier from a timestamp and a short tag.

    Format: ``YYYY-MM-DD_HH-MM-SS_<tag>``.

    Args:
        tag: Short human-readable label, e.g. ``"branin_full"`` or
            ``"smoke_1d"``.  Spaces are replaced with underscores.

    Returns:
        Run ID string, e.g. ``"2026-05-08_14-30-00_branin_full"``.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    safe_tag = tag.replace(" ", "_")
    return f"{ts}_{safe_tag}"


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class RunLogger:
    """Manages the directory structure and result files for one experiment run.

    Directory layout created by :meth:`create`::

        <results_root>/<experiment>/runs/<run_id>/
            config.yaml          <- saved by save_config()
            metadata.json        <- saved by save_metadata()
            raw/
                <benchmark>/
                    seed_000.json
                    seed_001.json
                    ...
            processed/           <- save_processed() writes here
            figures/             <- reserved for make_figures.py
            logs/
                run.log          <- Python logging handler target

    Args:
        run_dir: Path to the run directory (must already exist).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        experiment: str,
        tag: str,
        results_root: Union[str, Path] = Path("results"),
        run_id: Optional[str] = None,
        ) -> "RunLogger":
        """Create the run directory tree and return a :class:`RunLogger`.

        Idempotent: if the directory already exists (e.g. from a previous
        partial run with the same ``run_id``), no error is raised.

        Args:
            experiment: Experiment name, e.g. ``"exp_a_target_matching"``.
            tag: Short label describing this run, e.g. ``"branin_full"``.
            results_root: Root directory for all experiment results.
                Defaults to ``results/`` relative to the current working dir.
            run_id: If provided, use this run ID instead of generating one
                from the current timestamp.  Useful for resuming a partial run.

        Returns:
            A :class:`RunLogger` pointing at the new run directory.
        """
        results_root = Path(results_root)
        if run_id is None:
            run_id = make_run_id(tag)

        run_dir = results_root / experiment / "runs" / run_id
        for subdir in ("raw", "processed", "figures", "logs"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        return cls(run_dir)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def save_config(self, config: dict[str, Any]) -> Path:
        """Write the experiment configuration to ``config.json``.

        Args:
            config: Serializable dict loaded from a YAML config file (or
                constructed programmatically).

        Returns:
            Path to the written file.
        """
        path = self.run_dir / "config.json"
        save_json(config, path)
        return path

    def load_config(self) -> dict[str, Any]:
        """Load and return the saved configuration dict."""
        return load_json(self.run_dir / "config.json")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def save_metadata(
        self,
        extra: Optional[dict[str, Any]] = None,
        repo_root: Optional[Path] = None,
        ) -> Path:
        """Write run metadata to ``metadata.json``.

        Automatically records:

        - ISO-8601 UTC timestamp.
        - Python version string.
        - Git commit hash (``None`` if git is unavailable).
        - Git dirty flag (``None`` if git is unavailable).
        - Command used to launch the run (``sys.argv``).
        - OS platform string.

        Args:
            extra: Optional dict of additional fields to merge into the
                metadata (e.g. experiment name, budget, seed list).  Keys
                from ``extra`` take precedence over auto-recorded fields.
            repo_root: Optional path to the git repository root, passed to
                the git helpers.  Defaults to the current working directory.

        Returns:
            Path to the written file.
        """
        meta: dict[str, Any] = {
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_commit_hash(repo_root),
            "git_dirty": _git_is_dirty(repo_root),
            "command": sys.argv,
        }
        if extra:
            meta.update(extra)

        path = self.run_dir / "metadata.json"
        save_json(meta, path)
        return path

    def load_metadata(self) -> dict[str, Any]:
        """Load and return the saved metadata dict."""
        return load_json(self.run_dir / "metadata.json")

    # ------------------------------------------------------------------
    # Raw per-seed results
    # ------------------------------------------------------------------

    def seed_result_path(self, benchmark: str, seed: int) -> Path:
        """Return the canonical path for one benchmark/seed result file.

        Path: ``raw/<benchmark>/seed_<NNN>.json``

        The benchmark subdirectory is created if it does not exist.

        Args:
            benchmark: Benchmark name, e.g. ``"branin_2d"`` or
                ``"sine_trend_1d"``.
            seed: Integer seed index.

        Returns:
            :class:`~pathlib.Path` to the (not necessarily existing) file.
        """
        bench_dir = self.run_dir / "raw" / benchmark
        bench_dir.mkdir(parents=True, exist_ok=True)
        return bench_dir / f"seed_{seed:03d}.json"

    def save_seed_result(
        self,
        benchmark: str,
        seed: int,
        result: dict[str, Any],
        ) -> Path:
        """Save the paired multi-method result for one benchmark/seed.

        Expected top-level structure of ``result``::

            {
              "metadata": {
                "experiment": ...,
                "benchmark": ...,
                "seed": ...,
                "budget": ...,
                ...
              },
              "methods": {
                "wrong_gp_on_abs": {
                  "trajectory": [...],
                  "model_diagnostics": [...],
                  "final_metrics": {...}
                },
                "right_target_ucb": { ... }
              }
            }

        All tensors inside ``result`` must already be converted to Python
        lists before calling this function.  Use
        :func:`~common.serialization.tensors_to_lists` if needed.

        Args:
            benchmark: Benchmark name, e.g. ``"branin_2d"``.
            seed: Integer seed index.
            result: Serializable dict containing all methods for this seed.

        Returns:
            Path to the written file.
        """
        path = self.seed_result_path(benchmark, seed)
        save_json(result, path)
        return path

    def load_seed_result(self, benchmark: str, seed: int) -> dict[str, Any]:
        """Load the result dict for one benchmark/seed.

        Args:
            benchmark: Benchmark name.
            seed: Integer seed index.

        Returns:
            Parsed result dict.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        return load_json(self.seed_result_path(benchmark, seed))

    def iter_seed_results(self, benchmark: str) -> list[dict[str, Any]]:
        """Load all seed result files for a benchmark, sorted by seed.

        Args:
            benchmark: Benchmark name.

        Returns:
            List of parsed result dicts, one per seed file found, ordered by
            seed index (ascending).  Returns an empty list if no files exist.
        """
        bench_dir = self.run_dir / "raw" / benchmark
        if not bench_dir.exists():
            return []
        paths = sorted(bench_dir.glob("seed_*.json"))
        return [load_json(p) for p in paths]

    # ------------------------------------------------------------------
    # Processed outputs
    # ------------------------------------------------------------------

    def save_processed(
        self,
        filename: str,
        data: Union[dict[str, Any], list[dict[str, Any]]],
        fieldnames: Optional[list[str]] = None,
        ) -> Path:
        """Write a processed result file to ``processed/<filename>``.

        Supports JSON (``*.json``) and CSV (``*.csv``) based on the file
        extension.

        Args:
            filename: File name including extension, e.g. ``"summary.csv"``
                or ``"metrics.json"``.
            data: For JSON, any serializable object.  For CSV, a list of
                row dicts.
            fieldnames: For CSV only — optional column ordering.

        Returns:
            Path to the written file.

        Raises:
            ValueError: If a ``*.csv`` extension is used with a non-list
                ``data`` argument.
            ValueError: If the extension is not ``.json`` or ``.csv``.
        """
        path = self.run_dir / "processed" / filename
        suffix = Path(filename).suffix.lower()
        if suffix == ".json":
            save_json(data, path)
        elif suffix == ".csv":
            if not isinstance(data, list):
                raise ValueError(
                    "save_processed with a .csv extension requires a list of dicts."
                )
            save_csv(data, path, fieldnames=fieldnames)
        else:
            raise ValueError(
                f"Unsupported extension '{suffix}'. Use '.json' or '.csv'."
            )
        return path

    def load_processed(
        self, filename: str
        ) -> Union[dict[str, Any], list[dict[str, str]]]:
        """Load a processed result file from ``processed/<filename>``.

        Returns the parsed content as a dict (JSON) or list of dicts (CSV).
        """
        path = self.run_dir / "processed" / filename
        suffix = Path(filename).suffix.lower()
        if suffix == ".json":
            return load_json(path)
        if suffix == ".csv":
            from common.serialization import load_csv
            return load_csv(path)
        raise ValueError(f"Unsupported extension '{suffix}'. Use '.json' or '.csv'.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def get_log_path(self) -> Path:
        """Return the path to the run log file (``logs/run.log``)."""
        return self.run_dir / "logs" / "run.log"

    def setup_file_logger(
        self,
        name: str = "bo_run",
        level: int = logging.INFO,
        ) -> logging.Logger:
        """Create a Python logger that writes to ``logs/run.log``.

        Attaches a :class:`~logging.FileHandler` (append mode) and a
        :class:`~logging.StreamHandler` for console output.  Safe to call
        multiple times — duplicate handlers are not added.

        Args:
            name: Logger name.  Use a unique name per experiment script to
                avoid handler pollution across runs in the same process.
            level: Logging level.  Defaults to ``logging.INFO``.

        Returns:
            Configured :class:`~logging.Logger`.
        """
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        logger.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(self.get_log_path(), mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        return logger

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def run_id(self) -> str:
        """Return the run identifier (the name of the run directory)."""
        return self.run_dir.name

    def __repr__(self) -> str:
        return f"RunLogger(run_dir={self.run_dir})"
