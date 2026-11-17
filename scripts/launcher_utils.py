"""YAML loading shared by the seed-parallel command-line launchers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_launcher_config(path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Read a YAML mapping or report a CLI error before creating run files."""
    path = path.expanduser()
    if not path.is_file():
        parser.error(f"Configuration must be an existing file: {path}")
    if path.suffix.lower() not in (".yaml", ".yml"):
        parser.error(f"Configuration must have a .yaml or .yml extension: {path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        parser.error(f"Cannot read YAML configuration {path}: {exc}")
    if not isinstance(config, dict):
        parser.error(f"Configuration must contain a YAML mapping: {path}")
    return config
