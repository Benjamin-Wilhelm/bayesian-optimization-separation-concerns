"""Serialization helpers for BO experiment results.

Provides pure format-conversion utilities with no experiment logic:

- Converting PyTorch tensors to JSON-compatible Python objects.
- Saving and loading JSON files.
- Saving and loading CSV files (via the stdlib ``csv`` module).
- Optional Pickle serialization for heavy intermediate objects.

Nothing in this module creates directories, records metadata, or knows about
run structures — that belongs in :mod:`common.logging`.

Usage::

    from common.serialization import tensor_to_list, save_json, load_json

    data = {"x": tensor_to_list(candidates), "y": tensor_to_list(y_obs)}
    save_json(data, path / "seed_000.json")
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Union

import torch

# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def tensor_to_list(t: torch.Tensor) -> Union[list, float]:
    """Convert a tensor to a JSON-compatible Python object.

    - Scalar (0-D) tensors are returned as a Python ``float``.
    - All other tensors are returned as nested Python lists via ``.tolist()``.

    Args:
        t: Any PyTorch tensor.

    Returns:
        A ``float`` for 0-D tensors, or a (nested) ``list`` otherwise.
    """
    if t.ndim == 0:
        return t.item()
    return t.tolist()


def maybe_tensor_to_list(value: Any) -> Any:
    """Convert a value to JSON-compatible form if it is a tensor; pass through otherwise.

    Useful when building dicts that may contain a mix of tensors and plain
    Python objects.

    Args:
        value: Any Python object.

    Returns:
        Converted value if ``value`` is a :class:`torch.Tensor`, else ``value``
        unchanged.
    """
    if isinstance(value, torch.Tensor):
        return tensor_to_list(value)
    return value


def tensors_to_lists(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert all tensor values in a dict to JSON-compatible lists.

    Traverses nested dicts.  Lists and tuples are traversed element-wise.
    All other non-tensor values are passed through unchanged.

    Args:
        d: Arbitrary nested dict, possibly containing tensors.

    Returns:
        New dict with the same structure but tensors replaced by lists/floats.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = tensor_to_list(v)
        elif isinstance(v, dict):
            out[k] = tensors_to_lists(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [tensor_to_list(item) if isinstance(item, torch.Tensor) else item for item in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialize ``data`` to a JSON file.

    Args:
        data: Any JSON-serializable Python object.  Use
            :func:`tensors_to_lists` first if ``data`` contains tensors.
        path: Destination file path.  The parent directory must already exist.
        indent: JSON indentation level.  Defaults to 2 for human-readable
            output.

    Raises:
        TypeError: If ``data`` contains non-serializable objects.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file and return the parsed object.

    Args:
        path: Source file path.

    Returns:
        Parsed Python object (dict, list, str, etc.).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(
    rows: list[dict[str, Any]],
    path: Union[str, Path],
    fieldnames: list[str] | None = None,
    ) -> None:
    """Write a list of dicts to a CSV file.

    Args:
        rows: List of row dicts.  All dicts should share the same keys.
        path: Destination file path.  Parent directory must exist.
        fieldnames: Column ordering.  If ``None``, inferred from the first
            row's keys in insertion order.

    Raises:
        ValueError: If ``rows`` is empty and ``fieldnames`` is ``None``.
    """
    path = Path(path)
    if not rows and fieldnames is None:
        raise ValueError("Cannot write an empty CSV without explicit fieldnames.")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Union[str, Path]) -> list[dict[str, str]]:
    """Load a CSV file and return a list of row dicts.

    All values are returned as strings; callers are responsible for casting.

    Args:
        path: Source file path.

    Returns:
        List of ``{column: value}`` dicts.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Serialize an arbitrary Python object to a Pickle file.

    Use this only for heavy intermediate objects (e.g., fitted GP models,
    large tensors) that cannot be represented in JSON.  Do not use Pickle as
    the primary storage format for results — prefer JSON for portability.

    Args:
        obj: Any picklable Python object.
        path: Destination file path.  Parent directory must exist.
    """
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a Pickle file.

    Args:
        path: Source file path.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
