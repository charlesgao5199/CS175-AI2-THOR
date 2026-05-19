"""Small configuration helpers.

The project will likely move to a fuller config system later. For now, this
keeps the baseline scripts runnable without adding another dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a flat key-value YAML file.

    Supported syntax is intentionally small: comments, blank lines, and
    ``key: value`` pairs with scalar values.
    """

    config: Dict[str, Any] = {}
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line {line_number}: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing config key on line {line_number}")
        config[key] = _parse_scalar(value)
    return config
