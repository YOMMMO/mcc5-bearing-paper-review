"""Small YAML configuration helpers for CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict when unavailable."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def config_default(config: dict[str, Any], key: str, default: Any) -> Any:
    """Return a config value, treating missing or null as the script default."""
    value = config.get(key, default)
    return default if value is None else value
