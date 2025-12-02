"""
Shared utility helpers for the visualizer pipeline.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: Path) -> Path:
    """Create a directory path if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Dict[str, Any]) -> Path:
    """Persist a dict as JSON using UTF-8 encoding."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON if the file exists; return None otherwise."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, content: str) -> Path:
    """Save plain text content to a file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
