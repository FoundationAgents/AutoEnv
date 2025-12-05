"""Image utilities."""

import base64
from pathlib import Path


def save_base64_image(img_b64: str, path: Path) -> None:
    """Save a base64-encoded image to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bytes = base64.b64decode(img_b64)
    path.write_bytes(img_bytes)

