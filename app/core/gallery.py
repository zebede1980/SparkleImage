"""Output gallery — saving results and listing them back."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image

OUTPUT_DIR = Path("data/output")


def _timestamp() -> str:
    """A wall-clock, sortable stamp.

    The old code used `asyncio.get_event_loop().time()` — a monotonic clock with
    an arbitrary epoch — so filenames were meaningless and collided after a
    restart.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_result(
    image: Image.Image,
    operation_id: str,
    output_format: str = "png",
    jpeg_quality: int = 95,
    metadata: Optional[dict[str, Any]] = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save a result and its sidecar metadata, returning the image path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lower()
    suffix = "jpg" if fmt in ("jpg", "jpeg") else fmt
    path = output_dir / f"{_timestamp()}_{operation_id}.{suffix}"

    counter = 2
    while path.exists():
        path = output_dir / f"{_timestamp()}_{operation_id}-{counter}.{suffix}"
        counter += 1

    save_kwargs: dict[str, Any] = {}
    to_save = image
    if suffix in ("jpg", "jpeg"):
        save_kwargs["quality"] = jpeg_quality
        if to_save.mode in ("RGBA", "P"):
            to_save = to_save.convert("RGB")
    exif = image.info.get("exif")
    if exif:
        save_kwargs["exif"] = exif

    to_save.save(path, **save_kwargs)

    if metadata:
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps({"saved": _timestamp(), "operation": operation_id, **metadata}, indent=1)
        )
    return path


def recent(limit: int = 50, output_dir: Path = OUTPUT_DIR) -> list[str]:
    """Most recently saved outputs, newest first."""
    if not output_dir.exists():
        return []
    files = [
        p for p in output_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]
