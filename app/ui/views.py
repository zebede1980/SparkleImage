"""What the UI shows for a job: labels, previews, face comparison strips.

Kept out of app.py so it can be tested without Gradio. Anything image-shaped is
cached as a JPEG in the job's folder: a 20MP scan is far too much to push to a
phone on every refresh.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from app.core.jobs import Job, JobStore
from app.core.restore import RELIABLE_LIKENESS

STATUS_MARK = {"queued": "⏳", "running": "⚙️", "done": "✅", "failed": "⚠️"}
MODE_LABEL = {"colourise": "colourised", "correct": "colour corrected", "keep_bw": "kept black & white"}
PREVIEW_EDGE = 1600
FACE_TILE = 220


def job_label(job: Job) -> str:
    when = time.strftime("%d %b %H:%M", time.localtime(job.created))
    return f"{STATUS_MARK.get(job.status, '')} {job.photo} — {job.stage if job.status != 'done' else 'done'} · {when}"


def job_summary(job: Job) -> str:
    lines = [f"### {job.photo}", f"**{STATUS_MARK.get(job.status, '')} {job.stage}**"]
    if job.error:
        lines.append(f"⚠️ {job.error}")
    if job.mode:
        lines.append(f"{job.size[0]}×{job.size[1]}, {MODE_LABEL.get(job.mode, job.mode)}"
                     + (f", {job.options.era}" if job.options.era else ""))
    if job.candidates:
        base = next((c for c in job.candidates if c["id"] == job.base), None)
        if job.faces:
            lines.append(f"{len(job.faces)} face(s) found. Base: **{job.base}**"
                         + (f" (likeness {base['mean']:.2f})" if base and base.get("mean") is not None else "")
                         + (f"; {len(job.face_choices)} face(s) taken from other candidates" if job.face_choices else ""))
            faint = [f["index"] + 1 for f in job.faces if (_best(job, f["index"]) or 0) < RELIABLE_LIKENESS]
            if faint:
                lines.append(
                    f"👀 Face(s) {', '.join(map(str, faint))} have too little detail in the original to measure "
                    "likeness reliably — the restoration has had to rebuild them. Check those by eye under "
                    "*Candidates and faces*."
                )
        else:
            lines.append(f"No faces found to compare — base: **{job.base}**")
    return "\n\n".join(lines)


def _best(job: Job, index: int) -> Optional[float]:
    scores = [c["scores"][index] for c in job.candidates
              if index < len(c.get("scores", [])) and c["scores"][index] is not None]
    return max(scores) if scores else None


def preview(path: Optional[Path], edge: int = PREVIEW_EDGE) -> Optional[str]:
    """A cached, phone-sized JPEG of `path` (regenerated if the original is newer)."""
    if path is None or not path.exists():
        return None
    target = path.with_name(f"preview_{path.stem}.jpg")
    if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((edge, edge), Image.LANCZOS)
            image.save(target, quality=88)
    return str(target)


def source_preview(store: JobStore, job: Job) -> Optional[str]:
    path = next(store.dir(job.id).glob("source.*"), None)
    return preview(path)


def candidate_gallery(store: JobStore, job: Job) -> list[tuple[str, str]]:
    items = []
    for c in job.candidates:
        path = store.image(job, f"cand_{c['id']}.png")
        if not path:
            continue
        caption = c["id"] + (f" · likeness {c['mean']:.2f}" if c.get("mean") is not None else "")
        if c["id"] == job.base:
            caption += " · BASE"
        items.append((preview(path, 900), caption))
    return items


def face_options(job: Job, index: int) -> list[tuple[str, str]]:
    """Dropdown choices for one face: keep the base's, or take it from a candidate."""
    base = next((c for c in job.candidates if c["id"] == job.base), None)
    base_score = base["scores"][index] if base and index < len(base["scores"]) else None
    options = [(f"Base's face ({job.base}{f', {base_score:.2f}' if base_score is not None else ''})", "")]
    for c in job.candidates:
        if c["id"] == job.base or index >= len(c.get("scores", [])) or c["scores"][index] is None:
            continue
        options.append((f"{c['id']} ({c['scores'][index]:.2f})", c["id"]))
    return options


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def face_strip(store: JobStore, job: Job, index: int) -> Optional[str]:
    """Source face, then each candidate's version of it, labelled with its likeness score."""
    face = next((f for f in job.faces if f["index"] == index), None)
    if face is None:
        return None
    target = store.dir(job.id) / f"facestrip_{index}.jpg"
    if target.exists():
        return str(target)

    fw, fh = face["frame"]
    x, y, w, h = face["box"]
    pad_x, pad_y = w * 0.35, h * 0.35

    def crop(image: Image.Image) -> Image.Image:
        sx, sy = image.width / fw, image.height / fh
        box = (int((x - pad_x) * sx), int((y - pad_y) * sy), int((x + w + pad_x) * sx), int((y + h + pad_y) * sy))
        return image.crop(box).resize((FACE_TILE, int(FACE_TILE * (h + 2 * pad_y) / (w + 2 * pad_x))), Image.LANCZOS)

    tiles = [(crop(store.source(job)), "original")]
    for c in job.candidates:
        path = store.image(job, f"cand_{c['id']}.png")
        if not path:
            continue
        score = c["scores"][index] if index < len(c.get("scores", [])) else None
        with Image.open(path) as image:
            tiles.append((crop(image.convert("RGB")), f"{c['id']} {score:.2f}" if score is not None else f"{c['id']} —"))

    label_h = 26
    tile_h = tiles[0][0].height
    strip = Image.new("RGB", (len(tiles) * (FACE_TILE + 6), tile_h + label_h), "white")
    draw = ImageDraw.Draw(strip)
    font = _font(15)
    for i, (tile, label) in enumerate(tiles):
        strip.paste(tile, (i * (FACE_TILE + 6), 0))
        draw.text((i * (FACE_TILE + 6) + 4, tile_h + 4), label, fill="black", font=font)
    strip.save(target, quality=88)
    return str(target)
