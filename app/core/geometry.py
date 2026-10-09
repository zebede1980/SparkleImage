"""Output geometry: choosing a size the model accepts that keeps the photo's shape.

The single biggest failure of the DALL-E-era code was padding every photograph
into a 1024x1024 square (see plans/revival-2026.md 2.2). Restoration only means
anything if the result has the same framing as the original, so every decision
here is driven by the *source* aspect ratio and the model's declared limits,
never by a fixed default.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from PIL import Image


@dataclass(frozen=True)
class CustomResolution:
    """A model's free-form width/height range (nano-gpt `customResolution`)."""

    min_width: int
    min_height: int
    max_width: int
    max_height: int
    step: int = 1
    min_total_pixels: int = 0

    @classmethod
    def from_metadata(cls, data: dict) -> Optional["CustomResolution"]:
        if not data or not data.get("enabled"):
            return None
        return cls(
            min_width=int(data.get("minWidth", 1)),
            min_height=int(data.get("minHeight", 1)),
            max_width=int(data.get("maxWidth", 1 << 16)),
            max_height=int(data.get("maxHeight", 1 << 16)),
            step=max(1, int(data.get("step", 1))),
            min_total_pixels=int(data.get("minTotalPixels", 0)),
        )


def _round_to_step(value: int, step: int, low: int, high: int) -> int:
    """Snap `value` onto the model's step grid, clamped into [low, high]."""
    if step <= 1:
        return max(low, min(high, int(value)))
    snapped = int(round(value / step)) * step
    # Clamp *after* snapping, then re-snap inwards so the bounds are respected
    # even when they are not themselves multiples of step.
    if snapped < low:
        snapped = int(math.ceil(low / step)) * step
    if snapped > high:
        snapped = int(math.floor(high / step)) * step
    return max(1, snapped)


def _aspect_error(width: int, height: int, aspect: float) -> float:
    """Relative distance between a candidate's aspect ratio and the target."""
    if height <= 0:
        return float("inf")
    return abs((width / height) - aspect) / aspect


def choose_custom_size(
    source_size: tuple[int, int],
    limits: CustomResolution,
) -> tuple[int, int]:
    """Pick a width/height on the model's grid that matches the source's shape.

    Seedream's `minTotalPixels` floor (3,686,400 on v4.5) is enforced *silently*
    by the provider: ask for less and it returns a square at its own default
    size rather than an error, which is exactly the bug this function exists to
    prevent. So the area is raised to the floor before anything else, and the
    step-rounding is checked afterwards in case it rounded back under.
    """
    src_w, src_h = source_size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source image has no area")
    aspect = src_w / src_h

    # Seeding the search at or above the floor keeps the first guess close to
    # the source's shape; the growth loop at the end is what actually
    # *guarantees* the floor is met once rounding has had its say.
    max_area = limits.max_width * limits.max_height
    target_area = min(max(src_w * src_h, limits.min_total_pixels), max_area)

    ideal_w = math.sqrt(target_area * aspect)
    ideal_h = math.sqrt(target_area / aspect)

    # Fit inside the model's box by *scaling both axes together* — clamping one
    # axis on its own is what turns a 3:2 scan into a 1.23:1 letterbox.
    shrink = min(1.0, limits.max_width / ideal_w, limits.max_height / ideal_h)
    ideal_w, ideal_h = ideal_w * shrink, ideal_h * shrink
    grow = max(1.0, limits.min_width / ideal_w, limits.min_height / ideal_h)
    ideal_w, ideal_h = ideal_w * grow, ideal_h * grow

    # Only an aspect ratio the box cannot represent at all (a panorama wider
    # than max_width:min_height) reaches this clamp, and it necessarily loses
    # some of the shape — there is no size that keeps it.
    width = _round_to_step(ideal_w, limits.step, limits.min_width, limits.max_width)
    height = _round_to_step(ideal_h, limits.step, limits.min_height, limits.max_height)

    # Snapping down to the grid can drop the area back under the floor. Grow
    # along whichever axis keeps the aspect ratio closest, one step at a time,
    # until the floor is met or both axes are maxed out.
    while width * height < limits.min_total_pixels:
        grow_w = width + limits.step <= limits.max_width
        grow_h = height + limits.step <= limits.max_height
        if not grow_w and not grow_h:
            break
        if grow_w and grow_h:
            if _aspect_error(width + limits.step, height, aspect) <= _aspect_error(
                width, height + limits.step, aspect
            ):
                width += limits.step
            else:
                height += limits.step
        elif grow_w:
            width += limits.step
        else:
            height += limits.step

    return width, height


def choose_preset_size(
    source_size: tuple[int, int],
    presets: Iterable[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    """Pick the listed resolution whose aspect ratio is closest to the source.

    Ties (a model listing both 2048x1536 and 4096x3072, say) go to the larger
    one: a restored scan is downsampled back to the original dimensions at the
    end, so more pixels from the model is never worse.
    """
    src_w, src_h = source_size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source image has no area")
    aspect = src_w / src_h

    best: Optional[tuple[float, int, tuple[int, int]]] = None
    for width, height in presets:
        if width <= 0 or height <= 0:
            continue
        key = (round(_aspect_error(width, height, aspect), 4), -(width * height))
        if best is None or key < (best[0], best[1]):
            best = (key[0], key[1], (width, height))
    return best[2] if best else None


def encode_for_upload(
    image: Image.Image,
    max_bytes: Optional[int] = None,
    prefer: str = "PNG",
) -> tuple[bytes, str]:
    """Encode an image for upload, staying under the model's input size limit.

    Seedream v4.5 caps input at 10MB, which a large PNG scan blows through
    easily. PNG is tried first (lossless, no generation loss on the source we
    are trying to restore); failing that a JPEG quality ladder, and only if
    even quality 60 is too big do we downscale. The caller keeps the original
    at full resolution regardless — this only affects what goes up the wire.
    """
    rgb = image.convert("RGB") if image.mode not in ("RGB", "L") else image

    def _encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=fmt, **kwargs)
        return buf.getvalue()

    if prefer.upper() == "PNG":
        data = _encode(rgb, "PNG", optimize=True)
        if max_bytes is None or len(data) <= max_bytes:
            return data, "image/png"

    for quality in (95, 90, 85, 80, 70, 60):
        data = _encode(rgb, "JPEG", quality=quality, optimize=True)
        if max_bytes is None or len(data) <= max_bytes:
            return data, "image/jpeg"

    # Still too large at quality 60 — shrink until it fits.
    working = rgb
    for _ in range(8):
        working = working.resize(
            (max(1, working.width * 3 // 4), max(1, working.height * 3 // 4)),
            Image.LANCZOS,
        )
        data = _encode(working, "JPEG", quality=85, optimize=True)
        if max_bytes is None or len(data) <= max_bytes:
            return data, "image/jpeg"
    return data, "image/jpeg"


def restore_geometry(result: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    """Resize a model's output back to the original photo's pixel dimensions."""
    if result.size == original_size:
        return result
    return result.resize(original_size, Image.LANCZOS)


def feather_mask(mask: Image.Image, radius: int) -> Image.Image:
    """Soften a hard-edged mask so a composited edit doesn't show a seam."""
    from PIL import ImageFilter

    grey = mask.convert("L")
    if radius <= 0:
        return grey
    return grey.filter(ImageFilter.GaussianBlur(radius))


def composite_masked(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
    feather: int = 6,
) -> Image.Image:
    """Blend `edited` into `original` only where `mask` is white.

    Localised operations (scratch removal, object removal) must leave the rest
    of the photograph bit-identical — a generative model returns a whole new
    frame, so without this every "remove this lamppost" silently repaints the
    entire picture.
    """
    base = original.convert("RGB")
    top = edited.convert("RGB")
    if top.size != base.size:
        top = top.resize(base.size, Image.LANCZOS)
    alpha = feather_mask(mask, feather)
    if alpha.size != base.size:
        alpha = alpha.resize(base.size, Image.LANCZOS)
    return Image.composite(top, base, alpha)
