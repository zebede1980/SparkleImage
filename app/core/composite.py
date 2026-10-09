"""Pixel work that needs no model: colour matching, face pasting, chroma transfer.

All of it runs on CPU in well under a second, so the UI can re-composite
instantly when someone overrides which version of a face to use.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.core.faces import FaceBox

# The 95th percentile of chroma left after removing an image's overall tint.
# Toned and yellowed black-and-white prints measured 4.3-10.8 (worst: a sepia
# print on cream card); a faded colour slide measured 16.1. The UI lets the
# user override the guess, because one colour example is a thin margin.
MONOCHROME_CHROMA_P95 = 12.0


def _lab(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)


def _rgb(lab: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def looks_monochrome(image: Image.Image) -> bool:
    """Whether a photo is black and white — including sepia, toned or yellowed prints.

    A uniform tint (sepia, yellowing) isn't colour, so it is subtracted first;
    what's left is how much the colour actually varies across the picture.
    """
    small = image.convert("RGB")
    small.thumbnail((512, 512))
    lab = _lab(small)
    a, b = lab[..., 1], lab[..., 2]
    residual = np.hypot(a - a.mean(), b - b.mean())
    return float(np.percentile(residual, 95)) < MONOCHROME_CHROMA_P95


def ellipse_mask(size: tuple[int, int], face: FaceBox) -> np.ndarray:
    """A soft-edged ellipse over a face (forehead to chin, ear to ear), values 0..1."""
    width, height = size
    mask = np.zeros((height, width), np.float32)
    cx, cy = face.centre
    cv2.ellipse(mask, (int(cx), int(cy)), (int(face.width * 0.62), int(face.height * 0.68)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, face.width * 0.08))


def match_colour(patch: Image.Image, reference: Image.Image, mask: np.ndarray) -> Image.Image:
    """Shift `patch`'s Lab mean and spread to `reference`'s inside `mask`.

    Different seeds grade a photo differently — one a touch warmer, another
    more contrasty — so a face taken from one and pasted into another needs
    its colour brought in line or it reads as a patch.
    """
    p, r = _lab(patch), _lab(reference)
    inside = mask > 0.5
    if not inside.any():
        return patch
    for channel in range(3):
        ps, rs = p[..., channel][inside], r[..., channel][inside]
        p[..., channel] = (p[..., channel] - ps.mean()) * (rs.std() / (ps.std() + 1e-6)) + rs.mean()
    return _rgb(p)


def paste_face(base: Image.Image, donor: Image.Image, face: FaceBox) -> Image.Image:
    """`base` with the face at `face` taken from `donor`, colour-matched and feathered.

    `donor` is resized to `base` if needed; both are expected to be restorations
    of the same photo, so the face is in the same place in each.
    """
    base = base.convert("RGB")
    donor = donor.convert("RGB")
    if donor.size != base.size:
        donor = donor.resize(base.size, Image.LANCZOS)
    mask = ellipse_mask(base.size, face)
    donor = match_colour(donor, base, mask)
    blended = np.asarray(donor, np.float32) * mask[..., None] + np.asarray(base, np.float32) * (1 - mask[..., None])
    return Image.fromarray(blended.astype(np.uint8))


def chroma_transfer(scan: Image.Image, colour: Image.Image) -> Image.Image:
    """The scan's own brightness at full resolution, with colour from `colour`.

    Everything a person sees as detail and likeness lives in luminance, so this
    keeps every pixel of a good scan's real detail and only borrows colour from
    the restoration (which can be far lower resolution — colour is smooth).
    The trade-off: damage that is visible in the scan's brightness — creases,
    scratches, paper texture — stays too.
    """
    scan = scan.convert("RGB")
    lab = _lab(scan)
    coloured = _lab(colour.convert("RGB").resize(scan.size, Image.LANCZOS))
    lab[..., 1:] = coloured[..., 1:]
    return _rgb(lab)
