"""Face detection, with the landmarks identity scoring needs.

The previous implementation counted faces before and after an edit and called
that "face preservation". Counting says nothing about whether they are the same
people; `app.core.identity` measures that. This module only finds faces — and
their five landmarks (eyes, nose, mouth corners), which is what lets ArcFace
align a face before embedding it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# YuNet is small (~340KB), far better than a Haar cascade, and fetched at image
# build time. When it is absent — a bare checkout, a dev box — detection falls
# back to the cascade that ships inside opencv-python itself, so there is no
# configuration under which the detector silently finds nothing. The cascade
# gives no landmarks, so identity scoring is unavailable on that path.
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
# /app/models is where the image bakes it in. Not /app/data/models: compose
# bind-mounts ./data over /app/data, which hides anything the image put there.
YUNET_SEARCH_PATHS = (
    Path(os.environ.get("SPARKLE_YUNET_PATH", "")),
    Path("data/models") / YUNET_FILENAME,
    Path("/app/models") / YUNET_FILENAME,
    Path("/app/data/models") / YUNET_FILENAME,
)

# YuNet works on the image at whatever size it's given, and no one size suits
# every photo. Faces in a group shot are often under 60px — too small unless
# enlarged — yet on a 1930s wedding print YuNet found 8 faces at 1200px, 6 at
# 1600px and none at 2400px: at high resolution the paper's grain swamps it.
# So detection runs at several sizes and the results are merged.
DETECT_LONG_EDGES = (1200, 1800, 2400)
DETECT_LONG_EDGE = DETECT_LONG_EDGES[-1]
MAX_UPSCALE = 4.0
SAME_FACE_IOU = 0.3

Landmarks = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FaceBox:
    """A detected face, in pixels of the image it was found in."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    landmarks: Optional[Landmarks] = None

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def scaled(self, factor_x: float, factor_y: float) -> "FaceBox":
        """This face mapped into an image of a different size."""
        marks = tuple((px * factor_x, py * factor_y) for px, py in self.landmarks) if self.landmarks else None
        return FaceBox(
            int(round(self.x * factor_x)),
            int(round(self.y * factor_y)),
            max(1, int(round(self.width * factor_x))),
            max(1, int(round(self.height * factor_y))),
            self.confidence,
            marks,
        )


def _iou(a: FaceBox, b: FaceBox) -> float:
    left, top = max(a.x, b.x), max(a.y, b.y)
    right, bottom = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    overlap = max(0, right - left) * max(0, bottom - top)
    return overlap / float(a.area + b.area - overlap) if overlap else 0.0


def merge_duplicates(faces: list[FaceBox]) -> list[FaceBox]:
    """One box per face: where detections overlap, keep the most confident."""
    kept: list[FaceBox] = []
    for face in sorted(faces, key=lambda f: f.confidence, reverse=True):
        if all(_iou(face, k) < SAME_FACE_IOU for k in kept):
            kept.append(face)
    return kept


def _find_yunet() -> Optional[Path]:
    for candidate in YUNET_SEARCH_PATHS:
        if candidate and candidate.name and candidate.exists():
            return candidate
    return None


class FaceDetector:
    """Detects faces with YuNet when available, else a bundled Haar cascade."""

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold
        self._yunet_path = _find_yunet()
        self._cascade: Optional[cv2.CascadeClassifier] = None

    @property
    def backend(self) -> str:
        return "yunet" if self._yunet_path else "haar"

    @property
    def gives_landmarks(self) -> bool:
        return self._yunet_path is not None

    def _detect_yunet(self, bgr: np.ndarray) -> list[FaceBox]:
        height, width = bgr.shape[:2]
        detector = cv2.FaceDetectorYN.create(
            str(self._yunet_path), "", (width, height), self.confidence_threshold, 0.3, 5000
        )
        _, faces = detector.detect(bgr)
        results: list[FaceBox] = []
        for face in faces if faces is not None else []:
            x, y, w, h = (int(round(v)) for v in face[:4])
            marks = tuple((float(face[4 + 2 * i]), float(face[5 + 2 * i])) for i in range(5))
            results.append(FaceBox(max(0, x), max(0, y), max(1, w), max(1, h), float(face[14]), marks))
        return results

    def _detect_cascade(self, bgr: np.ndarray) -> list[FaceBox]:
        if self._cascade is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if not cascade_path.exists():  # pragma: no cover - packaging failure
                raise FileNotFoundError(
                    f"OpenCV's bundled cascades are missing at {cascade_path}. "
                    "Pin opencv-python-headless>=4.9,<5, which ships them."
                )
            self._cascade = cv2.CascadeClassifier(str(cascade_path))
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)
        found = self._cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [FaceBox(int(x), int(y), int(w), int(h), 1.0) for x, y, w, h in found]

    def _detect_at(self, rgb: Image.Image, scale: float) -> list[FaceBox]:
        if abs(scale - 1.0) > 0.05:
            resample = Image.BICUBIC if scale > 1 else Image.LANCZOS
            rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), resample)
        else:
            scale = 1.0
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        faces = self._detect_yunet(bgr) if self._yunet_path else self._detect_cascade(bgr)
        return [f.scaled(1 / scale, 1 / scale) for f in faces] if scale != 1.0 else faces

    def _detect_array(self, image: Image.Image) -> list[FaceBox]:
        rgb = image.convert("RGB")
        scales = sorted({round(min(MAX_UPSCALE, edge / max(rgb.size)), 3) for edge in DETECT_LONG_EDGES})
        found: list[FaceBox] = []
        for scale in scales:
            found.extend(self._detect_at(rgb, scale))
        return merge_duplicates(found)

    def detect(self, image: Image.Image) -> list[FaceBox]:
        """Find faces, largest first.

        Faded sepia scans — the photographs this app exists for — sit in a
        narrow band of the histogram and a detector will walk straight past a
        face that is plainly visible to a person. So a first pass that finds
        nothing is retried on a contrast-normalised copy; the boxes map back
        unchanged because normalising does not move any pixel.
        """
        faces = self._detect_array(image)
        if not faces:
            normalised = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
            faces = self._detect_array(normalised)
            if faces:
                logger.info("Faces found only after contrast normalisation (%d)", len(faces))
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def has_faces(self, image: Image.Image) -> bool:
        return bool(self.detect(image))
