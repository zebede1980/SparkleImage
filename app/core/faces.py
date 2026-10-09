"""Face detection, and the crop-edit-paste-back that actually preserves identity.

The previous implementation counted faces before and after an edit and called
that "face preservation". It never ran (the weights it looked for were in no
image and no code downloaded them) and would not have meant anything if it had:
two faces before and two after says nothing about whether they are the same two
people.

What genuinely helps is resolution. A face 200px across in a 4000px scan is a
handful of pixels by the time a model has worked on the whole frame, and that is
where identity goes. So faces are cropped, edited at full size in their own
right, and composited back.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# YuNet is small (~340KB), far better than a Haar cascade, and fetched at image
# build time. When it is absent — a bare checkout, a dev box — detection falls
# back to the cascade that ships inside opencv-python itself, so there is no
# configuration under which the detector silently finds nothing.
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_SEARCH_PATHS = (
    Path(os.environ.get("SPARKLE_YUNET_PATH", "")),
    Path("data/models") / YUNET_FILENAME,
    Path("/app/data/models") / YUNET_FILENAME,
)


@dataclass(frozen=True)
class FaceBox:
    """A detected face, in pixels."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def area(self) -> int:
        return self.width * self.height

    def padded(self, image_size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
        """This face's box grown by `padding`, clipped to the image."""
        img_w, img_h = image_size
        pad_x = int(self.width * padding)
        pad_y = int(self.height * padding)
        left = max(0, self.x - pad_x)
        top = max(0, self.y - pad_y)
        right = min(img_w, self.x + self.width + pad_x)
        bottom = min(img_h, self.y + self.height + pad_y)
        return left, top, right, bottom


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

    def _detect_yunet(self, bgr: np.ndarray) -> list[FaceBox]:
        height, width = bgr.shape[:2]
        detector = cv2.FaceDetectorYN.create(
            str(self._yunet_path), "", (width, height), self.confidence_threshold
        )
        detector.setInputSize((width, height))
        _, faces = detector.detect(bgr)
        results: list[FaceBox] = []
        for face in faces if faces is not None else []:
            x, y, w, h = (int(round(v)) for v in face[:4])
            results.append(
                FaceBox(max(0, x), max(0, y), max(1, w), max(1, h), float(face[-1]))
            )
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

    def _detect_array(self, image: Image.Image) -> list[FaceBox]:
        array = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return self._detect_yunet(bgr) if self._yunet_path else self._detect_cascade(bgr)

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
            from PIL import ImageOps

            normalised = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
            faces = self._detect_array(normalised)
            if faces:
                logger.info("Faces found only after contrast normalisation (%d)", len(faces))
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def has_faces(self, image: Image.Image) -> bool:
        return bool(self.detect(image))


def crop_face(
    image: Image.Image, face: FaceBox, padding: float
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop a padded region around `face`, returning the crop and its box."""
    box = face.padded(image.size, padding)
    return image.crop(box), box


def paste_face(
    canvas: Image.Image,
    patch: Image.Image,
    box: tuple[int, int, int, int],
    feather: int = 8,
) -> Image.Image:
    """Composite an edited face patch back into `canvas` with a soft edge."""
    from PIL import ImageDraw

    from app.core.geometry import feather_mask

    left, top, right, bottom = box
    target_size = (right - left, bottom - top)
    if patch.size != target_size:
        patch = patch.resize(target_size, Image.LANCZOS)

    # A rectangle inset by the feather radius, blurred outwards, so the patch
    # fades into the surrounding photograph instead of showing its edges.
    mask = Image.new("L", target_size, 0)
    inset = min(feather, target_size[0] // 3, target_size[1] // 3)
    ImageDraw.Draw(mask).rectangle(
        [inset, inset, target_size[0] - inset - 1, target_size[1] - inset - 1], fill=255
    )
    if feather > 0:
        mask = feather_mask(mask, feather)

    result = canvas.convert("RGB").copy()
    result.paste(patch.convert("RGB"), (left, top), mask)
    return result
