"""Is it still the same person? ArcFace similarity between source and result.

Every generative restoration redraws faces. Most of the time the redrawn face
is recognisably the same person; sometimes — depending on nothing more than
the seed — a child comes back with bigger eyes and a slimmer face. Measured on
real family photos, ArcFace cosine similarity ranks those outcomes the same way
a person looking at them does, which is what makes automatic per-face selection
possible.

The embedding model is insightface's `w600k_r50.onnx` (ResNet-50 trained on
WebFace600K), run directly with onnxruntime. The `insightface` package itself
is not used: it needs a C++ toolchain to install and has no ARM64 wheels,
while all it would add here is a 20-line alignment step. Note the weights are
licensed for non-commercial use only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from app.core.faces import FaceBox, FaceDetector

logger = logging.getLogger(__name__)

ARCFACE_FILENAME = "w600k_r50.onnx"
ARCFACE_SEARCH_PATHS = (
    Path(os.environ.get("SPARKLE_ARCFACE_PATH", "")),
    Path("data/models") / ARCFACE_FILENAME,
    Path("/app/data/models") / ARCFACE_FILENAME,
)

# Where ArcFace expects the five landmarks in its 112x112 input. This is the
# template the model was trained with; aligning to it is not optional.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

# A candidate face is the same person's face if its centre is within this
# fraction of the source face's width — restorations keep composition, so
# faces don't move far.
MATCH_DISTANCE = 0.5


def _find_arcface() -> Optional[Path]:
    for candidate in ARCFACE_SEARCH_PATHS:
        if candidate and candidate.name and candidate.exists():
            return candidate
    return None


class FaceEmbedder:
    """ArcFace embeddings for aligned faces. Loads the model on first use."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._path = model_path or _find_arcface()
        self._session = None

    @property
    def available(self) -> bool:
        return self._path is not None

    def _get_session(self):
        if self._session is None:
            if self._path is None:
                raise RuntimeError(f"ArcFace model {ARCFACE_FILENAME} not found in {[str(p) for p in ARCFACE_SEARCH_PATHS]}")
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self._path), providers=["CPUExecutionProvider"])
        return self._session

    @staticmethod
    def align(rgb: np.ndarray, landmarks: Sequence[tuple[float, float]]) -> np.ndarray:
        """The 112x112 face crop ArcFace expects, rotated and scaled onto its template."""
        matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, dtype=np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
        return cv2.warpAffine(rgb, matrix, (112, 112), borderValue=0)

    def embed(self, rgb: np.ndarray, landmarks: Sequence[tuple[float, float]]) -> np.ndarray:
        """Unit-length 512-d embedding of the face at `landmarks` in an RGB array."""
        crop = self.align(rgb, landmarks).astype(np.float32)
        tensor = ((crop - 127.5) / 127.5).transpose(2, 0, 1)[None]
        session = self._get_session()
        vector = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
        return vector / np.linalg.norm(vector)


@dataclass
class SourceFace:
    """A face in the source photograph: where it is, and who it is."""

    index: int
    box: FaceBox
    embedding: np.ndarray
    frame: tuple[int, int]  # size of the image `box` is measured in


def _match(face: FaceBox, candidates: list[FaceBox]) -> Optional[FaceBox]:
    if not candidates:
        return None
    cx, cy = face.centre
    nearest = min(candidates, key=lambda c: (c.centre[0] - cx) ** 2 + (c.centre[1] - cy) ** 2)
    distance = ((nearest.centre[0] - cx) ** 2 + (nearest.centre[1] - cy) ** 2) ** 0.5
    return nearest if distance <= MATCH_DISTANCE * face.width else None


class IdentityScorer:
    """Finds the people in a source photo and scores how well a result keeps them."""

    def __init__(self, detector: Optional[FaceDetector] = None, embedder: Optional[FaceEmbedder] = None) -> None:
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()

    @property
    def available(self) -> bool:
        return self.embedder.available and self.detector.gives_landmarks

    def source_faces(self, source: Image.Image, frame: tuple[int, int]) -> list[SourceFace]:
        """Faces in `source`, mapped into a `frame`-sized image, left to right.

        Candidates come back from the model at its own resolution, so source
        faces are measured in that frame rather than the scan's. Faces without
        landmarks (cascade fallback) can't be embedded and are skipped.
        """
        if not self.available:
            return []
        resized = source.convert("RGB").resize(frame, Image.LANCZOS)
        rgb = np.asarray(resized)
        faces = sorted(self.detector.detect(resized), key=lambda f: f.x)
        return [
            SourceFace(i, face, self.embedder.embed(rgb, face.landmarks), frame)
            for i, face in enumerate(f for f in faces if f.landmarks)
        ]

    def score(self, candidate: Image.Image, faces: list[SourceFace]) -> list[Optional[float]]:
        """Cosine similarity for each source face, or None where it wasn't found.

        A face not found is usually turned away or hidden in the result; it is
        reported as None rather than 0 so it doesn't drag an average down.
        """
        if not faces:
            return []
        frame = faces[0].frame
        image = candidate.convert("RGB")
        if image.size != frame:
            image = image.resize(frame, Image.LANCZOS)
        rgb = np.asarray(image)
        found = [f for f in self.detector.detect(image) if f.landmarks]
        scores: list[Optional[float]] = []
        for face in faces:
            match = _match(face.box, found)
            scores.append(float(face.embedding @ self.embedder.embed(rgb, match.landmarks)) if match else None)
        return scores


def mean_score(scores: Sequence[Optional[float]]) -> Optional[float]:
    present = [s for s in scores if s is not None]
    return sum(present) / len(present) if present else None
