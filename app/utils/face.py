"""Face detection and preservation validation utilities."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FaceBox:
    """Detected face bounding box."""

    x: int
    y: int
    width: int
    height: int
    confidence: float


class FaceDetector:
    """OpenCV DNN-based face detector for pre/post processing validation."""

    # Model files (will be downloaded on first use if not present)
    MODEL_FILE = "data/opencv_face_detector_uint8.pb"
    CONFIG_FILE = "data/opencv_face_detector.pbtxt"
    MODEL_URL = (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    CONFIG_URL = (
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/"
        "opencv_face_detector.pbtxt"
    )

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold
        self._net: Optional[cv2.dnn.Net] = None

    def _load_model(self) -> cv2.dnn.Net:
        """Lazy-load the DNN face detection model."""
        if self._net is not None:
            return self._net

        import os
        from pathlib import Path

        model_path = Path(self.MODEL_FILE)
        config_path = Path(self.CONFIG_FILE)

        # Fallback to OpenCV's built-in samples if files not present
        if not model_path.exists() or not config_path.exists():
            # Try to use OpenCV's sample paths or raise informative error
            raise FileNotFoundError(
                f"Face detection model files not found. "
                f"Please download them to {model_path.parent}/"
            )

        self._net = cv2.dnn.readNetFromTensorflow(str(model_path), str(config_path))
        return self._net

    def detect(self, image: Image.Image) -> List[FaceBox]:
        """Detect faces in a PIL Image.

        Args:
            image: Input PIL Image.

        Returns:
            List of detected face bounding boxes.
        """
        try:
            net = self._load_model()
        except Exception as exc:
            logger.warning(f"Face detection model unavailable: {exc}")
            return []

        # Convert PIL to OpenCV format (RGB -> BGR)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False
        )
        net.setInput(blob)
        detections = net.forward()

        faces: List[FaceBox] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append(
                    FaceBox(
                        x=max(0, x1),
                        y=max(0, y1),
                        width=x2 - x1,
                        height=y2 - y1,
                        confidence=confidence,
                    )
                )
        return faces

    def has_faces(self, image: Image.Image) -> bool:
        """Quick check if any faces are present in the image."""
        return len(self.detect(image)) > 0

    def get_face_regions(
        self, image: Image.Image, padding: float = 0.2
    ) -> List[Tuple[int, int, int, int]]:
        """Get face regions with optional padding as (x, y, w, h).

        Args:
            image: Input PIL Image.
            padding: Fraction of face size to pad around each face.

        Returns:
            List of (x, y, width, height) tuples.
        """
        faces = self.detect(image)
        regions = []
        img_w, img_h = image.size

        for face in faces:
            pad_x = int(face.width * padding)
            pad_y = int(face.height * padding)
            x = max(0, face.x - pad_x)
            y = max(0, face.y - pad_y)
            w = min(img_w - x, face.width + 2 * pad_x)
            h = min(img_h - y, face.height + 2 * pad_y)
            regions.append((x, y, w, h))

        return regions

    def compare_face_presence(
        self, before: Image.Image, after: Image.Image
    ) -> Tuple[bool, str]:
        """Compare face detection before and after processing.

        Args:
            before: Original image.
            after: Processed image.

        Returns:
            Tuple of (passed, message).
        """
        before_faces = self.detect(before)
        after_faces = self.detect(after)

        if not before_faces and not after_faces:
            return True, "No faces detected in either image."

        if before_faces and not after_faces:
            return False, "WARNING: Faces were lost during processing!"

        if len(before_faces) != len(after_faces):
            return (
                False,
                f"WARNING: Face count changed from {len(before_faces)} to {len(after_faces)}.",
            )

        return True, f"All {len(before_faces)} face(s) preserved."
