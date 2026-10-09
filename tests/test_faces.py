"""Face detection and the crop/paste-back mechanics."""

from pathlib import Path

import pytest
from PIL import Image

from app.core.faces import DETECT_LONG_EDGES, MAX_UPSCALE, FaceBox, FaceDetector, merge_duplicates


def test_merge_keeps_the_most_confident_of_overlapping_boxes_and_all_separate_ones():
    a = FaceBox(100, 100, 50, 50, 0.7)
    a_better = FaceBox(105, 102, 50, 50, 0.9)
    b = FaceBox(400, 100, 50, 50, 0.6)
    assert merge_duplicates([a, b, a_better]) == [a_better, b]
from tests.conftest import make_photo


class TestFaceBox:
    def test_largest_face_sorts_first(self):
        detector = FaceDetector()
        small, large = FaceBox(0, 0, 10, 10), FaceBox(0, 0, 100, 100)
        assert sorted([small, large], key=lambda f: f.area, reverse=True)[0] is large


class TestFadedScanRetry:
    """A faded sepia scan can hide a face from the detector until it is normalised."""

    class OnlyAfterNormalising(FaceDetector):
        def __init__(self):
            super().__init__()
            self.attempts = []

        def _detect_array(self, image):
            # Stand in for a detector that needs contrast: report a face only
            # once the histogram has been stretched wide enough.
            low, high = image.convert("L").getextrema()
            self.attempts.append(high - low)
            return [FaceBox(10, 10, 40, 40, 0.9)] if (high - low) > 50 else []

    def test_a_first_pass_miss_is_retried_on_a_normalised_copy(self):
        faded = make_photo((200, 200)).point(lambda v: 110 + v // 8)
        detector = self.OnlyAfterNormalising()
        assert detector.detect(faded)
        assert len(detector.attempts) == 2  # original, then normalised

    def test_a_first_pass_hit_is_not_retried(self):
        detector = self.OnlyAfterNormalising()
        assert detector.detect(make_photo((200, 200)))
        assert len(detector.attempts) == 1

    def test_no_face_anywhere_returns_empty(self):
        class Never(FaceDetector):
            def _detect_array(self, image):
                return []

        assert Never().detect(make_photo((200, 200))) == []


class TestScaling:
    def test_box_and_landmarks_scale_together(self):
        face = FaceBox(100, 50, 40, 60, 0.9, ((110.0, 70.0), (130.0, 70.0), (120.0, 80.0), (112.0, 95.0), (128.0, 95.0)))
        half = face.scaled(0.5, 0.5)
        assert (half.x, half.y, half.width, half.height) == (50, 25, 20, 30)
        assert half.landmarks[0] == (55.0, 35.0)
        assert half.confidence == 0.9

    def test_centre(self):
        assert FaceBox(10, 20, 30, 40).centre == (25.0, 40.0)


class TestDetectionSize:
    """Detection runs on a copy near DETECT_LONG_EDGE; results come back in source pixels."""

    class Recorder(FaceDetector):
        def __init__(self):
            super().__init__()
            self._yunet_path = Path("fake.onnx")  # force the YuNet branch
            self.seen = []

        def _detect_yunet(self, bgr):
            height, width = bgr.shape[:2]
            self.seen.append((width, height))
            # One face filling the middle quarter of whatever it was shown.
            return [FaceBox(width // 4, height // 4, width // 2, height // 2, 0.9,
                            ((width / 2, height / 2),) * 5)]

    @pytest.mark.parametrize("size", [(300, 200), (6000, 4000)])
    def test_detection_runs_at_several_sizes_and_merges_to_one_box_per_face(self, size):
        detector = self.Recorder()
        faces = detector.detect(Image.new("RGB", size, (128, 128, 128)))
        expected = sorted({round(max(size) * min(MAX_UPSCALE, edge / max(size))) for edge in DETECT_LONG_EDGES})
        assert sorted(max(s) for s in detector.seen) == pytest.approx(expected, abs=2)
        assert len(faces) == 1  # the same face found at every size is one face
        face = faces[0]
        assert face.x == pytest.approx(size[0] // 4, abs=4)
        assert face.width == pytest.approx(size[0] // 2, abs=4)
        assert face.landmarks[0][0] == pytest.approx(size[0] / 2, abs=4)
