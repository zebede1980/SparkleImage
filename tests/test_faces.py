"""Face detection and the crop/paste-back mechanics."""

import pytest
from PIL import Image

from app.core.faces import FaceBox, FaceDetector, crop_face, paste_face
from tests.conftest import make_photo


class TestFaceBox:
    def test_padding_grows_the_box(self):
        box = FaceBox(100, 100, 200, 200).padded((1000, 1000), 0.5)
        assert box == (0, 0, 400, 400)

    def test_padding_is_clipped_to_the_image(self):
        box = FaceBox(10, 10, 100, 100).padded((150, 150), 1.0)
        assert box == (0, 0, 150, 150)

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


class TestCropAndPaste:
    def test_crop_returns_the_padded_region(self):
        photo = make_photo((400, 400))
        patch, box = crop_face(photo, FaceBox(100, 100, 100, 100), padding=0.5)
        assert box == (50, 50, 250, 250)
        assert patch.size == (200, 200)

    def test_paste_puts_the_patch_back_where_it_came_from(self):
        canvas = Image.new("RGB", (300, 300), (255, 0, 0))
        patch = Image.new("RGB", (100, 100), (0, 0, 255))
        result = paste_face(canvas, patch, (100, 100, 200, 200), feather=0)
        assert result.getpixel((150, 150)) == (0, 0, 255)  # inside the patch
        assert result.getpixel((10, 10)) == (255, 0, 0)    # outside it

    def test_a_patch_at_the_wrong_size_is_resized_to_the_box(self):
        canvas = Image.new("RGB", (300, 300), (255, 0, 0))
        patch = Image.new("RGB", (512, 512), (0, 0, 255))
        result = paste_face(canvas, patch, (100, 100, 200, 200), feather=0)
        assert result.size == (300, 300)
        assert result.getpixel((150, 150)) == (0, 0, 255)

    def test_feathering_softens_the_edge_without_moving_it(self):
        canvas = Image.new("RGB", (300, 300), (255, 0, 0))
        patch = Image.new("RGB", (100, 100), (0, 0, 255))
        result = paste_face(canvas, patch, (100, 100, 200, 200), feather=10)
        # Centre fully replaced, far outside untouched, edge a blend of the two.
        assert result.getpixel((150, 150)) == (0, 0, 255)
        assert result.getpixel((5, 5)) == (255, 0, 0)
        edge = result.getpixel((101, 150))
        assert 0 < edge[0] < 255 and 0 < edge[2] < 255
