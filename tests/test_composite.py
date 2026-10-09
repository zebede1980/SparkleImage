"""Colour matching, face pasting, chroma transfer and the black-and-white guess."""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from app.core.composite import chroma_transfer, ellipse_mask, looks_monochrome, match_colour, paste_face
from app.core.faces import FaceBox
from tests.conftest import make_photo


class TestMonochrome:
    def test_a_colour_photo_is_colour(self):
        assert not looks_monochrome(make_photo((400, 300)))

    def test_greyscale_is_monochrome(self):
        assert looks_monochrome(ImageOps.grayscale(make_photo((400, 300))).convert("RGB"))

    def test_a_sepia_print_is_monochrome(self):
        # A uniform tint is not colour; it's subtracted before measuring.
        sepia = ImageOps.colorize(ImageOps.grayscale(make_photo((400, 300))), (40, 25, 10), (250, 235, 205))
        assert looks_monochrome(sepia)


class TestPasteFace:
    face = FaceBox(100, 100, 100, 100)

    def test_the_face_comes_from_the_donor_and_the_rest_from_the_base(self):
        base = make_photo((300, 300))
        donor = make_photo((300, 300)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # same tones, different layout
        result = np.asarray(paste_face(base, donor, self.face)).astype(int)
        assert (result[5, 5] == np.asarray(base)[5, 5]).all()   # far corner untouched
        centre = (slice(140, 160), slice(140, 160))
        assert np.abs(result[centre] - np.asarray(base, int)[centre]).mean() > 10  # donor's content arrived

    def test_the_donor_is_colour_matched_to_the_base(self):
        base = make_photo((300, 300))
        warm = Image.merge("RGB", [c.point(lambda v, k=k: min(255, v + k)) for c, k in zip(base.split(), (40, 0, -40))])
        result = paste_face(base, warm, self.face)
        # The shifted donor is pulled back to the base's grade inside the face.
        inside = (slice(130, 170), slice(130, 170))
        assert np.abs(np.asarray(result, int)[inside] - np.asarray(base, int)[inside]).mean() < 12

    def test_a_donor_at_another_size_is_resized(self):
        base = Image.new("RGB", (300, 300), (10, 10, 10))
        result = paste_face(base, make_photo((600, 600)), self.face)
        assert result.size == (300, 300)

    def test_mask_is_soft_edged(self):
        mask = ellipse_mask((300, 300), self.face)
        assert mask[150, 150] == pytest.approx(1.0, abs=0.01)
        assert mask[0, 0] == pytest.approx(0.0, abs=0.01)
        assert 0.05 < mask[150, 89] < 0.95  # just inside the ellipse's edge: a blend


class TestMatchColour:
    def test_empty_mask_returns_patch_unchanged(self):
        patch = make_photo((50, 50))
        assert match_colour(patch, make_photo((50, 50), seed=3), np.zeros((50, 50))) is patch


class TestChromaTransfer:
    def test_keeps_the_scans_size_and_brightness_and_takes_the_colour(self):
        scan = ImageOps.grayscale(make_photo((800, 600))).convert("RGB")
        colour = make_photo((400, 300))
        result = chroma_transfer(scan, colour)
        assert result.size == (800, 600)
        # Lab lightness is what's carried over (not Rec.601 grey, which diverges on saturated colour).
        lightness = lambda im: cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB)[..., 0].astype(int)
        assert np.abs(lightness(result) - lightness(scan)).mean() < 4   # detail is the scan's
        assert not looks_monochrome(result)                  # colour arrived
