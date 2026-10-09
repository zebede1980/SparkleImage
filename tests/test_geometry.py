"""Geometry tests — the rules that stop a photo being squashed into a square."""

import pytest
from PIL import Image

from app.core.geometry import (
    CustomResolution,
    choose_custom_size,
    choose_preset_size,
    composite_masked,
    encode_for_upload,
    restore_geometry,
)


# Seedream 4.5's real declared limits, from GET /api/models.
SEEDREAM = CustomResolution(
    min_width=1024, min_height=1024, max_width=4096, max_height=4096,
    step=64, min_total_pixels=3_686_400,
)


class TestChooseCustomSize:
    @pytest.mark.parametrize(
        "source",
        [(4000, 3000), (3000, 4000), (1600, 1200), (800, 600), (5184, 3456), (640, 480)],
    )
    def test_meets_the_silent_pixel_floor(self, source):
        # Under minTotalPixels seedream returns a square at its own size with no
        # error at all, so this floor is the whole reason the function exists.
        width, height = choose_custom_size(source, SEEDREAM)
        assert width * height >= SEEDREAM.min_total_pixels

    @pytest.mark.parametrize(
        "source",
        [(4000, 3000), (3000, 4000), (1600, 1200), (5184, 3456), (1024, 768), (800, 600), (640, 480)],
    )
    def test_preserves_aspect_ratio(self, source):
        width, height = choose_custom_size(source, SEEDREAM)
        source_aspect = source[0] / source[1]
        assert abs((width / height) - source_aspect) / source_aspect < 0.05

    @pytest.mark.parametrize("source", [(4000, 3000), (777, 1013), (5184, 3456), (100, 3000)])
    def test_stays_on_the_step_grid_and_within_bounds(self, source):
        width, height = choose_custom_size(source, SEEDREAM)
        assert width % SEEDREAM.step == 0 and height % SEEDREAM.step == 0
        assert SEEDREAM.min_width <= width <= SEEDREAM.max_width
        assert SEEDREAM.min_height <= height <= SEEDREAM.max_height

    def test_never_returns_a_square_for_a_landscape_photo(self):
        # The exact failure of the old code path.
        width, height = choose_custom_size((4000, 3000), SEEDREAM)
        assert width != height
        assert width > height

    def test_extreme_panorama_clamps_without_crashing(self):
        width, height = choose_custom_size((8000, 1000), SEEDREAM)
        assert (width, height) == (4096, 1024)

    def test_rejects_empty_source(self):
        with pytest.raises(ValueError):
            choose_custom_size((0, 100), SEEDREAM)


class TestChoosePresetSize:
    PRESETS = [(1024, 1024), (1344, 768), (768, 1344), (1152, 896), (896, 1152)]

    def test_picks_closest_aspect(self):
        assert choose_preset_size((4000, 3000), self.PRESETS) == (1152, 896)
        assert choose_preset_size((3000, 4000), self.PRESETS) == (896, 1152)
        assert choose_preset_size((1000, 1000), self.PRESETS) == (1024, 1024)

    def test_ties_go_to_the_larger_option(self):
        assert choose_preset_size((1000, 1000), [(512, 512), (2048, 2048)]) == (2048, 2048)

    def test_empty_presets(self):
        assert choose_preset_size((100, 100), []) is None


class TestEncodeForUpload:
    def test_prefers_lossless_png_when_it_fits(self):
        image = Image.new("RGB", (64, 64), (10, 120, 200))
        _, mime = encode_for_upload(image, max_bytes=10_000_000)
        assert mime == "image/png"

    def test_falls_back_to_jpeg_under_a_tight_budget(self):
        image = Image.effect_noise((900, 900), 90).convert("RGB")
        data, mime = encode_for_upload(image, max_bytes=120_000)
        assert mime == "image/jpeg"
        assert len(data) <= 120_000

    def test_downscales_when_even_low_quality_is_too_big(self):
        image = Image.effect_noise((2000, 2000), 120).convert("RGB")
        data, _ = encode_for_upload(image, max_bytes=20_000)
        assert len(data) <= 20_000

    def test_no_budget_means_no_limit(self):
        image = Image.new("RGB", (32, 32))
        data, mime = encode_for_upload(image)
        assert mime == "image/png" and data


class TestCompositing:
    def test_restore_geometry_returns_original_dimensions(self):
        result = restore_geometry(Image.new("RGB", (2048, 1536)), (4000, 3000))
        assert result.size == (4000, 3000)

    def test_masked_composite_leaves_unmasked_pixels_untouched(self):
        original = Image.new("RGB", (100, 100), (255, 0, 0))
        edited = Image.new("RGB", (100, 100), (0, 0, 255))
        mask = Image.new("L", (100, 100), 0)
        for x in range(60, 100):
            for y in range(60, 100):
                mask.putpixel((x, y), 255)

        out = composite_masked(original, edited, mask, feather=0)
        assert out.getpixel((10, 10)) == (255, 0, 0)     # outside the mask: original
        assert out.getpixel((90, 90)) == (0, 0, 255)     # inside the mask: edited

    def test_masked_composite_resizes_a_mismatched_edit(self):
        original = Image.new("RGB", (100, 100), (255, 0, 0))
        edited = Image.new("RGB", (400, 400), (0, 0, 255))
        mask = Image.new("L", (100, 100), 255)
        assert composite_masked(original, edited, mask, feather=0).size == (100, 100)
