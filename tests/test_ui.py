"""UI helper tests — the bits between Gradio's data shapes and the pipeline."""

from PIL import Image

from app.ui.app import _mask_from_editor, _status
from app.core.pipeline import RunResult
from tests.conftest import make_photo


def _layer(size, painted_box=None) -> Image.Image:
    """An ImageEditor layer: transparent except where the user painted."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if painted_box:
        for x in range(painted_box[0], painted_box[2]):
            for y in range(painted_box[1], painted_box[3]):
                layer.putpixel((x, y), (255, 0, 0, 255))
    return layer


class TestMaskExtraction:
    def test_none_is_handled(self):
        assert _mask_from_editor(None) == (None, None)

    def test_a_plain_image_has_no_mask(self):
        photo = make_photo((100, 100))
        assert _mask_from_editor(photo) == (photo, None)

    def test_painted_area_becomes_the_mask(self):
        photo = make_photo((100, 100))
        value = {"background": photo, "layers": [_layer((100, 100), (10, 10, 40, 40))]}
        returned_photo, mask = _mask_from_editor(value)
        assert returned_photo is photo
        assert mask is not None
        assert mask.getpixel((20, 20)) == 255   # painted
        assert mask.getpixel((80, 80)) == 0     # untouched

    def test_an_untouched_canvas_yields_no_mask(self):
        # Nothing painted must not read as "edit the whole photograph".
        photo = make_photo((100, 100))
        value = {"background": photo, "layers": [_layer((100, 100))]}
        _, mask = _mask_from_editor(value)
        assert mask is None

    def test_several_layers_are_merged(self):
        photo = make_photo((100, 100))
        value = {
            "background": photo,
            "layers": [_layer((100, 100), (0, 0, 20, 20)), _layer((100, 100), (60, 60, 90, 90))],
        }
        _, mask = _mask_from_editor(value)
        assert mask.getpixel((10, 10)) == 255
        assert mask.getpixel((75, 75)) == 255
        assert mask.getpixel((40, 40)) == 0

    def test_a_layer_at_another_size_is_resized(self):
        photo = make_photo((100, 100))
        value = {"background": photo, "layers": [_layer((50, 50), (0, 0, 25, 25))]}
        _, mask = _mask_from_editor(value)
        assert mask.size == (100, 100)
        assert mask.getpixel((10, 10)) == 255


class TestStatusRendering:
    def test_success_lists_the_notes(self):
        text = _status(RunResult(image=make_photo((8, 8)), message="Done.", notes=["Face 1: re-edited"]))
        assert "**Done.**" in text and "- Face 1: re-edited" in text

    def test_failure_is_marked(self):
        text = _status(RunResult(image=make_photo((8, 8)), success=False, message="Filter rejected it"))
        assert text.startswith("⚠️")
