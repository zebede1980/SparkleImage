"""Pipeline tests — the guarantees the old code broke."""

import io
import json
from typing import Optional

import httpx
import pytest
from PIL import Image

from app.clients.catalog import parse_image_models
from app.clients.image_api import ImageAPIClient
from app.config.models import ImageAPIConfig, ProcessingConfig
from app.core.faces import FaceBox, FaceDetector
from app.core.operations import get_operation
from app.core.pipeline import run_operation
from tests.conftest import make_photo
from tests.test_catalog import FIXTURE


@pytest.fixture(scope="module")
def models():
    return parse_image_models(json.loads(FIXTURE.read_text()))


def photo(size=(4000, 3000)) -> Image.Image:
    return make_photo(size)


def client_returning(image: Image.Image, capture: Optional[dict] = None) -> ImageAPIClient:
    """A client whose every call returns `image`, recording what was asked for."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.setdefault("calls", []).append(json.loads(request.content))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        import base64

        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]})

    return ImageAPIClient("https://x/api/v1", "k", transport=httpx.MockTransport(handler))


class NoFaces(FaceDetector):
    def detect(self, image):
        return []


class OneFace(FaceDetector):
    """Reports a single small-in-frame face, as a real scan would."""

    def detect(self, image):
        return [FaceBox(x=400, y=300, width=300, height=300, confidence=0.9)]


@pytest.fixture
def config():
    return ImageAPIConfig(api_key="k", edit_model="seedream-v4.5"), ProcessingConfig()


class TestGeometryGuarantee:
    @pytest.mark.asyncio
    async def test_output_keeps_the_source_dimensions(self, models, config):
        api_config, processing = config
        # The model hands back a square at its own size, as they all do.
        client = client_returning(photo((2048, 2048)))
        async with client:
            result = await run_operation(
                client, get_operation("colorize"), photo((4000, 3000)),
                models, api_config, processing, detector=NoFaces(),
            )
        assert result.success
        assert result.image.size == (4000, 3000)

    @pytest.mark.asyncio
    async def test_oversized_source_is_downscaled_for_upload_only(self, models, config):
        api_config, processing = config
        processing.max_upload_megapixels = 4.0
        capture: dict = {}
        client = client_returning(photo((2048, 1536)), capture)
        async with client:
            result = await run_operation(
                client, get_operation("enhance"), photo((6000, 4500)),
                models, api_config, processing, detector=NoFaces(),
            )
        assert result.image.size == (6000, 4500)
        assert any("downscaled" in n for n in result.notes)


class TestMaskedOperations:
    @pytest.mark.asyncio
    async def test_unmasked_pixels_are_untouched(self, models, config):
        api_config, processing = config
        processing.composite_feather = 0
        source = Image.new("RGB", (200, 200), (255, 0, 0))

        # A blue frame with a vertical gradient: distinct from the source, and
        # with enough large-scale structure not to read as a blank frame.
        edited_frame = Image.new("RGB", (200, 200))
        for y in range(200):
            for x in range(200):
                edited_frame.putpixel((x, y), (0, y, 255 - y))

        mask = Image.new("L", (200, 200), 0)
        for x in range(150, 200):
            for y in range(150, 200):
                mask.putpixel((x, y), 255)

        client = client_returning(edited_frame)
        async with client:
            result = await run_operation(
                client, get_operation("remove_object"), source, models, api_config, processing,
                params={"object_description": "the blot"}, mask=mask, detector=NoFaces(),
            )
        assert result.success
        assert result.image.getpixel((20, 20)) == (255, 0, 0)          # outside: the original photo
        assert result.image.getpixel((180, 180)) == edited_frame.getpixel((180, 180))  # inside: the edit

    @pytest.mark.asyncio
    async def test_missing_mask_is_refused_before_spending_a_call(self, models, config):
        api_config, processing = config
        capture: dict = {}
        client = client_returning(photo((100, 100)), capture)
        async with client:
            result = await run_operation(
                client, get_operation("remove_object"), photo((100, 100)),
                models, api_config, processing, detector=NoFaces(),
            )
        assert not result.success
        assert capture.get("calls") is None


class TestModelChecks:
    @pytest.mark.asyncio
    async def test_a_text_to_image_model_is_refused_with_an_explanation(self, models, config):
        api_config, processing = config
        api_config.edit_model = "flux-2-pro"  # generate-only
        client = client_returning(photo((100, 100)))
        async with client:
            result = await run_operation(
                client, get_operation("colorize"), photo((100, 100)),
                models, api_config, processing, detector=NoFaces(),
            )
        assert not result.success
        assert "text-to-image" in result.message

    @pytest.mark.asyncio
    async def test_an_unknown_model_names_the_problem(self, models, config):
        api_config, processing = config
        api_config.edit_model = "does-not-exist"
        client = client_returning(photo((100, 100)))
        async with client:
            result = await run_operation(
                client, get_operation("colorize"), photo((100, 100)),
                models, api_config, processing, detector=NoFaces(),
            )
        assert not result.success and "not in the catalogue" in result.message

    @pytest.mark.asyncio
    async def test_a_filtered_model_is_flagged_in_the_notes(self, models, config):
        api_config, processing = config
        api_config.edit_model = "seedream-v5.0-lite"  # censored: true
        client = client_returning(photo((1024, 768)))
        async with client:
            result = await run_operation(
                client, get_operation("colorize"), photo((1024, 768)),
                models, api_config, processing, detector=NoFaces(),
            )
        assert result.success
        assert any("content-filtered" in n for n in result.notes)


class TestFacePass:
    @pytest.mark.asyncio
    async def test_a_detected_face_gets_its_own_call(self, models, config):
        api_config, processing = config
        capture: dict = {}
        client = client_returning(photo((1024, 1024)), capture)
        async with client:
            result = await run_operation(
                client, get_operation("colorize"), photo((2000, 1500)),
                models, api_config, processing, detector=OneFace(),
            )
        # One whole-image call plus one face call.
        assert len(capture["calls"]) == 2
        assert any("Face 1" in n for n in result.notes)

    @pytest.mark.asyncio
    async def test_face_pass_can_be_switched_off(self, models, config):
        api_config, processing = config
        processing.face_crop_edit = False
        capture: dict = {}
        client = client_returning(photo((1024, 1024)), capture)
        async with client:
            await run_operation(
                client, get_operation("colorize"), photo((2000, 1500)),
                models, api_config, processing, detector=OneFace(),
            )
        assert len(capture["calls"]) == 1

    @pytest.mark.asyncio
    async def test_operations_that_should_not_touch_faces_do_not(self, models, config):
        api_config, processing = config
        capture: dict = {}
        client = client_returning(photo((200, 200)), capture)
        mask = Image.new("L", (200, 200), 255)
        async with client:
            await run_operation(
                client, get_operation("remove_object"), photo((200, 200)),
                models, api_config, processing, params={"object_description": "x"},
                mask=mask, detector=OneFace(),
            )
        assert len(capture["calls"]) == 1

    @pytest.mark.asyncio
    async def test_a_failing_face_call_does_not_lose_the_main_result(self, models, config):
        api_config, processing = config
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                buf = io.BytesIO()
                photo((1024, 1024)).save(buf, format="PNG")
                import base64
                return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]})
            return httpx.Response(500, text="face call exploded")

        client = ImageAPIClient("https://x/api/v1", "k", max_retries=0,
                                transport=httpx.MockTransport(handler))
        async with client:
            result = await run_operation(
                client, get_operation("colorize"), photo((2000, 1500)),
                models, api_config, processing, detector=OneFace(),
            )
        assert result.success
        assert result.image.size == (2000, 1500)
        assert any("skipped" in n for n in result.notes)


class TestInstructions:
    def test_the_instruction_is_short_and_carries_the_preservation_clause(self):
        instruction = get_operation("colorize").instruction({"era_hint": "the 1950s"})
        assert "1950s" in instruction
        assert "face" in instruction
        # Short imperatives, not the old 60-word "You are an expert..." preamble.
        assert len(instruction) < 400
        assert "You are an expert" not in instruction
