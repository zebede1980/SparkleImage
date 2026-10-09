"""Client tests — the request shape, and the failures that look like successes."""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from app.clients.catalog import ImageModel, parse_image_models
from app.clients.image_api import (
    ImageAPIClient,
    ImageAPIError,
    SafetyBlockedError,
    looks_blank,
)
from tests.conftest import make_photo
from tests.test_catalog import FIXTURE


@pytest.fixture(scope="module")
def models() -> dict[str, ImageModel]:
    return parse_image_models(json.loads(FIXTURE.read_text()))


def photo(size=(400, 300)) -> Image.Image:
    return make_photo(size)


def encoded(image: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def json_route(handler):
    return httpx.MockTransport(handler)


class TestEditRequestShape:
    @pytest.mark.asyncio
    async def test_sends_the_source_image_inline_not_as_multipart(self, models):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"b64_json": encoded(photo())}]})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            await client.edit(photo((4000, 3000)), "Colourise this", models["seedream-v4.5"])

        assert seen["model"] == "seedream-v4.5"
        assert seen["prompt"] == "Colourise this"
        # The whole point: the source rides along in the JSON body of
        # /images/generations, not a multipart /images/edits call with a mask.
        assert seen["image"].startswith("data:image/")
        assert "mask" not in seen

    @pytest.mark.asyncio
    async def test_requested_size_follows_the_source_aspect_ratio(self, models):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"b64_json": encoded(photo())}]})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            await client.edit(photo((4000, 3000)), "Repair", models["seedream-v4.5"])

        assert seen["width"] > seen["height"]              # landscape in, landscape out
        assert seen["size"] == f"{seen['width']}x{seen['height']}"
        assert seen["width"] * seen["height"] >= 3_686_400  # seedream's silent floor

    @pytest.mark.asyncio
    async def test_portrait_source_requests_a_portrait_output(self, models):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"b64_json": encoded(photo())}]})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            await client.edit(photo((1500, 2000)), "Repair", models["seedream-v4.5"])

        assert seen["height"] > seen["width"]

    @pytest.mark.asyncio
    async def test_blank_instruction_is_rejected_before_any_call(self, models):
        async with ImageAPIClient("https://x/api/v1", "k") as client:
            with pytest.raises(ValueError):
                await client.edit(photo(), "   ", models["seedream-v4.5"])


class TestResponseHandling:
    @pytest.mark.asyncio
    async def test_reads_a_url_response_by_downloading_it(self, models):
        buf = io.BytesIO()
        photo().save(buf, format="PNG")
        png = buf.getvalue()

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, content=png, headers={"content-type": "image/png"})
            return httpx.Response(200, json={"data": [{"url": "https://cdn.example/out.png"}]})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            result = await client.edit(photo(), "Enhance", models["seedream-v4.5"])
        assert result.size == (400, 300)

    @pytest.mark.asyncio
    async def test_an_error_object_in_a_200_is_still_an_error(self, models):
        def handler(request):
            return httpx.Response(200, json={"error": {"message": "model unavailable"}})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            with pytest.raises(ImageAPIError, match="model unavailable"):
                await client.edit(photo(), "Enhance", models["seedream-v4.5"])

    @pytest.mark.asyncio
    async def test_http_error_surfaces_the_body(self, models):
        def handler(request):
            return httpx.Response(400, text="bad prompt")

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            with pytest.raises(ImageAPIError, match="bad prompt"):
                await client.edit(photo(), "Enhance", models["seedream-v4.5"])

    @pytest.mark.asyncio
    async def test_retries_a_transient_status_then_succeeds(self, models):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, json={"data": [{"b64_json": encoded(photo())}]})

        client = ImageAPIClient("https://x/api/v1", "k", max_retries=2, transport=json_route(handler))
        async with client:
            await client.edit(photo(), "Enhance", models["seedream-v4.5"])
        assert calls["n"] == 2


class TestSafetyBlockDetection:
    def test_a_solid_frame_is_flagged_and_a_photo_is_not(self):
        assert looks_blank(Image.new("RGB", (256, 256), (0, 0, 0)))
        assert looks_blank(Image.new("RGB", (256, 256), (255, 255, 255)))
        assert not looks_blank(photo((256, 256)))

    @pytest.mark.asyncio
    async def test_a_blank_result_raises_instead_of_returning_a_black_image(self, models):
        def handler(request):
            black = Image.new("RGB", (64, 64), (0, 0, 0))
            return httpx.Response(200, json={"data": [{"b64_json": encoded(black)}]})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            with pytest.raises(SafetyBlockedError, match="content filter"):
                await client.edit(photo(), "Colourise", models["seedream-v4.5"])

    @pytest.mark.asyncio
    async def test_a_filtered_model_says_so_in_the_error(self, models):
        def handler(request):
            black = Image.new("RGB", (64, 64), (0, 0, 0))
            return httpx.Response(200, json={"data": [{"b64_json": encoded(black)}]})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            with pytest.raises(SafetyBlockedError, match="marked as content-filtered"):
                await client.edit(photo(), "Colourise", models["seedream-v5.0-lite"])


class TestUpscalePayloads:
    """Each upscaler exposes different controls; the payload follows the model."""

    async def _capture(self, model, **kwargs):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"b64_json": encoded(photo())}]})

        async with ImageAPIClient("https://x/api/v1", "k", transport=json_route(handler)) as client:
            await client.upscale(photo(), model, **kwargs)
        return seen

    @pytest.mark.asyncio
    async def test_clarity_gets_megapixels_and_zero_creativity(self, models):
        seen = await self._capture(models["clarity-ai-pro-upscaler"], target_megapixels=8)
        assert seen["target_megapixels"] == 8
        # Zero means "sharpen what is there", not "invent detail" — the only
        # defensible default when the subject is a real person.
        assert seen["creativity"] == 0

    @pytest.mark.asyncio
    async def test_seedvr2_gets_a_resolution_token(self, models):
        seen = await self._capture(models["seedvr2-image"], target_megapixels=4)
        assert seen["target_resolution"] in ("2k", "4k", "8k")

    @pytest.mark.asyncio
    async def test_plain_upscaler_gets_a_scale_factor(self, models):
        seen = await self._capture(models["Upscaler"], scale=4)
        assert seen["upscaling_resize"] == 4


class TestAuthenticationErrors:
    """A rejected key must say so — the model list succeeding proves nothing."""

    @pytest.mark.asyncio
    async def test_401_names_the_key_and_the_public_catalogue_trap(self, models):
        def handler(request):
            return httpx.Response(
                401, json={"error": {"message": "Invalid session", "code": "invalid_api_key"}}
            )

        from app.clients.image_api import AuthenticationError

        async with ImageAPIClient("https://x/api/v1", "stale", transport=json_route(handler)) as client:
            with pytest.raises(AuthenticationError, match="needs no authentication"):
                await client.edit(photo(), "Enhance", models["seedream-v4.5"])

    @pytest.mark.asyncio
    async def test_403_is_treated_the_same(self, models):
        def handler(request):
            return httpx.Response(403, text="forbidden")

        from app.clients.image_api import AuthenticationError

        async with ImageAPIClient("https://x/api/v1", "stale", transport=json_route(handler)) as client:
            with pytest.raises(AuthenticationError):
                await client.edit(photo(), "Enhance", models["seedream-v4.5"])

    @pytest.mark.asyncio
    async def test_an_auth_failure_is_not_retried(self, models):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(401, text="nope")

        from app.clients.image_api import AuthenticationError

        client = ImageAPIClient("https://x/api/v1", "stale", max_retries=3,
                                transport=json_route(handler))
        async with client:
            with pytest.raises(AuthenticationError):
                await client.edit(photo(), "Enhance", models["seedream-v4.5"])
        assert calls["n"] == 1  # a bad key will not become a good one on retry
