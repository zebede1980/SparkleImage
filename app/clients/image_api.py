"""Image API client — generate, edit and upscale against an OpenAI-compatible host.

The call that matters is `edit()`: a source image plus a plain-language
instruction, sent to an image-to-image model. It is the same
`/images/generations` endpoint used for text-to-image, with one extra `image`
field carrying the source as a base64 data URI. This is *not* OpenAI's
`/images/edits` multipart-plus-mask shape, which is DALL-E 2 masked inpainting
and cannot preserve a photograph (plans/revival-2026.md 2.1).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import random
from typing import Any, Optional

import httpx
from PIL import Image

from app.clients.catalog import ImageModel
from app.core.geometry import choose_custom_size, choose_preset_size, encode_for_upload

logger = logging.getLogger(__name__)

# Below this luminance variance an image is effectively a solid colour. A real
# photograph never is, so this is the signature of a content filter returning a
# blank frame with HTTP 200 instead of an error.
BLANK_VARIANCE_THRESHOLD = 4.0


class ImageAPIError(Exception):
    """An image API call failed."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SafetyBlockedError(ImageAPIError):
    """The model returned a blank image — almost always a filter rejection."""


class AuthenticationError(ImageAPIError):
    """The API rejected the key.

    Worth its own type because the catalogue endpoint is public: a model list
    can be fetched successfully with a key that is expired, rotated or absent,
    so the first call that actually checks it is an image request.
    """


def image_variance(image: Image.Image, sample: int = 32) -> float:
    """Luminance variance of a small downscale of `image`."""
    grey = image.convert("L").resize((sample, sample), Image.LANCZOS)
    pixels = grey.tobytes()
    n = len(pixels)
    if n == 0:
        return 0.0
    mean = sum(pixels) / n
    return sum((p - mean) ** 2 for p in pixels) / n


def looks_blank(image: Image.Image) -> bool:
    """Whether an image is suspiciously uniform (see BLANK_VARIANCE_THRESHOLD)."""
    return image_variance(image) < BLANK_VARIANCE_THRESHOLD


def to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class ImageAPIClient:
    """Async client for the image endpoints of an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 300.0,
        max_retries: int = 2,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # Injectable so tests can drive the client without a network.
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
                transport=self._transport,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ImageAPIClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _post(self, path: str, payload: dict) -> dict:
        """POST with retry on transient failures.

        Backoff is `await asyncio.sleep`, not `time.sleep` — the old client
        blocked the whole event loop while one request backed off.
        """
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload)
                if response.status_code in (429, 502, 503, 504) and attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        "The image API rejected the API key. Check it in Settings — "
                        "note that fetching the model list can succeed with a bad key, "
                        "because that endpoint needs no authentication.",
                        status_code=response.status_code,
                    )
                if response.status_code >= 400:
                    raise ImageAPIError(
                        f"HTTP {response.status_code} from {path}: {response.text[:500]}",
                        status_code=response.status_code,
                    )
                return response.json()
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                raise ImageAPIError(f"Connection failed after {self.max_retries} retries: {exc}") from exc

        raise ImageAPIError(f"Request to {path} failed: {last_error}")

    async def _fetch_result_image(self, payload: dict) -> Image.Image:
        """Pull the image out of a response, whether it came back as URL or b64."""
        if isinstance(payload.get("error"), (str, dict)):
            error = payload["error"]
            message = error if isinstance(error, str) else (
                error.get("message") or error.get("details") or str(error)
            )
            raise ImageAPIError(f"API returned an error: {message}")

        entries = payload.get("data") or []
        first = entries[0] if entries else {}

        b64 = first.get("b64_json") if isinstance(first, dict) else None
        if b64:
            return Image.open(io.BytesIO(base64.b64decode(b64)))

        url = (first.get("url") if isinstance(first, dict) else None) or payload.get("image") or payload.get("url")
        if not url:
            raise ImageAPIError(f"No image in response: {str(payload)[:300]}")

        if url.startswith("data:"):
            return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))

        client = await self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    # ── Size selection ────────────────────────────────────────────────────────

    @staticmethod
    def output_size_for(model: ImageModel, source_size: tuple[int, int]) -> Optional[tuple[int, int]]:
        """The output size to request for this model and this source photo.

        Free-form resolutions are preferred: they match the photo's own aspect
        ratio exactly instead of snapping it to the nearest preset.
        """
        if model.custom_resolution:
            return choose_custom_size(source_size, model.custom_resolution)
        if model.presets:
            return choose_preset_size(source_size, model.presets)
        return None

    def _size_payload(self, model: ImageModel, source_size: tuple[int, int]) -> dict:
        size = self.output_size_for(model, source_size)
        if not size:
            return {}
        width, height = size
        # Hosts disagree about which of these they read, and sending all three
        # consistently is what CardGenV2 found to work across model families.
        return {"width": width, "height": height, "size": f"{width}x{height}"}

    # ── Operations ────────────────────────────────────────────────────────────

    async def edit(
        self,
        image: Image.Image,
        instruction: str,
        model: ImageModel,
        seed: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
        check_blank: bool = True,
    ) -> Image.Image:
        """Edit `image` according to `instruction` using an image-to-image model."""
        if not instruction.strip():
            raise ValueError("An edit instruction is required")

        data, mime = encode_for_upload(image, max_bytes=model.max_input_bytes)
        payload: dict[str, Any] = {
            "model": model.id,
            "prompt": instruction.strip(),
            "image": to_data_uri(data, mime),
            "n": 1,
            "response_format": "url",
            "seed": seed if seed is not None else random.randint(0, 2_147_483_647),
        }
        payload.update(self._size_payload(model, image.size))
        if extra:
            payload.update(extra)

        logger.info("Edit: model=%s size=%s upload=%.1fKB", model.id, payload.get("size"), len(data) / 1024)
        result = await self._fetch_result_image(await self._post("/images/generations", payload))

        if check_blank and looks_blank(result):
            raise SafetyBlockedError(
                f"{model.name} returned a blank image. This normally means its content filter "
                f"rejected the source photo"
                + (" — this model is marked as content-filtered; try one that is not."
                   if model.is_filtered else ". Try a different edit model.")
            )
        return result

    async def generate(
        self,
        prompt: str,
        model: ImageModel,
        size: Optional[tuple[int, int]] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Text-to-image. Used for reference material, not for restoration."""
        payload: dict[str, Any] = {
            "model": model.id,
            "prompt": prompt.strip(),
            "n": 1,
            "response_format": "url",
            "seed": seed if seed is not None else random.randint(0, 2_147_483_647),
        }
        if size:
            payload.update(self._size_payload(model, size))
        return await self._fetch_result_image(await self._post("/images/generations", payload))

    async def upscale(
        self,
        image: Image.Image,
        model: ImageModel,
        target_megapixels: Optional[float] = None,
        creativity: Optional[int] = None,
        scale: Optional[int] = None,
    ) -> Image.Image:
        """Enlarge `image` with a dedicated upscaler.

        Upscalers expose different controls — `target_megapixels` + `creativity`
        (Clarity AI), `target_resolution` (SeedVR2), `upscaling_resize` (2x/4x) —
        so the payload is built from whatever this model actually declares.
        Creativity defaults to 0: at or below zero the model sharpens what is
        there rather than inventing detail, which is the only honest setting for
        a photograph of a real person.
        """
        data, mime = encode_for_upload(image, max_bytes=model.max_input_bytes)
        payload: dict[str, Any] = {
            "model": model.id,
            "prompt": "",
            "image": to_data_uri(data, mime),
            "n": 1,
            "response_format": "url",
        }
        params = model.additional_params

        if "target_megapixels" in params:
            default_mp = params["target_megapixels"].get("default", 4)
            payload["target_megapixels"] = int(target_megapixels or default_mp)
            payload["creativity"] = int(creativity if creativity is not None else 0)
        elif "upscaling_resize" in params:
            payload["upscaling_resize"] = int(scale or params["upscaling_resize"].get("default", 2))
        elif "target_resolution" in params:
            options = params["target_resolution"].get("options") or []
            default = params["target_resolution"].get("default")
            wanted = None
            if target_megapixels:
                # 2k ~ 4MP, 4k ~ 8MP, 8k ~ 33MP — pick the smallest that covers.
                for token, megapixels in (("2k", 4), ("4k", 8), ("8k", 33)):
                    if megapixels >= target_megapixels and any(
                        (o.get("value") if isinstance(o, dict) else o) == token for o in options
                    ):
                        wanted = token
                        break
            payload["target_resolution"] = wanted or default or "4k"

        logger.info("Upscale: model=%s params=%s", model.id, {k: v for k, v in payload.items() if k != "image"})
        return await self._fetch_result_image(await self._post("/images/generations", payload))
