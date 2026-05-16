"""DALL-E 3 image generation and editing client."""

import base64
import io
from typing import Any, Optional

import httpx
from PIL import Image

from app.config.models import NanoGPTConfig
from app.clients.nanogpt import NanoGPTError


class Dalle3Client:
    """Client for DALL-E 3 image generation and editing via NanoGPT/OpenAI API."""

    def __init__(self, config: NanoGPTConfig) -> None:
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_url,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic."""
        import time

        client = await self._get_client()
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.request(method, path, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_exception = exc
                if exc.response.status_code in (429, 502, 503, 504):
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                raise NanoGPTError(
                    f"HTTP {exc.response.status_code}: {exc.response.text}",
                    status_code=exc.response.status_code,
                ) from exc
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exception = exc
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                raise NanoGPTError(f"Connection failed after {self.config.max_retries} retries: {exc}") from exc

        raise NanoGPTError(f"Request failed after retries: {last_exception}")

    @staticmethod
    def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
        """Convert a PIL Image to a base64-encoded string."""
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def base64_to_image(b64_string: str) -> Image.Image:
        """Convert a base64 string to a PIL Image."""
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(data))

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
    ) -> Image.Image:
        """Generate an image from a text prompt using DALL-E 3.

        Args:
            prompt: Text description of the desired image.
            size: Image size ("1024x1024", "1792x1024", or "1024x1792").
            quality: "standard" or "hd".
            style: "vivid" or "natural".

        Returns:
            Generated PIL Image.
        """
        payload = {
            "model": self.config.generation_model.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": "b64_json",
        }

        response = await self._request("POST", "/images/generations", json=payload)
        b64_data = response["data"][0]["b64_json"]
        return self.base64_to_image(b64_data)

    async def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        mask: Optional[Image.Image] = None,
        size: str = "1024x1024",
    ) -> Image.Image:
        """Edit an image using DALL-E 2 edits endpoint.

        DALL-E 2 supports the /images/edits endpoint which requires both
        an image and a mask. When no mask is provided we create a fully
        white mask so the entire image is editable.

        Args:
            image: Input PIL Image.
            prompt: Edit instruction.
            mask: Optional mask image (white = edit, black = keep).
            size: Output size.

        Returns:
            Edited PIL Image.
        """
        # Convert images to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        b64_image = self.image_to_base64(image, fmt="PNG")

        # DALL-E 2 edits endpoint requires a mask. If none provided,
        # create a white mask so the whole image can be edited.
        if mask is None:
            mask = Image.new("RGB", image.size, (255, 255, 255))
        elif mask.mode != "RGB":
            mask = mask.convert("RGB")

        b64_mask = self.image_to_base64(mask, fmt="PNG")

        payload: dict[str, Any] = {
            "image": b64_image,
            "mask": b64_mask,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }

        response = await self._request("POST", "/images/edits", json=payload)
        b64_data = response["data"][0]["b64_json"]
        return self.base64_to_image(b64_data)

    async def create_variation(
        self,
        image: Image.Image,
        n: int = 1,
        size: str = "1024x1024",
    ) -> list[Image.Image]:
        """Create variations of an image.

        Args:
            image: Input PIL Image.
            n: Number of variations (1-4).
            size: Output size.

        Returns:
            List of variation PIL Images.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        b64_image = self.image_to_base64(image, fmt="PNG")

        payload = {
            "image": b64_image,
            "n": min(n, 4),
            "size": size,
            "response_format": "b64_json",
        }

        response = await self._request("POST", "/images/variations", json=payload)
        images = []
        for item in response.get("data", []):
            b64_data = item["b64_json"]
            images.append(self.base64_to_image(b64_data))
        return images
