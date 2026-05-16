"""DALL-E image generation and editing client."""

import base64
import io
from typing import Any, Optional

import httpx
from PIL import Image

from app.config.models import NanoGPTConfig
from app.clients.nanogpt import NanoGPTError


class Dalle3Client:
    """Client for DALL-E image generation and editing via NanoGPT/OpenAI API."""

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

    @staticmethod
    def _image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        """Convert a PIL Image to raw bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()

    @staticmethod
    def _make_square(image: Image.Image, size: int) -> Image.Image:
        """Resize image to a square by fitting within size and padding.

        Preserves the image mode (RGB or RGBA) so masks keep transparency.
        """
        # Resize to fit within the square while maintaining aspect ratio
        image.thumbnail((size, size), Image.LANCZOS)
        # Create a square canvas matching the image mode
        if image.mode == "RGBA":
            bg = (0, 0, 0, 0)
        else:
            bg = (255, 255, 255)
        square = Image.new(image.mode, (size, size), bg)
        offset = ((size - image.width) // 2, (size - image.height) // 2)
        square.paste(image, offset)
        return square

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

        The API expects multipart/form-data (not JSON) and requires square
        PNG images under 4MB.

        Args:
            image: Input PIL Image.
            prompt: Edit instruction.
            mask: Optional mask image (white = edit, black = keep).
            size: Output size (must be 256x256, 512x512, or 1024x1024).

        Returns:
            Edited PIL Image.
        """
        # Parse target dimension from size string (e.g. "1024x1024" -> 1024)
        target_dim = int(size.split("x")[0])

        # Convert to RGB and resize to a square — DALL-E 2 requires this
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self._make_square(image, target_dim)

        # DALL-E 2 edits endpoint requires a mask. If none provided,
        # create an almost-transparent mask so the whole image can be edited.
        # Transparent areas indicate where to edit; opaque areas indicate
        # where to preserve the original image.
        if mask is None:
            # Alpha=1 (almost transparent) avoids "invalid_mask_image_format"
            # errors that some APIs return for a fully transparent mask.
            mask = Image.new("RGBA", (target_dim, target_dim), (0, 0, 0, 1))
        else:
            if mask.mode != "RGBA":
                mask = mask.convert("RGBA")
            mask = self._make_square(mask, target_dim)

        # Build multipart/form-data payload — avoids base64 bloat and 413 errors.
        # Use BytesIO so httpx treats each part as a proper file upload.
        files = {
            "image": ("image.png", io.BytesIO(self._image_to_bytes(image, "PNG")), "image/png"),
            "mask": ("mask.png", io.BytesIO(self._image_to_bytes(mask, "PNG")), "image/png"),
        }
        data = {
            "prompt": prompt,
            "n": "1",
            "size": size,
            "response_format": "b64_json",
        }

        response = await self._request("POST", "/images/edits", files=files, data=data)
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
