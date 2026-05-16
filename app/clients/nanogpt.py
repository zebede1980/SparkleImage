"""Async NanoGPT API client with retry logic and image handling."""

import base64
import io
import json
import time
from typing import Any, Optional

import httpx
from PIL import Image

from app.config.models import NanoGPTConfig


class NanoGPTError(Exception):
    """Base exception for NanoGPT client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class NanoGPTClient:
    """Client for interacting with the NanoGPT API."""

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
        """Convert a PIL Image to a base64-encoded data URI."""
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{b64}"

    @staticmethod
    def base64_to_image(b64_string: str) -> Image.Image:
        """Convert a base64 string (with or without data URI) to a PIL Image."""
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(data))

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Send a chat completion request."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return await self._request("POST", "/chat/completions", json=payload)

    async def process_image(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.5,
    ) -> Image.Image:
        """Send an image to the model with a prompt and return the resulting image.

        This uses the vision capability by encoding the image as base64 and
        requesting the model to return a base64-encoded image in its response.
        """
        b64_image = self.image_to_base64(image)

        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": b64_image}},
            {"type": "text", "text": prompt},
        ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Request the model to output an image
        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )

        # Extract image from response
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        resp_content = message.get("content", "")

        # The model may return a markdown image tag or raw base64
        if "data:image" in resp_content:
            # Extract data URI
            start = resp_content.find("data:image")
            end = resp_content.find("\"", start)
            if end == -1:
                end = resp_content.find(")", start)
            if end == -1:
                end = len(resp_content)
            b64_data = resp_content[start:end]
            return self.base64_to_image(b64_data)

        # Try raw base64
        try:
            return self.base64_to_image(resp_content)
        except Exception as exc:
            raise NanoGPTError(f"Could not parse image from response: {exc}") from exc

    async def healthcheck(self) -> bool:
        """Check if the API is reachable and the key is valid."""
        try:
            # Try a simple models list or minimal request
            await self._request("GET", "/models")
            return True
        except NanoGPTError:
            return False
