"""ComfyUI client — the GPU on the home PC, reached through comfy-gateway.

ComfyUI has no authentication of its own. The gateway (nginx in front of it)
checks an `X-API-Key` header and allowlists the handful of endpoints used here:
`/system_stats`, `/prompt`, `/upload/image`, `/history/<id>` and `/view`. The
same gateway serves CardGenV2, so the two apps share one GPU queue.

The polling loop is deliberately forgiving. ComfyUI's web server stalls for
seconds at a time while it swaps models through VRAM (the text encoder and the
diffusion model don't fit in 16GB together), so a failed poll says nothing
about the job. Only the deadline ends a wait — CardGenV2 learned that treating
one failed poll as fatal abandoned live jobs.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from typing import Any, Optional

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class ComfyError(Exception):
    """A ComfyUI call failed, with a message fit to show the user."""


class ComfyOffline(ComfyError):
    """The PC is off, ComfyUI isn't running, or the gateway is unreachable."""


class ComfyClient:
    """Async client for the few ComfyUI endpoints a workflow run needs."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        run_timeout: float = 600.0,
        poll_interval: float = 1.5,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.run_timeout = run_timeout
        self.poll_interval = poll_interval
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            self._client = httpx.AsyncClient(timeout=60.0, headers=headers, transport=self._transport)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ComfyClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        client = await self._get_client()
        try:
            response = await client.request(method, f"{self.base_url}{path}", **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            # The detail names the gateway's address — log it, don't show it.
            logger.warning("ComfyUI unreachable: %s", exc)
            raise ComfyOffline("The local GPU is offline or unreachable — is the PC on and ComfyUI running?") from exc
        if response.status_code in (401, 403):
            raise ComfyError("The local GPU rejected the API key — check comfy.api_key against the gateway's key")
        return response

    async def available(self) -> bool:
        try:
            return (await self._request("GET", "/system_stats")).status_code == 200
        except ComfyError:
            return False

    async def upload(self, image: Image.Image, name: Optional[str] = None) -> str:
        """Upload a PNG into ComfyUI's input folder; returns the name LoadImage wants."""
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "PNG")
        filename = name or f"sparkle_{uuid.uuid4().hex}.png"
        response = await self._request(
            "POST",
            "/upload/image",
            files={"image": (filename, buffer.getvalue(), "image/png")},
            data={"subfolder": "sparkle"},
        )
        if response.status_code != 200:
            raise ComfyError(f"The local GPU refused the image upload ({response.status_code})")
        body = response.json()
        return f"{body['subfolder']}/{body['name']}" if body.get("subfolder") else body["name"]

    async def queue(self, graph: dict) -> str:
        response = await self._request("POST", "/prompt", json={"prompt": graph})
        if response.status_code != 200:
            # ComfyUI explains validation failures (missing model file, bad input) in the body.
            raise ComfyError(f"The local GPU rejected the workflow: {response.text[:500]}")
        return response.json()["prompt_id"]

    async def wait(self, prompt_id: str, output_node: str) -> dict:
        """Poll until the job finishes; returns the output node's first image record."""
        deadline = time.monotonic() + self.run_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval)
            try:
                response = await self._request("GET", f"/history/{prompt_id}")
            except ComfyOffline:
                continue
            if response.status_code != 200:
                continue
            entry = response.json().get(prompt_id)
            if not entry or entry.get("status", {}).get("completed") is None:
                continue

            status = entry["status"]
            if status.get("status_str") != "success":
                detail = "unknown error"
                for kind, data in status.get("messages", []):
                    if kind == "execution_error":
                        detail = data.get("exception_message", detail).strip()
                raise ComfyError(f"The local GPU failed: {detail}")
            images = entry.get("outputs", {}).get(output_node, {}).get("images") or []
            if not images:
                raise ComfyError("The local GPU finished but produced no image")
            return images[0]
        raise ComfyError("The local GPU took too long — it may be busy with another job")

    async def fetch(self, record: dict) -> Image.Image:
        params = {"filename": record["filename"], "subfolder": record.get("subfolder", ""), "type": record.get("type", "output")}
        response = await self._request("GET", "/view", params=params)
        if response.status_code != 200:
            raise ComfyError(f"Couldn't fetch the result from the local GPU ({response.status_code})")
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    async def run(self, graph: dict, output_node: str) -> Image.Image:
        """Queue a workflow, wait for it, and return its image."""
        started = time.monotonic()
        prompt_id = await self.queue(graph)
        record = await self.wait(prompt_id, output_node)
        image = await self.fetch(record)
        logger.info("ComfyUI job %s: %dx%d in %.0fs", prompt_id, *image.size, time.monotonic() - started)
        return image
