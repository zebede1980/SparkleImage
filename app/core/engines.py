"""One interface over the two places a restoration can run.

`LocalEngine` is Qwen-Image 2.1 Edit and SeedVR2 on the home PC's GPU via
ComfyUI. `CloudEngine` is any nano-gpt image-to-image model. The pipeline asks
an engine for an edit and doesn't care which it is talking to.
"""

from __future__ import annotations

from typing import Optional, Protocol

from PIL import Image

from app.clients import workflows
from app.clients.catalog import ImageModel
from app.clients.comfy import ComfyClient
from app.clients.image_api import ImageAPIClient


class Engine(Protocol):
    name: str
    is_local: bool

    async def edit(self, image: Image.Image, instruction: str, seed: int) -> Image.Image: ...


class LocalEngine:
    """Qwen-Image 2.1 Edit + SeedVR2 on ComfyUI."""

    name = "local/qwen-image-2.1"
    is_local = True

    def __init__(self, client: ComfyClient, edit_megapixels: float = 1.5, steps: int = 25) -> None:
        self.client = client
        self.edit_megapixels = edit_megapixels
        self.steps = steps
        self._last_upload: Optional[tuple[Image.Image, str]] = None

    async def _upload(self, image: Image.Image) -> str:
        # Seeds of the same photo share one upload rather than re-sending it each time.
        if self._last_upload is None or self._last_upload[0] is not image:
            self._last_upload = (image, await self.client.upload(image))
        return self._last_upload[1]

    async def edit(self, image: Image.Image, instruction: str, seed: int) -> Image.Image:
        source = await self._upload(image)
        graph = workflows.qwen_edit(source, instruction, seed, self.edit_megapixels, self.steps)
        return await self.client.run(graph, workflows.OUTPUT_NODE)

    async def upscale(self, image: Image.Image, short_edge: int) -> Image.Image:
        source = await self.client.upload(image)
        return await self.client.run(workflows.seedvr2_upscale(source, short_edge), workflows.OUTPUT_NODE)


class CloudEngine:
    """A nano-gpt image-to-image model."""

    is_local = False

    def __init__(self, client: ImageAPIClient, model: ImageModel) -> None:
        self.client = client
        self.model = model
        self.name = model.id

    async def edit(self, image: Image.Image, instruction: str, seed: int) -> Image.Image:
        return (await self.client.edit(image, instruction, self.model, seed=seed)).convert("RGB")


def short_edge_for(size: tuple[int, int], target: int) -> int:
    """SeedVR2's target: `target` on the short side, but never smaller than the input already is."""
    return max(min(size), target)
