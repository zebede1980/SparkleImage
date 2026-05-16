"""Upscale processor for increasing image resolution."""

from typing import Optional

from PIL import Image

from app.processors.base import BaseProcessor, ProcessingResult
from app.utils.prompts import get_system_prompt


class UpscaleProcessor(BaseProcessor):
    """Upscale images while enhancing detail and sharpness."""

    @property
    def name(self) -> str:
        return "Upscale"

    @property
    def description(self) -> str:
        return "Increase image resolution (2x or 4x) with detail enhancement."

    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        original = image.copy()
        image = self._resize_if_needed(image)

        scale = kwargs.get("scale", 2)
        if scale not in (2, 4):
            scale = 2

        prompt = (
            f"Upscale this image by {scale}x. "
            "Enhance fine details, textures, and edges. "
            "Make the image sharper and clearer while keeping it natural. "
            "Do not introduce artifacts or hallucinate incorrect details."
        )

        system_prompt = get_system_prompt("upscale")

        try:
            result_image = await self.client.process_image(
                image=image,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            result_image = self._preserve_exif(original, result_image)
            return ProcessingResult(
                image=result_image,
                success=True,
                message=f"Upscaled {scale}x.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Upscaling failed: {exc}",
            )
