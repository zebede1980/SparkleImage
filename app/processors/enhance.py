"""Enhance processor for general picture quality improvement."""

from typing import Optional

from PIL import Image

from app.processors.base import BaseProcessor, ProcessingResult
from app.utils.prompts import get_system_prompt


class EnhanceProcessor(BaseProcessor):
    """General image quality enhancement: clarity, contrast, color balance."""

    @property
    def name(self) -> str:
        return "Enhance"

    @property
    def description(self) -> str:
        return "Improve overall picture quality, clarity, contrast, and color balance."

    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        original = image.copy()
        image = self._resize_if_needed(image)

        focus = kwargs.get("focus", "clarity and color")

        prompt = (
            f"Enhance this image focusing on {focus}. "
            "Improve clarity, contrast, and color balance. "
            "Make the image look more vibrant and professionally processed "
            "while keeping it natural and realistic."
        )

        system_prompt = get_system_prompt("enhance")

        try:
            result_image = await self.client.process_image(
                image=image,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
            )
            result_image = self._preserve_exif(original, result_image)
            return ProcessingResult(
                image=result_image,
                success=True,
                message="Enhancement complete.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Enhancement failed: {exc}",
            )
