"""Denoise processor for reducing grain and scanner noise."""

from typing import Optional

from PIL import Image

from app.processors.base import BaseProcessor, ProcessingResult
from app.utils.prompts import get_system_prompt


class DenoiseProcessor(BaseProcessor):
    """Reduce grain, scanner noise, and compression artifacts."""

    @property
    def name(self) -> str:
        return "Denoise"

    @property
    def description(self) -> str:
        return "Remove grain, scanner noise, and compression artifacts."

    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        original = image.copy()
        image = self._resize_if_needed(image)

        noise_type = kwargs.get("noise_type", "grain and noise")
        strength = kwargs.get("strength", "medium")

        prompt = (
            f"Denoise this image. Remove {noise_type} with {strength} strength. "
            "Preserve all fine details, textures, and edges. "
            "The result should look clean and natural, not artificially smoothed."
        )

        system_prompt = get_system_prompt("denoise")

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
                message="Denoising complete.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Denoising failed: {exc}",
            )
