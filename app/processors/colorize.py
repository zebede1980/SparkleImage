"""Colorize processor for black & white or sepia photos."""

from typing import Optional

from PIL import Image

from app.processors.base import BaseProcessor, ProcessingResult
from app.utils.prompts import get_system_prompt


class ColorizeProcessor(BaseProcessor):
    """Colorize black and white or sepia photographs."""

    @property
    def name(self) -> str:
        return "Colorize"

    @property
    def description(self) -> str:
        return "Convert black & white or sepia photos to realistic color."

    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        original = image.copy()
        image = self._resize_if_needed(image)

        era_hint = kwargs.get("era_hint", "")
        era_text = f" The photo appears to be from {era_hint}." if era_hint else ""

        prompt = (
            "Colorize this black and white or sepia photograph. "
            "Add realistic, historically appropriate colors."
            f"{era_text}"
            " Ensure skin tones are natural and the overall result looks authentic."
        )

        system_prompt = get_system_prompt("colorize")

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
                message="Colorization complete.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Colorization failed: {exc}",
            )
