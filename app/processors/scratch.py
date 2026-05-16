"""Scratch removal processor for thin-line scratches on scanned photos."""

from typing import Optional

from PIL import Image

from app.processors.base import BaseProcessor, ProcessingResult
from app.utils.prompts import get_system_prompt


class ScratchProcessor(BaseProcessor):
    """Remove thin-line scratches and dust marks from scanned photographs."""

    @property
    def name(self) -> str:
        return "Remove Scratches"

    @property
    def description(self) -> str:
        return "Remove thin-line scratches, dust, and speckles from scanned photos."

    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        original = image.copy()
        image = self._resize_if_needed(image)

        severity = kwargs.get("severity", "moderate")

        prompt = (
            f"Remove {severity} thin-line scratches, dust, and speckles from this scanned photograph. "
            "Preserve the original image content underneath. "
            "Fill scratched areas with matching texture and color seamlessly."
        )

        system_prompt = get_system_prompt("scratch")

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
                message="Scratch removal complete.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Scratch removal failed: {exc}",
            )
