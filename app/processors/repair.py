"""Repair processor for damaged photos (creases, tears, stains)."""

from typing import Optional

from PIL import Image

from app.processors.base import BaseProcessor, ProcessingResult
from app.utils.prompts import get_system_prompt


class RepairProcessor(BaseProcessor):
    """Repair damaged photos such as scans with creases, tears, or stains."""

    @property
    def name(self) -> str:
        return "Repair"

    @property
    def description(self) -> str:
        return "Restore damaged photos by removing creases, tears, and stains."

    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        original = image.copy()
        image = self._resize_if_needed(image)

        damage_type = kwargs.get("damage_type", "creases and tears")

        prompt = (
            f"Repair this damaged photograph. Remove {damage_type}. "
            "Restore the underlying image content naturally. "
            "Match the surrounding texture, color, and lighting seamlessly."
        )

        system_prompt = get_system_prompt("repair")

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
                message="Repair complete.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Repair failed: {exc}",
            )
