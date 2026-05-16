"""Inpaint processor for removing unwanted objects."""

from typing import Optional

from PIL import Image

from app.processors.base import BaseProcessor, ProcessingResult
from app.utils.prompts import get_system_prompt


class InpaintProcessor(BaseProcessor):
    """Remove unwanted objects or areas using a user-provided mask."""

    @property
    def name(self) -> str:
        return "Remove Object"

    @property
    def description(self) -> str:
        return "Remove unwanted objects by highlighting them with a mask."

    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        original = image.copy()
        image = self._resize_if_needed(image)

        if mask is None:
            return ProcessingResult(
                image=original,
                success=False,
                message="No mask provided for inpainting.",
            )

        # Ensure mask is same size as image
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        object_description = kwargs.get("object_description", "the highlighted object")

        prompt = (
            f"Remove {object_description} from this image. "
            "The masked area should be filled with content that matches the surrounding "
            "background naturally. Ensure the result is seamless and undetectable."
        )

        system_prompt = get_system_prompt("inpaint")

        try:
            # For inpainting we send both image and mask to the API
            # Some APIs accept a mask; here we composite the mask into the prompt
            # or rely on the model's ability to interpret the masked region.
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
                message="Object removal complete.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Inpainting failed: {exc}",
            )
