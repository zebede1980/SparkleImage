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

    async def _enhance_prompt(self, image: Image.Image, base_prompt: str) -> str:
        """Optionally use GPT-4o to enhance the prompt."""
        if not self.vision_client.config.use_prompt_enhancement:
            return base_prompt

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert photo restoration prompt engineer. "
                        "Given an image and a basic instruction, write a detailed, "
                        "precise prompt for DALL-E 3 to achieve the best result. "
                        "Focus on technical accuracy and artistic quality."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.vision_client.image_to_base64(image)}},
                        {"type": "text", "text": f"Enhance this prompt for removing an object from a photo: {base_prompt}"},
                    ],
                },
            ]
            response = await self.vision_client.chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=500,
            )
            enhanced = response["choices"][0]["message"]["content"]
            return enhanced if enhanced else base_prompt
        except Exception:
            return base_prompt

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

        base_prompt = (
            f"Remove {object_description} from this image. "
            "The masked area should be filled with content that matches the surrounding "
            "background naturally. Ensure the result is seamless and undetectable."
        )

        system_prompt = get_system_prompt("inpaint")
        prompt = await self._enhance_prompt(image, base_prompt)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            result_image = await self.generation_client.edit_image(
                image=image,
                prompt=full_prompt,
                mask=mask,
                size="1024x1024",
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
