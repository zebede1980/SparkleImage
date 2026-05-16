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
                        {"type": "text", "text": f"Enhance this prompt for enhancing a photo: {base_prompt}"},
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

        focus = kwargs.get("focus", "clarity and color")

        base_prompt = (
            f"Enhance this image focusing on {focus}. "
            "Improve clarity, contrast, and color balance. "
            "Make the image look more vibrant and professionally processed "
            "while keeping it natural and realistic."
        )

        system_prompt = get_system_prompt("enhance")
        prompt = await self._enhance_prompt(image, base_prompt)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            result_image = await self.generation_client.edit_image(
                image=image,
                prompt=full_prompt,
                size="1024x1024",
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
