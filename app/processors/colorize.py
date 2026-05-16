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
                        {"type": "text", "text": f"Enhance this prompt for colorizing a photo: {base_prompt}"},
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

        era_hint = kwargs.get("era_hint", "")
        era_text = f" The photo appears to be from {era_hint}." if era_hint else ""

        base_prompt = (
            "Colorize this black and white or sepia photograph. "
            "Add realistic, historically appropriate colors."
            f"{era_text}"
            " Ensure skin tones are natural and the overall result looks authentic."
        )

        system_prompt = get_system_prompt("colorize")
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
                message="Colorization complete.",
            )
        except Exception as exc:
            return ProcessingResult(
                image=original,
                success=False,
                message=f"Colorization failed: {exc}",
            )
