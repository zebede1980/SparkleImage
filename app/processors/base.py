"""Base processor class for all image operations."""

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from app.clients.dalle3 import Dalle3Client
from app.clients.nanogpt import NanoGPTClient
from app.config.models import ProcessingConfig


@dataclass
class ProcessingResult:
    """Result of an image processing operation."""

    image: Image.Image
    success: bool = True
    message: str = ""
    metadata: Optional[dict] = None


class BaseProcessor(ABC):
    """Abstract base class for image processing pipelines."""

    def __init__(
        self,
        vision_client: NanoGPTClient,
        generation_client: Dalle3Client,
        config: ProcessingConfig,
    ) -> None:
        self.vision_client = vision_client
        self.generation_client = generation_client
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the processor."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the processor does."""
        ...

    @abstractmethod
    async def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        **kwargs: object,
    ) -> ProcessingResult:
        """Process an image and return the result.

        Args:
            image: Input PIL Image.
            mask: Optional mask image for inpainting operations.
            **kwargs: Additional processor-specific parameters.

        Returns:
            ProcessingResult containing the output image and status.
        """
        ...

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds max_resolution while maintaining aspect ratio."""
        max_dim = self.config.max_resolution
        width, height = image.size
        if max(width, height) > max_dim:
            ratio = max_dim / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.LANCZOS)
        return image

    def _preserve_exif(self, source: Image.Image, target: Image.Image) -> Image.Image:
        """Copy EXIF data from source to target if configured."""
        if not self.config.preserve_exif:
            return target
        if "exif" in source.info:
            target.info["exif"] = source.info["exif"]
        return target

    def _save_to_bytes(self, image: Image.Image, fmt: Optional[str] = None) -> bytes:
        """Save image to bytes in the configured output format."""
        fmt = fmt or self.config.output_format.upper()
        buffer = io.BytesIO()
        save_kwargs: dict = {}
        if fmt in ("JPEG", "JPG"):
            save_kwargs["quality"] = self.config.jpeg_quality
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
        image.save(buffer, format=fmt, **save_kwargs)
        return buffer.getvalue()
