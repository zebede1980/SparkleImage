"""Image processing pipelines for SparkleImage."""

from app.processors.base import BaseProcessor, ProcessingResult
from app.processors.colorize import ColorizeProcessor
from app.processors.repair import RepairProcessor
from app.processors.inpaint import InpaintProcessor
from app.processors.upscale import UpscaleProcessor
from app.processors.denoise import DenoiseProcessor
from app.processors.enhance import EnhanceProcessor
from app.processors.scratch import ScratchProcessor

__all__ = [
    "BaseProcessor",
    "ProcessingResult",
    "ColorizeProcessor",
    "RepairProcessor",
    "InpaintProcessor",
    "UpscaleProcessor",
    "DenoiseProcessor",
    "EnhanceProcessor",
    "ScratchProcessor",
]
