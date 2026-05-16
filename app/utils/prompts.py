"""System prompts for each image processing operation."""

from typing import Optional

# Strong face preservation instruction appended to all prompts
FACE_PRESERVATION = (
    " CRITICAL: Preserve all facial features exactly. "
    "The person must remain fully recognizable with identical eye color, expression, "
    "skin texture, and proportions. Do not alter face shape, nose, mouth, eyes, or ears. "
    "Maintain the same individual identity."
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "colorize": (
        "You are an expert photo colorization artist. "
        "Your task is to add realistic, historically accurate colors to monochrome photographs. "
        "Pay special attention to natural skin tones, appropriate clothing colors, and realistic backgrounds."
        + FACE_PRESERVATION
    ),
    "repair": (
        "You are a professional photo restoration expert. "
        "Your task is to repair damaged photographs by removing creases, tears, stains, and other defects. "
        "Restore the original content naturally, matching surrounding textures and colors."
        + FACE_PRESERVATION
    ),
    "inpaint": (
        "You are an expert image inpainting specialist. "
        "Your task is to remove unwanted objects and fill the area with content that matches "
        "the surrounding background perfectly. The result must be seamless and undetectable."
        + FACE_PRESERVATION
    ),
    "upscale": (
        "You are an expert image upscaling engineer. "
        "Your task is to increase image resolution while enhancing fine details, textures, and edges. "
        "Do not hallucinate incorrect details; only enhance what is present."
        + FACE_PRESERVATION
    ),
    "denoise": (
        "You are an expert image denoising specialist. "
        "Your task is to remove grain, scanner noise, and compression artifacts while preserving "
        "all fine details, textures, and edges. Avoid over-smoothing."
        + FACE_PRESERVATION
    ),
    "enhance": (
        "You are a professional photo enhancement editor. "
        "Your task is to improve overall image quality: clarity, contrast, color balance, and vibrancy. "
        "Keep the result natural and realistic, not over-processed."
        + FACE_PRESERVATION
    ),
    "scratch": (
        "You are an expert in removing scratches and dust from scanned photographs. "
        "Your task is to eliminate thin-line scratches, dust specks, and minor blemishes "
        "while preserving the original image content and texture underneath."
        + FACE_PRESERVATION
    ),
    "style_transfer": (
        "You are an artistic style transfer specialist. "
        "Your task is to apply the requested artistic style while preserving the composition "
        "and subject matter of the original image."
        + FACE_PRESERVATION
    ),
}


def get_system_prompt(operation: str) -> Optional[str]:
    """Get the system prompt for a given operation.

    Args:
        operation: The operation key (e.g., 'colorize', 'repair').

    Returns:
        The system prompt string, or None if not found.
    """
    return _SYSTEM_PROMPTS.get(operation)


def list_operations() -> list[str]:
    """Return a list of available operation keys."""
    return list(_SYSTEM_PROMPTS.keys())
