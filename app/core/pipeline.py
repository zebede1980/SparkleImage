"""Running an operation against a photograph.

Every operation goes through one function. What it guarantees, and what the old
code did not:

- the output has the same pixel dimensions as the source (never a padded square),
- a masked operation changes only the masked region,
- faces get a second pass at their own resolution, then are composited back,
- EXIF survives,
- a blank "success" from a content filter is reported as the error it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

from app.clients.catalog import ImageModel
from app.clients.image_api import ImageAPIClient, SafetyBlockedError
from app.config.models import ImageAPIConfig, ProcessingConfig
from app.core.faces import FaceDetector, crop_face, paste_face
from app.core.geometry import composite_masked, restore_geometry
from app.core.operations import Operation

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """The outcome of one operation."""

    image: Image.Image
    success: bool = True
    message: str = ""
    notes: list[str] = field(default_factory=list)
    instruction: str = ""

    def note(self, text: str) -> None:
        self.notes.append(text)


def _downscale_for_upload(image: Image.Image, max_megapixels: float) -> tuple[Image.Image, bool]:
    """Shrink an oversized source before upload, keeping its aspect ratio."""
    megapixels = (image.width * image.height) / 1e6
    if megapixels <= max_megapixels:
        return image, False
    scale = (max_megapixels / megapixels) ** 0.5
    return (
        image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS),
        True,
    )


def _carry_exif(source: Image.Image, target: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return target
    exif = source.info.get("exif")
    if exif:
        target.info["exif"] = exif
    return target


async def refine_faces(
    client: ImageAPIClient,
    model: ImageModel,
    source: Image.Image,
    edited: Image.Image,
    instruction: str,
    processing: ProcessingConfig,
    detector: Optional[FaceDetector] = None,
    max_faces: int = 4,
) -> tuple[Image.Image, list[str]]:
    """Re-run the edit on each face at full resolution and composite it back.

    The crop comes from the *source*, not from the already-edited frame: the
    point is to give the model the face at the resolution the scan actually
    holds, rather than whatever survived the whole-image pass.
    """
    detector = detector or FaceDetector()
    notes: list[str] = []
    faces = detector.detect(source)
    if not faces:
        return edited, ["No faces detected — no face pass needed."]

    result = edited
    for index, face in enumerate(faces[:max_faces], start=1):
        patch, box = crop_face(source, face, processing.face_padding)
        try:
            refined = await client.edit(patch, instruction, model)
        except SafetyBlockedError as exc:
            notes.append(f"Face {index}: skipped ({exc})")
            continue
        except Exception as exc:  # a failed face must not lose the whole result
            logger.warning("Face refine %d failed: %s", index, exc)
            notes.append(f"Face {index}: skipped ({exc})")
            continue
        result = paste_face(result, refined, box, feather=processing.composite_feather)
        notes.append(
            f"Face {index}: re-edited at {patch.width}×{patch.height} "
            f"({detector.backend}, confidence {face.confidence:.2f})"
        )
    return result, notes


async def run_operation(
    client: ImageAPIClient,
    operation: Operation,
    image: Image.Image,
    models: dict[str, ImageModel],
    api_config: ImageAPIConfig,
    processing: ProcessingConfig,
    params: Optional[dict[str, Any]] = None,
    mask: Optional[Image.Image] = None,
    detector: Optional[FaceDetector] = None,
) -> RunResult:
    """Run one operation and return the result at the source's dimensions."""
    params = params or {}
    original = image.convert("RGB")
    original_size = original.size

    model_id = api_config.upscale_model if operation.kind == "upscale" else api_config.edit_model
    model = models.get(model_id)
    if model is None:
        return RunResult(
            image=original,
            success=False,
            message=(
                f"Model '{model_id}' is not in the catalogue. Open Settings and pick one "
                f"from the fetched list."
            ),
        )

    if operation.kind == "edit" and not model.can_edit:
        return RunResult(
            image=original,
            success=False,
            message=(
                f"'{model.name}' is a text-to-image model — handed a photograph it ignores it "
                f"and invents a new picture. Choose a model marked as image-to-image."
            ),
        )

    if operation.needs_mask and mask is None:
        return RunResult(image=original, success=False, message="Mark the area to work on first.")

    instruction = operation.instruction(params)
    result = RunResult(image=original, instruction=instruction)

    if operation.kind == "edit" and model.is_filtered and api_config.warn_on_filtered_models:
        result.note(
            f"'{model.name}' is content-filtered; photographs of real people are sometimes "
            f"rejected outright. seedream-v4.5 is not filtered."
        )

    source, was_downscaled = _downscale_for_upload(original, processing.max_upload_megapixels)
    if was_downscaled:
        result.note(
            f"Source downscaled to {source.width}×{source.height} for upload; "
            f"the result is returned at the original {original_size[0]}×{original_size[1]}."
        )

    try:
        if operation.kind == "upscale":
            target_mp = params.get("target_megapixels")
            edited = await client.upscale(
                source,
                model,
                target_megapixels=float(target_mp) if target_mp else None,
                creativity=api_config.upscale_creativity,
            )
            result.note(f"Upscaled with {model.name} at creativity {api_config.upscale_creativity}.")
        else:
            edited = await client.edit(source, instruction, model)
            requested = client.output_size_for(model, source.size)
            if requested:
                result.note(f"Requested {requested[0]}×{requested[1]} to match the source's shape.")
    except SafetyBlockedError as exc:
        return RunResult(image=original, success=False, message=str(exc), notes=result.notes)
    except Exception as exc:
        logger.exception("%s failed", operation.id)
        return RunResult(
            image=original, success=False, message=f"{operation.label} failed: {exc}", notes=result.notes
        )

    # Everything below restores the photograph's own geometry. A model returns
    # its own frame at its own size; the output must be the same photo.
    edited = restore_geometry(edited, original_size)

    if operation.refine_faces and processing.face_crop_edit and operation.kind == "edit":
        try:
            edited, face_notes = await refine_faces(
                client, model, original, edited, instruction, processing, detector
            )
            result.notes.extend(face_notes)
        except Exception as exc:
            logger.warning("Face pass failed: %s", exc)
            result.note(f"Face pass skipped: {exc}")

    if operation.composite_mask and mask is not None:
        edited = composite_masked(original, edited, mask, feather=processing.composite_feather)
        result.note("Composited back — pixels outside the marked area are unchanged.")

    edited = _carry_exif(original, edited, processing.preserve_exif)

    result.image = edited
    result.success = True
    result.message = f"{operation.label} complete."
    return result
