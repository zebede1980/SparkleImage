"""Runtime glue: config, the cached model catalogue, and running operations.

Keeps the UI thin — every tab calls into here rather than talking to the API
client directly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from PIL import Image

from app.clients.catalog import ImageModel, fetch_image_models
from app.clients.image_api import ImageAPIClient
from app.config.manager import get_config, get_config_manager
from app.config.models import AppConfig
from app.core.faces import FaceDetector
from app.core.gallery import save_result
from app.core.operations import RESTORE_CHAIN, Operation, get_operation
from app.core.pipeline import RunResult, run_operation

logger = logging.getLogger(__name__)

CATALOGUE_TTL_SECONDS = 3600


class Runtime:
    """Holds the catalogue and runs work against it."""

    def __init__(self) -> None:
        self._models: dict[str, ImageModel] = {}
        self._fetched_at: float = 0.0
        self._detector = FaceDetector()

    @property
    def config(self) -> AppConfig:
        return get_config()

    @property
    def models(self) -> dict[str, ImageModel]:
        return self._models

    @property
    def face_backend(self) -> str:
        return self._detector.backend

    # ── Catalogue ─────────────────────────────────────────────────────────────

    async def ensure_models(self, force: bool = False) -> dict[str, ImageModel]:
        """Fetch the catalogue if it is missing or stale."""
        fresh = (time.time() - self._fetched_at) < CATALOGUE_TTL_SECONDS
        if self._models and fresh and not force:
            return self._models

        config = self.config.image
        if not config.api_key:
            raise RuntimeError("No API key set — add one in Settings.")
        self._models = await fetch_image_models(config.base_url, config.api_key)
        self._fetched_at = time.time()
        logger.info("Fetched %d image models", len(self._models))
        return self._models

    def models_for(self, capability: str) -> list[ImageModel]:
        """Catalogue entries able to do `capability`, unfiltered ones first.

        A content-filtered model will reject photographs of real people, which
        is this app's entire subject matter, so those are pushed down the list
        rather than hidden — sometimes one of them is the right tool.
        """
        attribute = {
            "edit": "can_edit",
            "generate": "can_generate",
            "combine": "can_combine",
            "upscale": "can_upscale",
        }[capability]
        matching = [m for m in self._models.values() if getattr(m, attribute)]
        return sorted(matching, key=lambda m: (m.is_filtered, -m.max_megapixels, m.id))

    def model_label(self, model: ImageModel) -> str:
        marks = []
        if model.is_filtered:
            marks.append("filtered")
        if model.max_megapixels:
            marks.append(f"{model.max_megapixels:.0f}MP")
        if model.custom_resolution:
            marks.append("any ratio")
        return f"{model.name} ({model.id})" + (f" — {', '.join(marks)}" if marks else "")

    # ── Running work ──────────────────────────────────────────────────────────

    def _client(self) -> ImageAPIClient:
        config = self.config.image
        return ImageAPIClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def run(
        self,
        operation_id: str,
        image: Image.Image,
        params: Optional[dict[str, Any]] = None,
        mask: Optional[Image.Image] = None,
        save: bool = True,
    ) -> RunResult:
        """Run one operation, saving the result to the gallery."""
        await self.ensure_models()
        operation = get_operation(operation_id)
        config = self.config

        async with self._client() as client:
            result = await run_operation(
                client=client,
                operation=operation,
                image=image,
                models=self._models,
                api_config=config.image,
                processing=config.processing,
                params=params,
                mask=mask,
                detector=self._detector,
            )

        if result.success and save:
            path = save_result(
                result.image,
                operation.id,
                output_format=config.processing.output_format,
                jpeg_quality=config.processing.jpeg_quality,
                metadata={
                    "model": (
                        config.image.upscale_model
                        if operation.kind == "upscale"
                        else config.image.edit_model
                    ),
                    "instruction": result.instruction,
                    "notes": result.notes,
                    "size": list(result.image.size),
                },
            )
            result.note(f"Saved as {path.name}")
        return result

    async def restore_chain(
        self,
        image: Image.Image,
        steps: Optional[list[str]] = None,
        params: Optional[dict[str, dict[str, Any]]] = None,
        on_step: Optional[Any] = None,
    ) -> tuple[Image.Image, list[RunResult]]:
        """Run several operations in sequence, each on the previous output.

        The order in RESTORE_CHAIN is deliberate: physical damage is cleaned
        before a model is asked to interpret colour, and enlargement happens
        last so the upscaler works on a repaired photograph rather than
        magnifying its defects.
        """
        steps = list(steps or RESTORE_CHAIN)
        params = params or {}
        current = image
        results: list[RunResult] = []

        for index, step in enumerate(steps, start=1):
            if on_step:
                on_step(index, len(steps), get_operation(step))
            # Intermediates are not saved to the gallery — only the finished photo.
            result = await self.run(step, current, params.get(step), save=False)
            results.append(result)
            if result.success:
                current = result.image
            # A failed step is reported and skipped; the chain keeps going with
            # the last good image rather than abandoning the whole restoration.

        return current, results


_runtime: Optional[Runtime] = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


def save_config_changes(**changes: Any) -> None:
    """Apply dotted-path changes to the config and persist them."""
    config = get_config()
    for dotted, value in changes.items():
        section_name, _, field = dotted.partition(".")
        section = getattr(config, section_name)
        setattr(section, field, value)
    get_config_manager().save(config)


def operation_from(label_or_id: str) -> Operation:
    return get_operation(label_or_id)
