"""Runtime glue: config, the cached model catalogue, and running operations.

Keeps the UI thin — every tab calls into here rather than talking to the API
client directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from app.clients.catalog import ImageModel, fetch_image_models
from app.clients.comfy import ComfyClient
from app.clients.image_api import ImageAPIClient
from app.config.manager import get_config, get_config_manager
from app.config.models import AppConfig
from app.core.engines import CloudEngine, LocalEngine
from app.core.faces import FaceDetector
from app.core.gallery import save_result
from app.core.identity import IdentityScorer
from app.core.jobs import Engines, JobOptions, JobStore, Worker
from app.core.operations import Operation, get_operation
from app.core.pipeline import RunResult, run_operation

logger = logging.getLogger(__name__)

CATALOGUE_TTL_SECONDS = 3600


class Runtime:
    """Holds the catalogue and the job worker, and runs work against them."""

    def __init__(self, jobs_root: Path = Path("data/jobs")) -> None:
        self._models: dict[str, ImageModel] = {}
        self._fetched_at: float = 0.0
        self._detector = FaceDetector()
        self.jobs = JobStore(jobs_root)
        self.worker = Worker(self.jobs, self.engines_for, IdentityScorer(self._detector))

    def start(self) -> None:
        self.worker.start()

    async def engines_for(self, options: JobOptions) -> Engines:
        """Fresh clients for one job, built inside the worker's event loop."""
        config = self.config
        comfy = ComfyClient(config.comfy.url, config.comfy.api_key, run_timeout=config.comfy.run_timeout) if config.comfy.url else None
        local = LocalEngine(comfy, config.comfy.edit_megapixels) if comfy else None

        api: Optional[ImageAPIClient] = None
        cloud = []
        if options.cloud_models:
            await self.ensure_models()
            api = self._client()
            cloud = [CloudEngine(api, self._models[m]) for m in options.cloud_models if m in self._models]

        if local is None and not cloud:
            raise RuntimeError("Nothing to restore with: set the local GPU's URL in Settings, or pick a cloud model")

        async def close() -> None:
            if comfy:
                await comfy.close()
            if api:
                await api.close()

        return Engines(local, cloud, config.comfy.upscale_short_edge, config.restore.face_swap_margin, close)

    async def local_gpu_status(self) -> str:
        config = self.config.comfy
        if not config.url:
            return "Local GPU: not configured"
        async with ComfyClient(config.url, config.api_key) as client:
            return "Local GPU: online" if await client.available() else "Local GPU: **offline** — is the PC on and ComfyUI running?"

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
