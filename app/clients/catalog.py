"""The nano-gpt image-model catalogue.

`GET {host}/api/models` returns a rich, structured catalogue: 214 image models
each declaring whether they accept a source image, whether they are content
filtered, what output sizes they support and how large an input they will take.

This is *not* the endpoint the OpenAI-compatible `/v1/models` path serves —
that one returns text models only, with bare ids and no capabilities. Reading
the wrong one is why CardGenV2 has to guess edit-capability from model names
(plans/revival-2026.md 6.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from app.core.geometry import CustomResolution


# Models whose id/name/description look like a dedicated upscaler, used only
# when the parameter sniffing below finds nothing conclusive.
_UPSCALER_PATTERN = re.compile(r"upscal|super.?res|superres", re.I)

# Parameters that only a dedicated upscaler exposes. Presence of any of these
# is a stronger signal than the model's name.
_UPSCALE_PARAMS = ("target_megapixels", "upscaling_resize", "target_resolution")

# Fallback for the ~15 catalogue entries with no `iconLabel`.
_EDIT_NAME_MARKERS = ("image-to-image", "img2img", "-edit", "edit-", "kontext", "inpaint", "instruct")


@dataclass
class ImageModel:
    """One image model as the catalogue describes it."""

    id: str
    name: str
    provider: str = ""
    description: str = ""
    icon_label: Optional[str] = None
    censored: Optional[bool] = None
    multi_image: bool = False
    max_images: int = 1
    max_input_bytes: Optional[int] = None
    presets: list[tuple[int, int]] = field(default_factory=list)
    resolution_tokens: list[str] = field(default_factory=list)
    custom_resolution: Optional[CustomResolution] = None
    additional_params: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)

    # ── Capabilities ──────────────────────────────────────────────────────────
    # Derived from the catalogue rather than guessed from the id, except where
    # the catalogue is silent.

    @property
    def can_generate(self) -> bool:
        if self.icon_label in ("text-to-image", "both"):
            return True
        if self.icon_label == "image-to-image":
            return False
        return not self._looks_edit_by_name()

    @property
    def can_edit(self) -> bool:
        if self.icon_label in ("image-to-image", "both"):
            return True
        if self.icon_label == "text-to-image":
            return False
        return self._looks_edit_by_name()

    @property
    def can_combine(self) -> bool:
        return bool(self.multi_image)

    @property
    def can_upscale(self) -> bool:
        if any(p in self.additional_params for p in _UPSCALE_PARAMS):
            return True
        return bool(_UPSCALER_PATTERN.search(f"{self.id} {self.name} {self.description}"))

    def _looks_edit_by_name(self) -> bool:
        lowered = self.id.lower()
        return any(marker in lowered for marker in _EDIT_NAME_MARKERS)

    @property
    def is_filtered(self) -> bool:
        """True when the provider applies a content filter.

        Photographs of real people — which is the entire subject matter of this
        app — trip these filters often, and a filtered model typically returns a
        blank image with HTTP 200 rather than an error.
        """
        return self.censored is True

    @property
    def max_megapixels(self) -> float:
        areas = [w * h for w, h in self.presets]
        if self.custom_resolution:
            areas.append(self.custom_resolution.max_width * self.custom_resolution.max_height)
        return (max(areas) / 1e6) if areas else 0.0


def _parse_resolutions(entries: Any) -> tuple[list[tuple[int, int]], list[str]]:
    """Split a model's `resolutions` into explicit WxH pairs and opaque tokens.

    Entries are a mix of concrete sizes ("2048x3072") and provider-specific
    labels ("auto", "2k", "4k", "1" meaning one megapixel) — only the former
    can drive geometry, the latter are passed through untouched.
    """
    pairs: list[tuple[int, int]] = []
    tokens: list[str] = []
    for entry in entries or []:
        value = str(entry.get("value", "") if isinstance(entry, dict) else entry)
        match = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", value)
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
        elif value:
            tokens.append(value)
    return pairs, tokens


def parse_image_models(payload: dict) -> dict[str, ImageModel]:
    """Turn a `/api/models` response into ImageModel objects keyed by id."""
    section = (payload.get("models") or {}).get("image") or {}
    models: dict[str, ImageModel] = {}
    for model_id, data in section.items():
        if not isinstance(data, dict):
            continue
        presets, tokens = _parse_resolutions(data.get("resolutions"))
        params = data.get("additionalParams") or {}
        constraints = data.get("inputImageConstraints") or {}
        models[model_id] = ImageModel(
            id=model_id,
            name=str(data.get("name") or model_id),
            provider=str(data.get("provider") or ""),
            description=str(data.get("description") or ""),
            icon_label=data.get("iconLabel"),
            censored=data.get("censored"),
            multi_image=bool(data.get("supportsMultipleImg2Img")),
            max_images=int(data.get("maxImages") or 1),
            max_input_bytes=constraints.get("maxBytes"),
            presets=presets,
            resolution_tokens=tokens,
            custom_resolution=CustomResolution.from_metadata(params.get("customResolution") or {}),
            additional_params=params,
            cost=data.get("cost") or {},
        )
    return models


def catalogue_url(base_url: str) -> str:
    """Derive the catalogue URL from the configured API base.

    The base is the OpenAI-compatible root the image calls use
    (`https://nano-gpt.com/api/v1`); the catalogue lives one level up at
    `https://nano-gpt.com/api/models`.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/models"


async def fetch_image_models(
    base_url: str,
    api_key: str,
    timeout: float = 60.0,
) -> dict[str, ImageModel]:
    """Fetch and parse the image-model catalogue."""
    url = catalogue_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or "models" not in payload:
        raise ValueError(
            f"{url} did not return a model catalogue. "
            "Check the API base URL — the OpenAI-compatible /v1/models path "
            "lists text models only and has no image section."
        )
    return parse_image_models(payload)
