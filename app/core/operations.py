"""The operations table.

Seven near-identical processor classes (~700 lines whose only difference was a
prompt string) are replaced by one declarative table. Adding an operation is a
few lines here; nothing else changes.

Instructions are deliberately short and imperative. Instruction-following
image-to-image models want "colourise this photograph, keep everything else
identical" — long descriptive prose pushes them toward regenerating the scene,
which is the opposite of restoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

# Appended to every edit instruction. Short on purpose: it has to survive being
# read by a model that is also being told to change something.
PRESERVE = (
    "Keep the composition, framing and every person's face exactly as they are. "
    "Do not add, remove or reposition anything."
)


@dataclass(frozen=True)
class Param:
    """One user-adjustable input for an operation."""

    name: str
    label: str
    kind: Literal["text", "choice", "scale"] = "text"
    default: Any = ""
    choices: tuple[str, ...] = ()
    help: str = ""


@dataclass(frozen=True)
class Operation:
    """A restoration operation: an instruction, its inputs, and how to apply it."""

    id: str
    label: str
    description: str
    build_instruction: Callable[[dict[str, Any]], str]
    params: tuple[Param, ...] = ()
    kind: Literal["edit", "upscale"] = "edit"
    needs_mask: bool = False
    # Localised work must leave the rest of the photograph untouched; whole-image
    # work (colourising, denoising) legitimately changes every pixel.
    composite_mask: bool = False

    def instruction(self, params: Optional[dict[str, Any]] = None) -> str:
        text = self.build_instruction(params or {})
        return f"{text} {PRESERVE}".strip()


def _colorize(p: dict) -> str:
    era = str(p.get("era_hint") or "").strip()
    era_text = f" Use colours true to {era}." if era else ""
    return (
        "Colourise this black and white photograph with natural, realistic colour. "
        "Use believable skin tones." + era_text
    )


def _repair(p: dict) -> str:
    damage = str(p.get("damage_type") or "creases, tears and stains").strip()
    return (
        f"Repair the physical damage to this photograph: {damage}. "
        "Reconstruct what the damage covers so it matches the surrounding detail."
    )


def _descratch(p: dict) -> str:
    severity = str(p.get("severity") or "moderate")
    return (
        f"Remove the scratches, dust specks and thin white lines from this scanned photograph "
        f"({severity} damage). Leave the photograph's own grain and texture intact."
    )


def _denoise(p: dict) -> str:
    strength = str(p.get("strength") or "medium")
    return (
        f"Remove grain, scanner noise and compression artefacts from this photograph "
        f"({strength} strength). Do not smooth away real detail or texture."
    )


def _enhance(p: dict) -> str:
    focus = str(p.get("focus") or "clarity and colour balance").strip()
    return (
        f"Improve this photograph's {focus}. Correct fading and colour casts. "
        "Keep it looking like a photograph, not a digital rendering."
    )


def _remove_object(p: dict) -> str:
    what = str(p.get("object_description") or "the marked object").strip()
    return (
        f"Remove {what} from the marked area and fill it with background that matches "
        "what surrounds it. The result must be seamless."
    )


def _upscale(p: dict) -> str:
    return "Increase resolution and recover fine detail."


OPERATIONS: dict[str, Operation] = {
    "colorize": Operation(
        id="colorize",
        label="Colourise",
        description="Add natural colour to a black & white or sepia photograph.",
        build_instruction=_colorize,
        params=(
            Param("era_hint", "Era (optional)", "text", "", help="e.g. 1950s, wartime, Victorian"),
        ),
    ),
    "repair": Operation(
        id="repair",
        label="Repair damage",
        description="Reconstruct creases, tears, missing corners and stains.",
        build_instruction=_repair,
        params=(
            Param("damage_type", "Damage", "text", "creases, tears and stains"),
        ),
    ),
    "descratch": Operation(
        id="descratch",
        label="Remove scratches",
        description="Clear thin-line scratches and dust from a scan.",
        build_instruction=_descratch,
        params=(
            Param("severity", "Severity", "choice", "moderate", ("light", "moderate", "heavy")),
        ),
    ),
    "denoise": Operation(
        id="denoise",
        label="Denoise",
        description="Reduce grain, scanner noise and compression artefacts.",
        build_instruction=_denoise,
        params=(
            Param("strength", "Strength", "choice", "medium", ("low", "medium", "high")),
        ),
    ),
    "enhance": Operation(
        id="enhance",
        label="Enhance",
        description="General clarity, contrast and colour correction.",
        build_instruction=_enhance,
        params=(
            Param("focus", "Focus on", "text", "clarity and colour balance"),
        ),
    ),
    "remove_object": Operation(
        id="remove_object",
        label="Remove object",
        description="Paint out something you mark, and fill the gap.",
        build_instruction=_remove_object,
        params=(
            Param("object_description", "What to remove", "text", "", help="e.g. the power line"),
        ),
        needs_mask=True,
        composite_mask=True,
    ),
    "upscale": Operation(
        id="upscale",
        label="Upscale",
        description="Enlarge with a dedicated upscaler, not a generative model.",
        build_instruction=_upscale,
        params=(
            Param("target_megapixels", "Target size (MP)", "scale", 8,
                  help="Output size in megapixels"),
        ),
        kind="upscale",
    ),
}


# The one-click chain. Order matters: clean the physical damage before asking a
# model to interpret colour, and enlarge last so the upscaler works on a photo
# that is already repaired rather than magnifying its defects.
RESTORE_CHAIN: tuple[str, ...] = ("descratch", "repair", "denoise", "colorize", "upscale")


def get_operation(operation_id: str) -> Operation:
    try:
        return OPERATIONS[operation_id]
    except KeyError:
        raise KeyError(f"Unknown operation: {operation_id}") from None


def operation_choices() -> list[tuple[str, str]]:
    """(label, id) pairs for a dropdown, in table order."""
    return [(op.label, op.id) for op in OPERATIONS.values()]
