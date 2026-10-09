"""The restoration pipeline.

    source → N candidate restorations (seeds, optionally cloud models)
           → score every face in every candidate against the source (ArcFace)
           → base candidate + best version of each face, composited
           → finish: SeedVR2 upscale ("restored") and scan-detail + AI colour ("faithful")

Why this shape, rather than a chain of operations, is set out with the
measurements behind it in plans/redesign-2026-10.md. In short: one edit does
the whole job, seeds vary identity as much as models do, ArcFace picks faces
the way a person would, and re-editing zoomed face crops makes things worse.

Everything here works on images in memory; persistence and queueing live in
app.core.jobs, so this is testable without a GPU or a disk.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

from PIL import Image, ImageOps

from app.core.composite import chroma_transfer, looks_monochrome, paste_face
from app.core.engines import Engine, LocalEngine, short_edge_for
from app.core.identity import IdentityScorer, SourceFace, mean_score

logger = logging.getLogger(__name__)

Mode = Literal["colourise", "correct", "keep_bw"]

_KEEP_PEOPLE = (
    "Keep every person exactly the same: same faces, features, expressions, hair, clothing, poses "
    "and positions. Keep the photograph's framing; do not add, remove or move anything."
)

INSTRUCTIONS: dict[Mode, str] = {
    "colourise": (
        "Restore and colourise this old black and white photograph. Repair all damage: creases, cracks, "
        "tears, stains, spots, dust and scratches. Add natural, realistic colour appropriate to the era "
        "of the photograph. " + _KEEP_PEOPLE
    ),
    "correct": (
        "Restore this faded old photograph. Remove colour casts and restore natural, true-to-life colours, "
        "white balance and contrast. Repair any damage, and remove blur, grain and haze so it looks like a "
        "sharp, high-quality photograph. " + _KEEP_PEOPLE
    ),
    "keep_bw": (
        "Restore this old black and white photograph, keeping it black and white. Repair all damage: "
        "creases, cracks, tears, stains, spots, dust and scratches. Improve clarity and contrast so it "
        "looks like a sharp, high-quality print. " + _KEEP_PEOPLE
    ),
}


def choose_mode(image: Image.Image) -> Mode:
    return "colourise" if looks_monochrome(image) else "correct"


def instruction_for(mode: Mode, era: str = "") -> str:
    text = INSTRUCTIONS[mode]
    if era.strip() and mode == "colourise":
        text = text.replace("appropriate to the era of the photograph", f"appropriate to {era.strip()}")
    return text


def load_photo(image: Image.Image) -> Image.Image:
    """Upright RGB — phone scans often carry their rotation only in EXIF."""
    return ImageOps.exif_transpose(image).convert("RGB")


@dataclass
class Candidate:
    """One restoration of the whole photograph."""

    id: str
    engine: str
    seed: int
    is_local: bool
    image: Optional[Image.Image] = None
    error: str = ""
    scores: list[Optional[float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.image is not None

    @property
    def mean(self) -> Optional[float]:
        return mean_score(self.scores)


ProgressFn = Callable[[str], None]


async def _make(engine: Engine, source: Image.Image, instruction: str, seed: int, progress: ProgressFn) -> Candidate:
    tag = f"s{seed}" if engine.is_local else engine.name.replace("/", "_")
    candidate = Candidate(id=f"{'local' if engine.is_local else 'cloud'}-{tag}", engine=engine.name,
                          seed=seed, is_local=engine.is_local)
    try:
        candidate.image = await engine.edit(source, instruction, seed)
        progress(f"{candidate.id}: done")
    except Exception as exc:  # one failed candidate mustn't sink the job
        logger.warning("Candidate %s failed: %s", candidate.id, exc)
        candidate.error = str(exc)
        progress(f"{candidate.id}: failed — {exc}")
    return candidate


async def generate(
    source: Image.Image,
    instruction: str,
    local: Optional[Engine],
    seeds: Sequence[int],
    cloud: Sequence[Engine] = (),
    progress: ProgressFn = lambda _msg: None,
) -> list[Candidate]:
    """Every candidate restoration. Local seeds run one at a time (one GPU);
    cloud models run alongside them."""
    cloud_tasks = [asyncio.create_task(_make(e, source, instruction, seeds[0] if seeds else 0, progress)) for e in cloud]
    candidates = []
    if local is not None:
        for seed in seeds:
            candidates.append(await _make(local, source, instruction, seed, progress))
    candidates.extend(await asyncio.gather(*cloud_tasks))
    return candidates


def score(scorer: IdentityScorer, source: Image.Image, candidates: list[Candidate]) -> list[SourceFace]:
    """Find the people in the source and score each candidate's version of them.

    Faces are measured in the frame of the first local candidate (all local
    candidates share a size); other candidates are resized into it for scoring.
    """
    usable = [c for c in candidates if c.ok]
    if not usable or not scorer.available:
        return []
    reference = next((c for c in usable if c.is_local), usable[0])
    faces = scorer.source_faces(source, reference.image.size)
    for candidate in usable:
        candidate.scores = scorer.score(candidate.image, faces)
    return faces


def default_base(candidates: list[Candidate]) -> Optional[str]:
    """The best-scoring local candidate.

    Not the best-scoring overall: a model that barely changes the photo scores
    highly on identity precisely because it did little restoring (Seedream 4.5
    did exactly this in testing). Among seeds of the same model, that bias
    doesn't exist, so the ranking is fair. Cloud candidates can still be chosen
    as the base by hand, and contribute faces either way.
    """
    usable = [c for c in candidates if c.ok]
    pool = [c for c in usable if c.is_local] or usable
    if not pool:
        return None
    return max(pool, key=lambda c: c.mean if c.mean is not None else -1).id


# Below this, a likeness score says little. Seen on a grainy 1930s wedding
# group where faces were ~60px of mostly print grain: every candidate scored
# 0.2-0.56, and the model had in effect reconstructed the faces. Scores there
# are noise, so they don't drive swaps, and the UI asks for a human eye.
RELIABLE_LIKENESS = 0.5


def default_faces(candidates: list[Candidate], base_id: str, faces: list[SourceFace], margin: float = 0.02) -> dict[int, str]:
    """For each face, the candidate with its best version — if it beats the base's by `margin`.

    The margin stops near-ties swapping faces in for no visible gain; every swap
    is a seam that could have been avoided. Faces whose best score is below
    RELIABLE_LIKENESS are left alone for the same reason.
    """
    by_id = {c.id: c for c in candidates if c.ok}
    base = by_id[base_id]
    choices: dict[int, str] = {}
    for face in faces:
        base_score = base.scores[face.index] if face.index < len(base.scores) else None
        best = max(
            (c for c in by_id.values() if face.index < len(c.scores) and c.scores[face.index] is not None),
            key=lambda c: c.scores[face.index],
            default=None,
        )
        if best is None or best.id == base_id or best.scores[face.index] < RELIABLE_LIKENESS:
            continue
        if base_score is None or best.scores[face.index] - base_score >= margin:
            choices[face.index] = best.id
    return choices


def compose(candidates: list[Candidate], base_id: str, choices: dict[int, str], faces: list[SourceFace]) -> Image.Image:
    """The base candidate with each chosen face pasted in from its donor."""
    by_id = {c.id: c for c in candidates if c.ok}
    result = by_id[base_id].image.convert("RGB")
    for face in faces:
        donor_id = choices.get(face.index)
        if not donor_id or donor_id == base_id or donor_id not in by_id:
            continue
        fx, fy = result.width / face.frame[0], result.height / face.frame[1]
        result = paste_face(result, by_id[donor_id].image, face.box.scaled(fx, fy))
    return result


async def finish_restored(engine: LocalEngine, composite: Image.Image, short_edge: int) -> Image.Image:
    """SeedVR2 to `short_edge` on the short side — sharpening, not reinventing."""
    return await engine.upscale(composite, short_edge_for(composite.size, short_edge))


def finish_faithful(source: Image.Image, composite: Image.Image) -> Image.Image:
    """The source's own detail at its own resolution, coloured from the composite."""
    return chroma_transfer(source, composite)
