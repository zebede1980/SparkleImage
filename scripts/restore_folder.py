"""Restore every photo in a folder, headless. The test bench for the pipeline.

    python scripts/restore_folder.py <in_dir> <out_dir> [--comfy URL] [--key K] [--seeds 4] [--cloud m1,m2]

Writes, per photo, into <out_dir>/<stem>/: every candidate, the composite,
restored.png (SeedVR2), faithful.png (scan detail + AI colour) and report.json
with the identity scores and choices. Uses data/config.yaml / SPARKLE_* env
for anything not given on the command line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients.catalog import fetch_image_models  # noqa: E402
from app.clients.comfy import ComfyClient  # noqa: E402
from app.clients.image_api import ImageAPIClient  # noqa: E402
from app.config.manager import get_config  # noqa: E402
from app.core import restore  # noqa: E402
from app.core.engines import CloudEngine, LocalEngine  # noqa: E402
from app.core.identity import IdentityScorer  # noqa: E402

PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


async def restore_one(path: Path, out: Path, local: LocalEngine, cloud: list, scorer: IdentityScorer,
                      seeds: list[int], short_edge: int, margin: float) -> dict:
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    source = restore.load_photo(Image.open(path))
    mode = restore.choose_mode(source)
    instruction = restore.instruction_for(mode)
    print(f"{path.name}: {source.width}x{source.height}, {mode}", flush=True)

    candidates = await restore.generate(source, instruction, local, seeds, cloud, progress=lambda m: print(f"  {m}", flush=True))
    for c in candidates:
        if c.ok:
            c.image.save(out / f"cand_{c.id}.png")
    faces = restore.score(scorer, source, candidates)
    base = restore.default_base(candidates)
    if base is None:
        return {"photo": path.name, "error": "every candidate failed", "candidates": [c.error for c in candidates]}
    choices = restore.default_faces(candidates, base, faces, margin)
    composite = restore.compose(candidates, base, choices, faces)
    composite.save(out / "composite.png")

    restored = await restore.finish_restored(local, composite, short_edge)
    restored.save(out / "restored.png")
    restore.finish_faithful(source, composite).save(out / "faithful.jpg", quality=95)

    final_scores = scorer.score(composite, faces) if faces else []
    report = {
        "photo": path.name,
        "size": source.size,
        "mode": mode,
        "faces": len(faces),
        "base": base,
        "face_choices": {str(k): v for k, v in choices.items()},
        "candidates": {c.id: {"scores": c.scores, "mean": c.mean, "error": c.error} for c in candidates},
        "composite_scores": final_scores,
        "seconds": round(time.monotonic() - started),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"  base {base}, swaps {choices}, composite {[round(s, 3) if s else None for s in final_scores]}, "
          f"{report['seconds']}s", flush=True)
    return report


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--comfy", help="ComfyUI or gateway URL (default: config)")
    parser.add_argument("--key", help="gateway API key (default: config)")
    parser.add_argument("--seeds", type=int, help="local candidates per photo (default: config)")
    parser.add_argument("--cloud", default="", help="comma-separated nano-gpt models to add")
    parser.add_argument("--only", default="", help="comma-separated filenames to restrict to")
    args = parser.parse_args()

    config = get_config()
    url = args.comfy or config.comfy.url
    if not url:
        sys.exit("No ComfyUI URL: pass --comfy or set comfy.url")
    seeds = list(range(1, (args.seeds or config.restore.local_seeds) + 1))

    comfy = ComfyClient(url, args.key if args.key is not None else config.comfy.api_key, run_timeout=config.comfy.run_timeout)
    local = LocalEngine(comfy, config.comfy.edit_megapixels)
    if not await comfy.available():
        sys.exit(f"ComfyUI at {url} isn't answering")

    cloud, api = [], None
    wanted = [m for m in (args.cloud or ",".join(config.restore.cloud_models)).split(",") if m]
    if wanted:
        api = ImageAPIClient(config.image.base_url, config.image.api_key, timeout=config.image.timeout)
        catalogue = await fetch_image_models(config.image.base_url, config.image.api_key)
        cloud = [CloudEngine(api, catalogue[m]) for m in wanted if m in catalogue]

    scorer = IdentityScorer()
    print(f"identity scoring: {'on' if scorer.available else 'OFF (models missing)'}; seeds {seeds}; cloud {[c.name for c in cloud]}")

    only = {n for n in args.only.split(",") if n}
    photos = sorted(p for p in args.in_dir.iterdir() if p.suffix.lower() in PHOTO_TYPES and (not only or p.name in only))
    reports = []
    for path in photos:
        try:
            reports.append(await restore_one(path, args.out_dir / path.stem, local, cloud, scorer, seeds,
                                             config.comfy.upscale_short_edge, config.restore.face_swap_margin))
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            reports.append({"photo": path.name, "error": str(exc)})
    (args.out_dir / "summary.json").write_text(json.dumps(reports, indent=2))
    await comfy.close()
    if api:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
