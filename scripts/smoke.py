#!/usr/bin/env python
"""Run one operation against a real photograph and report what happened.

This is the live check that the unit tests cannot do: it spends real API credit.

    python scripts/smoke.py photo.jpg colorize --era "the 1950s"
    python scripts/smoke.py photo.jpg descratch --model seedream-v4.5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.manager import get_config  # noqa: E402
from app.core.faces import FaceDetector  # noqa: E402
from app.core.operations import OPERATIONS  # noqa: E402
from app.core.runtime import get_runtime  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("operation", choices=list(OPERATIONS))
    parser.add_argument("--model", help="Override the configured edit model")
    parser.add_argument("--era", default="", help="Era hint for colourise")
    parser.add_argument("--out", type=Path, help="Where to write the result")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = get_config()
    if args.model:
        config.image.edit_model = args.model
    if not config.image.api_key:
        print("No API key. Set SPARKLE_IMAGE__API_KEY or add one to data/config.yaml.")
        return 2

    source = Image.open(args.photo).convert("RGB")
    runtime = get_runtime()

    print(f"Source      : {args.photo} — {source.width}×{source.height} "
          f"({source.width * source.height / 1e6:.1f}MP)")

    models = await runtime.ensure_models()
    model = models.get(config.image.edit_model)
    print(f"Catalogue   : {len(models)} image models")
    if model:
        print(f"Edit model  : {model.name} ({model.id}) — "
              f"{'filtered' if model.is_filtered else 'unfiltered'}, "
              f"max {model.max_megapixels:.1f}MP, "
              f"{'free-form sizes' if model.custom_resolution else 'preset sizes'}")
        from app.clients.image_api import ImageAPIClient

        requested = ImageAPIClient.output_size_for(model, source.size)
        if requested:
            source_aspect = source.width / source.height
            asked_aspect = requested[0] / requested[1]
            print(f"Will request: {requested[0]}×{requested[1]} "
                  f"(source ratio {source_aspect:.3f}, requested {asked_aspect:.3f})")

    detector = FaceDetector()
    faces = detector.detect(source)
    print(f"Faces       : {len(faces)} via {detector.backend}"
          + (f" — largest {faces[0].width}×{faces[0].height}px" if faces else ""))

    params = {"era_hint": args.era} if args.operation == "colorize" else {}
    started = time.monotonic()
    result = await runtime.run(args.operation, source, params)
    elapsed = time.monotonic() - started

    print(f"\n{'✅' if result.success else '⚠️ '} {result.message}  [{elapsed:.1f}s]")
    for note in result.notes:
        print(f"   - {note}")

    if result.success:
        out = args.out or args.photo.with_name(f"{args.photo.stem}_{args.operation}.png")
        result.image.save(out)
        same = result.image.size == source.size
        print(f"\nOutput      : {out} — {result.image.width}×{result.image.height} "
              f"({'same size as the source' if same else 'SIZE CHANGED'})")
        return 0 if same else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
