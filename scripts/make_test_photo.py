#!/usr/bin/env python
"""Age a photograph, so the restoration operations have something to restore.

Makes a plausible damaged scan out of any colour photo: desaturated to sepia,
faded contrast, film grain, scratches, dust and a crease. Useful for checking
the pipeline end to end without reaching for someone's real family photographs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def age(image: Image.Image, seed: int = 7) -> Image.Image:
    rng = random.Random(seed)
    width, height = image.size

    grey = image.convert("L")
    sepia = Image.merge("RGB", [
        grey.point(lambda v: min(255, int(v * 1.07 + 18))),
        grey.point(lambda v: min(255, int(v * 0.96 + 12))),
        grey.point(lambda v: min(255, int(v * 0.78 + 6))),
    ])
    faded = ImageEnhance.Contrast(sepia).enhance(0.72)

    grain = Image.effect_noise((width, height), 18).convert("L")
    faded = Image.blend(faded, Image.merge("RGB", [grain] * 3), 0.10)

    draw = ImageDraw.Draw(faded)
    for _ in range(rng.randint(14, 22)):  # scratches
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        length = rng.randint(height // 12, height // 3)
        drift = rng.randint(-24, 24)
        shade = rng.randint(205, 250)
        draw.line([(x, y), (x + drift, y + length)], fill=(shade, shade, shade), width=rng.choice([1, 1, 2]))
    for _ in range(rng.randint(90, 160)):  # dust
        x, y = rng.randint(0, width), rng.randint(0, height)
        radius = rng.randint(1, 3)
        shade = rng.choice([25, 235])
        draw.ellipse([x, y, x + radius, y + radius], fill=(shade, shade, shade))

    crease_x = int(width * rng.uniform(0.35, 0.65))
    draw.line([(crease_x, 0), (crease_x + rng.randint(-40, 40), height)], fill=(232, 226, 210), width=4)

    # Corner wear.
    for corner in ((0, 0), (width, 0), (0, height), (width, height)):
        size = rng.randint(min(width, height) // 14, min(width, height) // 8)
        box = [corner[0] - size, corner[1] - size, corner[0] + size, corner[1] + size]
        draw.ellipse(box, fill=(238, 232, 214))

    return faded.filter(ImageFilter.GaussianBlur(0.4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-width", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    image = Image.open(args.source).convert("RGB")
    if image.width > args.max_width:
        ratio = args.max_width / image.width
        image = image.resize((args.max_width, int(image.height * ratio)), Image.LANCZOS)

    aged = age(image, args.seed)
    aged.save(args.destination, quality=92)
    print(f"{args.destination} — {aged.width}×{aged.height}")


if __name__ == "__main__":
    main()
