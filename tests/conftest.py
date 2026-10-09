"""Shared test helpers."""

from PIL import Image, ImageDraw


def make_photo(size: tuple[int, int] = (400, 300), seed: int = 0) -> Image.Image:
    """A synthetic stand-in with real large-scale structure.

    Deliberately *not* random noise: the blank-frame detector downsamples to
    32x32 before measuring variance, and uniform noise averages out to flat grey
    at that size, so a noise image reads as "blank". Real photographs have
    large-scale structure, and so must anything standing in for one.
    """
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        shade = int(255 * y / max(1, height - 1))
        draw.line([(0, y), (width, y)], fill=(shade, 60 + shade // 3, 255 - shade))
    draw.rectangle(
        [width // 6, height // 6, width // 2, height // 2], fill=(20 + seed % 200, 200, 40)
    )
    draw.ellipse([width // 2, height // 3, width - width // 8, height - height // 8], fill=(240, 30, 30))
    return image
