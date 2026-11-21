"""Draw object ids on the camera image, so a small VLM can refer to objects by id."""

from __future__ import annotations

from typing import Any

from hfteama.perception.scene import Scene


def to_pil(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")  # numpy HxWx3 uint8


def label_objects(image: Any, scene: Scene, ids: list[str] | None = None) -> Any:
    """Copy of the image with each object's id drawn at its pixel centre (if known)."""
    from PIL import ImageDraw

    img = to_pil(image).copy()
    draw = ImageDraw.Draw(img)
    for oid in ids if ids is not None else scene.ids():
        obj = scene.objects[oid]
        if obj.pixel is None:
            continue
        u, v = obj.pixel
        draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline=(255, 255, 0), width=2)
        x0, y0, x1, y1 = draw.textbbox((u + 6, v - 6), oid)
        draw.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=(0, 0, 0))
        draw.text((u + 6, v - 6), oid, fill=(255, 255, 0))
    return img
