"""Programmatic reference renderer for simple geometric scenes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from selfsight.schemas import Color, SceneObject, SceneSpec, Shape, Size

RGB_PALETTE: dict[Color, tuple[int, int, int]] = {
    Color.RED: (220, 45, 45),
    Color.BLUE: (45, 95, 220),
    Color.GREEN: (45, 165, 85),
    Color.YELLOW: (235, 190, 35),
}
BACKGROUND = (255, 255, 255)
HALF_EXTENT = {Size.SMALL: 36, Size.LARGE: 58}


def _draw_object(draw: ImageDraw.ImageDraw, obj: SceneObject) -> None:
    half = HALF_EXTENT[obj.size]
    x, y = obj.center
    box = (x - half, y - half, x + half, y + half)
    fill = RGB_PALETTE[obj.color]
    if obj.shape == Shape.CIRCLE:
        draw.ellipse(box, fill=fill)
    elif obj.shape == Shape.SQUARE:
        draw.rectangle(box, fill=fill)
    elif obj.shape == Shape.TRIANGLE:
        draw.polygon(((x, y - half), (x - half, y + half), (x + half, y + half)), fill=fill)
    else:
        raise ValueError(f"Unsupported shape: {obj.shape}")


def render_scene(scene: SceneSpec, destination: str | Path | None = None) -> Image.Image:
    image = Image.new("RGB", (scene.canvas_size, scene.canvas_size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for obj in scene.objects:
        _draw_object(draw, obj)
    if destination is not None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG", optimize=False)
    return image
