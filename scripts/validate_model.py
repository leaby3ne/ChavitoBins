"""Verificación de la etapa 1: ¿el modelo carga y detecta objetos en una imagen estática?

Uso (desde la raíz del proyecto):
    python -m scripts.validate_model models/waste_yolo11n.onnx samples/apple.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw

from src.preprocessing import load_image_bgr
from src.model_loader import create_detector


def main() -> None:
    """Carga el modelo, detecta sobre una imagen y guarda una copia con las cajas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("validation_output.jpg"))
    args = parser.parse_args()

    detector = create_detector(args.model, score_threshold=args.threshold)
    image = load_image_bgr(args.image)

    detector.detect(image)  # ejecución de calentamiento, no se incluye en la medición
    start = time.perf_counter()
    detections = detector.detect(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Image: {args.image} {image.shape[1]}x{image.shape[0]}")
    print(f"Inference + postprocess: {elapsed_ms:.0f} ms")
    print(f"Detections (confidence >= {args.threshold}): {len(detections)}")
    for d in detections:
        x1, y1, x2, y2 = d.box
        print(f"  {d.label:<14} {d.confidence:.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    canvas = Image.fromarray(image[:, :, ::-1])
    draw = ImageDraw.Draw(canvas)
    for d in detections:
        draw.rectangle(d.box, outline=(0, 255, 0), width=3)
        draw.text((d.box[0] + 4, d.box[1] + 4), f"{d.label} {d.confidence:.2f}", fill=(255, 255, 0))
    canvas.save(args.output)
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
