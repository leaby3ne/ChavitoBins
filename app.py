"""Punto de entrada de ChavitoBins.

Modo actual: clasificar una imagen estática (el modo cámara se agrega después).

Uso:
    python app.py --image samples/apple.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.classifier import DEFAULT_MIN_CONFIDENCE, WasteClassifier
from src.coco_labels import COCO_CLASSES
from src.display import format_classification
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError
from src.preprocessing import ImageLoadError, load_image_bgr
from src.vision import InferenceError, ModelLoadError, YoloxDetector

DEFAULT_MODEL_PATH = Path("models/yolox_s.onnx")
DEFAULT_RULES_PATH = Path("data/waste_rules.json")
NO_DETECTIONS_MESSAGE = "No se detectaron objetos con la confianza mínima del detector."


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChavitoBins: orientación para separar residuos.")
    parser.add_argument("--image", type=Path, required=True, help="Imagen a analizar.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument(
        "--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
        help="Confianza mínima para dar una recomendación (0-1).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Ejecuta el flujo completo: reglas, modelo, imagen, detección y recomendación."""
    # Se carga y valida todo antes de procesar la imagen, para fallar pronto.
    knowledge_base = KnowledgeBase.load(args.rules, known_labels=COCO_CLASSES)
    classifier = WasteClassifier(knowledge_base, min_confidence=args.min_confidence)
    detector = YoloxDetector(args.model)

    image = load_image_bgr(args.image)
    detections = detector.detect(image)
    if not detections:
        print(NO_DETECTIONS_MESSAGE)
        return
    for result in classifier.classify_all(detections):
        print(format_classification(result))


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la app y devuelve el código de salida (0 = correcto, 1 = error)."""
    args = parse_args(argv)
    try:
        run(args)
    # Solo se capturan errores esperados del proyecto; cualquier otro error
    # inesperado debe verse completo en lugar de ocultarse.
    except (KnowledgeBaseError, ModelLoadError, InferenceError, ImageLoadError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
