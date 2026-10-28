"""Punto de entrada de ChavitoBins.

Sin argumentos abre la interfaz gráfica (imagen o cámara). Con ``--image``
clasifica una imagen y escribe el resultado en la terminal.

Uso:
    python app.py                                  # interfaz gráfica
    python app.py --image samples/apple.jpg        # solo texto
    python app.py --image samples/apple.jpg --detalle

Filtro de personas (activo por defecto; usa models/yolox_nano.onnx, o yolox_s con --modelo-personas):
    python app.py --sin-filtro-personas

Modelo anterior (COCO, no específico de residuos):
    python app.py --model models/yolox_s.onnx --rules data/waste_rules_coco.json --sin-filtro-personas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.classifier import (
    DEFAULT_MIN_CONFIDENCE,
    WasteClassifier,
    group_recommendations,
)
from src.display import (
    NO_DETECTIONS_MESSAGE,
    NOT_RECOGNIZED_MESSAGE,
    format_debug_line,
    format_recommendation,
)
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError
from src.model_loader import allowed_rule_labels, create_detector_stack
from src.person_filter import (
    DEFAULT_COMPLEMENT_LABELS,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_REFRESH_FRAMES,
)
from src.preprocessing import ImageLoadError, load_image_bgr
from src.vision import InferenceError, ModelLoadError, ObjectDetector

DEFAULT_MODEL_PATH = Path("models/waste_yolo11n.onnx")
DEFAULT_RULES_PATH = Path("data/waste_rules.json")
DEFAULT_PERSON_MODEL_PATH = Path("models/yolox_nano.onnx")
TKINTER_MISSING_MESSAGE = (
    "Tkinter no está instalado, y la interfaz gráfica lo necesita. "
    "En Linux: sudo apt install python3-tk. En Windows viene con Python "
    "(reinstala marcando 'tcl/tk and IDLE')."
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="ChavitoBins: orientación para separar residuos.")
    parser.add_argument("--image", type=Path, help="Analiza esta imagen en la terminal (sin interfaz).")
    parser.add_argument("--camara", type=int, default=0, help="Índice de la cámara para la interfaz.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument(
        "--sin-filtro-personas", action="store_true",
        help="Desactiva el filtro que descarta falsos positivos sobre personas.",
    )
    parser.add_argument("--modelo-personas", type=Path, default=DEFAULT_PERSON_MODEL_PATH)
    parser.add_argument(
        "--sin-complemento-coco", action="store_true",
        help="No acepta botellas, frutas ni verduras del modelo COCO (solo filtra personas).",
    )
    parser.add_argument(
        "--iou-personas", type=float, default=DEFAULT_IOU_THRESHOLD,
        help="Solapamiento (IoU) con una persona a partir del cual se descarta un residuo.",
    )
    parser.add_argument(
        "--contencion-personas", type=float, default=DEFAULT_CONTAINMENT_THRESHOLD,
        help="Fracción de una persona cubierta por el residuo para descartarlo.",
    )
    parser.add_argument(
        "--intervalo-personas", type=int, default=DEFAULT_REFRESH_FRAMES,
        help="Cada cuántos fotogramas se vuelve a buscar personas (más alto = más rápido).",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
        help="Confianza mínima para dar una recomendación (0-1).",
    )
    parser.add_argument(
        "--detalle", action="store_true",
        help="Con --image: muestra cada detección con la etiqueta original del modelo.",
    )
    return parser.parse_args(argv)


def build_components(args: argparse.Namespace) -> tuple[ObjectDetector, WasteClassifier]:
    """Carga y valida modelo y reglas. Se hace antes de todo para fallar pronto."""
    detector = create_detector_stack(
        args.model,
        None if args.sin_filtro_personas else args.modelo_personas,
        iou_threshold=args.iou_personas,
        containment_threshold=args.contencion_personas,
        refresh_frames=args.intervalo_personas,
        complement_labels=() if args.sin_complemento_coco else DEFAULT_COMPLEMENT_LABELS,
    )
    # Las reglas se validan contra las etiquetas reales del modelo cargado: si se
    # cambia de modelo sin cambiar de reglas, el error sale aquí y es claro.
    knowledge_base = KnowledgeBase.load(args.rules, known_labels=allowed_rule_labels(detector))
    classifier = WasteClassifier(knowledge_base, min_confidence=args.min_confidence)
    return detector, classifier


def run_cli(args: argparse.Namespace) -> None:
    """Clasifica una imagen y escribe las recomendaciones en la terminal."""
    detector, classifier = build_components(args)
    image = load_image_bgr(args.image)
    detections = detector.detect(image)
    if not detections:
        discarded = getattr(detector, "last_discarded", 0)
        if discarded:
            print(f"Se descartó {discarded} detección(es) por coincidir con una persona.")
        else:
            print(NO_DETECTIONS_MESSAGE)
        return

    results = classifier.classify_all(detections)
    recommendations = group_recommendations(results)
    if recommendations:
        print("\n\n".join(format_recommendation(r) for r in recommendations))
    else:
        print(NOT_RECOGNIZED_MESSAGE)

    if args.detalle:
        print("\n--- Detalle (etiquetas originales del modelo) ---")
        print("\n".join(format_debug_line(r) for r in results))
        discarded = getattr(detector, "last_discarded", 0)
        if discarded:
            print(f"(El filtro de personas descartó {discarded} detección(es).)")


def run_gui_mode(args: argparse.Namespace) -> None:
    """Abre la interfaz gráfica."""
    # Import diferido: el modo texto no debe exigir Tkinter.
    from src.gui import run_gui

    detector, classifier = build_components(args)
    run_gui(detector, classifier, camera_index=args.camara)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la app y devuelve el código de salida (0 = correcto, 1 = error)."""
    args = parse_args(argv)
    try:
        if args.image is not None:
            run_cli(args)
        else:
            run_gui_mode(args)
    except ImportError as exc:
        # Solo el import de Tkinter se trata aquí; otros ImportError se re-lanzan.
        if "tkinter" not in str(exc).lower() and "_tkinter" not in str(exc).lower():
            raise
        print(f"Error: {TKINTER_MISSING_MESSAGE}", file=sys.stderr)
        return 1
    # Solo se capturan errores esperados del proyecto; cualquier otro error
    # inesperado debe verse completo en lugar de ocultarse.
    except (KnowledgeBaseError, ModelLoadError, InferenceError, ImageLoadError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
