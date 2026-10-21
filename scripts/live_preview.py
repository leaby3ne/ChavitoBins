"""Vista previa en vivo con detección y recomendación (herramienta de diagnóstico).

Sirve para medir el rendimiento real en cada PC antes de construir la interfaz
final. Pulsa 'q' o Esc en la ventana para salir.

Uso (desde la raíz del proyecto):
    python -m scripts.live_preview
    python -m scripts.live_preview --camera 0
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import DEFAULT_CAMERA_INDEX, Camera, CameraError
from src.classifier import (
    DEFAULT_MIN_CONFIDENCE,
    Classification,
    ClassificationStatus,
    WasteClassifier,
)
from src.coco_labels import COCO_CLASSES
from src.knowledge_base import KnowledgeBase
from src.vision import YoloxDetector

WINDOW_NAME = "ChavitoBins - vista previa"
QUIT_KEYS = (ord("q"), 27)  # 'q' o Esc

# Colores BGR por estado. Los textos van sin acentos porque las fuentes de
# OpenCV no dibujan caracteres como la ñ o las tildes.
STATUS_COLORS = {
    ClassificationStatus.RECOMMENDED: (0, 200, 0),
    ClassificationStatus.NEEDS_VERIFICATION: (0, 200, 255),
    ClassificationStatus.LOW_CONFIDENCE: (160, 160, 160),
    ClassificationStatus.NO_RULE: (160, 160, 160),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(frame: np.ndarray, results: list[Classification], ms: float) -> np.ndarray:
    """Dibuja cajas, etiquetas y el tiempo de procesamiento sobre una copia."""
    canvas = frame.copy()
    for result in results:
        x1, y1, x2, y2 = (int(v) for v in result.detection.box)
        color = STATUS_COLORS[result.status]
        text = f"{result.detection.label} {result.detection.confidence:.0%}"
        if result.category is not None:
            text += f" | {result.category.name}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        # Fondo oscuro detrás del texto para que se lea sobre cualquier imagen.
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        text_y = max(y1 - 6, text_height + 8)
        cv2.rectangle(
            canvas,
            (x1, text_y - text_height - 4),
            (x1 + text_width + 8, text_y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(canvas, text, (x1 + 4, text_y), FONT, 0.5, color, 1)
    cv2.putText(canvas, f"Deteccion: {ms:.0f} ms", (8, 20), FONT, 0.6, (255, 255, 255), 2)
    return canvas


def _detect_frame(detector: YoloxDetector, classifier: WasteClassifier, frame: np.ndarray) -> tuple[list[Classification], float]:
    start = time.perf_counter()
    results = classifier.classify_all(detector.detect(frame))
    return results, (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--model", type=Path, default=Path("models/yolox_s.onnx"))
    parser.add_argument("--rules", type=Path, default=Path("data/waste_rules.json"))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    args = parser.parse_args()

    knowledge_base = KnowledgeBase.load(args.rules, known_labels=COCO_CLASSES)
    classifier = WasteClassifier(knowledge_base, min_confidence=args.min_confidence)
    detector = YoloxDetector(args.model)

    try:
        with Camera(index=args.camera) as camera:
            latest_results: list[Classification] = []
            elapsed_ms = 0.0
            pending: Future[tuple[list[Classification], float]] | None = None

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="Detection") as executor:
                while True:
                    frame = camera.read()

                    if pending is not None and pending.done():
                        latest_results, elapsed_ms = pending.result()
                        pending = None

                    if pending is None:
                        pending = executor.submit(
                            _detect_frame, detector, classifier, frame.copy()
                        )

                    cv2.imshow(WINDOW_NAME, annotate(frame, latest_results, elapsed_ms))
                    if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                        break
    except CameraError as exc:
        print(f"Error de cámara: {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
