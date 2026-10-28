"""Mide la latencia (ms por imagen) y los FPS del flujo completo.

Compara el detector de residuos solo contra el detector con filtro de personas,
para comprobar cuánta latencia añade el filtro. Cada medición incluye
preprocesamiento, inferencia, decodificación y reglas. Se miden dos escenarios:

  * "Imágenes sueltas" (peor caso): cada imagen es una escena nueva, así que el
    filtro siempre busca personas.
  * "Cámara continua": la misma escena se repite como en un video, y el filtro
    reutiliza las personas encontradas durante varios fotogramas.

Uso (desde la raíz del proyecto):
    python -m scripts.benchmark
    python -m scripts.benchmark --imagenes samples/apple.jpg samples/zidane.jpg --repeticiones 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from src.classifier import WasteClassifier, group_recommendations
from src.knowledge_base import KnowledgeBase
from src.model_loader import allowed_rule_labels, create_detector_stack
from src.person_filter import DEFAULT_COMPLEMENT_LABELS
from src.preprocessing import load_image_bgr
from src.vision import ObjectDetector

DEFAULT_IMAGES_DIR = Path("samples")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def measure(
    detector: ObjectDetector,
    classifier: WasteClassifier,
    images: list,
    repetitions: int,
    continuous: bool,
) -> list[float]:
    """Devuelve la latencia en milisegundos de cada pasada por el flujo completo.

    Con ``continuous=True`` cada imagen se repite seguida (como un video) sin
    reiniciar la memoria del filtro de personas.
    """
    # Pasada de calentamiento: la primera inferencia es más lenta y no cuenta.
    for image in images:
        group_recommendations(classifier.classify_all(detector.detect(image)))

    invalidate = getattr(detector, "invalidate_cache", None)
    timings = []
    for image in images:
        if invalidate is not None:
            invalidate()  # cada imagen empieza como una escena nueva
        for _ in range(repetitions):
            if invalidate is not None and not continuous:
                invalidate()  # peor caso: siempre hay que buscar personas
            start = time.perf_counter()
            group_recommendations(classifier.classify_all(detector.detect(image)))
            timings.append((time.perf_counter() - start) * 1000)
    return timings


def main() -> None:
    """Mide tres configuraciones y las muestra en una tabla."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenes", type=Path, nargs="*", help="Imágenes a usar (por defecto, samples/).")
    parser.add_argument("--repeticiones", type=int, default=10)
    parser.add_argument("--model", type=Path, default=Path("models/waste_yolo11n.onnx"))
    parser.add_argument("--modelo-personas", type=Path, default=Path("models/yolox_nano.onnx"))
    parser.add_argument("--rules", type=Path, default=Path("data/waste_rules.json"))
    args = parser.parse_args()

    paths = args.imagenes or sorted(
        p for p in DEFAULT_IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    images = [load_image_bgr(path) for path in paths]
    print(f"{len(images)} imagen(es) x {args.repeticiones} repeticiones\n")

    # nombre -> (usa modelo COCO, clases complementarias, fotogramas entre búsquedas de personas)
    configs = {
        "Solo modelo de residuos": None,
        "+ filtro de personas (cada 3 fotogramas)": ((), 3),
        "+ filtro y complemento COCO (predeterminado)": (DEFAULT_COMPLEMENT_LABELS, 3),
    }
    for continuous, title in ((False, "Imágenes sueltas (peor caso)"), (True, "Cámara continua")):
        print(f"\n{title}")
        print(f"{'Configuración':<45}{'media ms':>10}{'p95 ms':>10}{'FPS':>8}")
        run_configs(args, images, configs, continuous)


def run_configs(args: argparse.Namespace, images: list, configs: dict, continuous: bool) -> None:
    """Mide cada configuración y escribe su fila en la tabla."""
    for name, refresh in configs.items():
        if refresh is None:
            detector = create_detector_stack(args.model, None)
        else:
            complement, frames = refresh
            detector = create_detector_stack(
                args.model, args.modelo_personas, refresh_frames=frames,
                complement_labels=complement,
            )
        classifier = WasteClassifier(
            KnowledgeBase.load(args.rules, known_labels=allowed_rule_labels(detector))
        )
        timings = sorted(measure(detector, classifier, images, args.repeticiones, continuous))
        mean = statistics.fmean(timings)
        p95 = timings[int(0.95 * (len(timings) - 1))]
        print(f"{name:<45}{mean:>10.0f}{p95:>10.0f}{1000 / mean:>8.1f}")


if __name__ == "__main__":
    main()
