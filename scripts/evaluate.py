"""Evalúa el flujo completo sobre carpetas de imágenes etiquetadas.

Estructura esperada (los nombres de carpeta indican lo que debería pasar):

    eval/
      verde/    imágenes de residuos orgánicos      -> debe recomendar el recipiente verde
      blanca/   imágenes de residuos aprovechables  -> debe recomendar el recipiente blanco
      negra/    imágenes de no aprovechables        -> debe ofrecer la pregunta que lleva al negro
      ninguna/  imágenes sin residuos (personas...) -> no debe recomendar nada

Criterio de la carpeta ``negra``: la cámara no puede ver si un empaque es
metalizado o está sucio, así que el sistema "acierta" cuando reconoce un objeto
cuya pregunta de verificación lleva al recipiente negro (el usuario responde Sí).

Uso (desde la raíz del proyecto):
    python -m scripts.evaluate eval
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from src.classifier import Recommendation, WasteClassifier, group_recommendations
from src.knowledge_base import KnowledgeBase
from src.model_loader import allowed_rule_labels, create_detector_stack
from src.preprocessing import load_image_bgr

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Carpeta -> categoría esperada (None significa "no debe recomendar nada").
EXPECTED_FOLDERS = {
    "verde": "green_bin",
    "blanca": "white_bin",
    "negra": "black_bin",
    "ninguna": None,
}


def immediate_categories(recommendations: Iterable[Recommendation]) -> set[str]:
    """Categorías que el sistema recomienda de inmediato, sin preguntar nada."""
    return {rec.category.id for rec in recommendations}


def reachable_categories(recommendations: Iterable[Recommendation]) -> set[str]:
    """Categorías inmediatas más las que se alcanzan respondiendo Sí a una pregunta."""
    categories = set()
    for rec in recommendations:
        categories.add(rec.category.id)
        if rec.follow_up is not None:
            categories.add(rec.follow_up.yes_category_id)
    return categories


def meets_expectation(expected: str | None, recommendations: list[Recommendation]) -> bool:
    """True si el resultado cumple lo esperado para esa carpeta (ver el docstring del módulo)."""
    if expected is None:
        return not recommendations
    if expected == "black_bin":
        return "black_bin" in reachable_categories(recommendations)
    return expected in immediate_categories(recommendations)


def main() -> None:
    """Recorre las carpetas, evalúa cada imagen y resume el acierto por carpeta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carpeta", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/waste_yolo11n.onnx"))
    parser.add_argument("--modelo-personas", type=Path, default=Path("models/yolox_nano.onnx"))
    parser.add_argument("--sin-filtro-personas", action="store_true")
    parser.add_argument("--rules", type=Path, default=Path("data/waste_rules.json"))
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    detector = create_detector_stack(
        args.model, None if args.sin_filtro_personas else args.modelo_personas, refresh_frames=1
    )
    classifier = WasteClassifier(
        KnowledgeBase.load(args.rules, known_labels=allowed_rule_labels(detector)),
        min_confidence=args.min_confidence,
    )

    summary = []
    for folder_name, expected in EXPECTED_FOLDERS.items():
        folder = args.carpeta / folder_name
        if not folder.is_dir():
            continue
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        correct = 0
        print(f"\n== {folder_name} (esperado: {expected or 'nada'}) ==")
        for path in paths:
            invalidate = getattr(detector, "invalidate_cache", None)
            if invalidate is not None:
                invalidate()  # cada imagen es una escena distinta
            recommendations = group_recommendations(
                classifier.classify_all(detector.detect(load_image_bgr(path)))
            )
            ok = meets_expectation(expected, recommendations)
            correct += ok
            shown = ", ".join(sorted(reachable_categories(recommendations))) or "-"
            print(f"  {'OK ' if ok else 'MAL'} {path.name:<28} recomendaciones/alcanzable: {shown}")
        summary.append((folder_name, correct, len(paths)))

    print("\nResumen")
    total_ok = total = 0
    for name, correct, count in summary:
        print(f"  {name:<8} {correct}/{count}" + (f" = {correct / count:.0%}" if count else ""))
        total_ok += correct
        total += count
    if total:
        print(f"  TOTAL    {total_ok}/{total} = {total_ok / total:.0%}")


if __name__ == "__main__":
    main()
