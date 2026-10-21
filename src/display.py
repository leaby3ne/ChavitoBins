"""Presentación de los resultados de clasificación (por ahora, solo texto)."""

from __future__ import annotations

from src.classifier import Classification, ClassificationStatus

STATUS_TAGS = {
    ClassificationStatus.RECOMMENDED: "OK",
    ClassificationStatus.NEEDS_VERIFICATION: "VERIFICAR",
    ClassificationStatus.LOW_CONFIDENCE: "INCIERTO",
    ClassificationStatus.NO_RULE: "SIN REGLA",
}
INDENT = "    "


def format_classification(result: Classification) -> str:
    """Convierte una clasificación en un texto corto de varias líneas, en español."""
    detection = result.detection
    header = f"[{STATUS_TAGS[result.status]}] {detection.label} ({detection.confidence:.0%})"
    if result.category is not None:
        header += f" → {result.category.name}"

    lines = [header, f"{INDENT}{result.message}"]
    lines.extend(f"{INDENT}⚠ {warning}" for warning in result.warnings)
    return "\n".join(lines)
