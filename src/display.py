"""Presentación de los resultados de clasificación (por ahora, solo texto)."""

from __future__ import annotations

from src.classifier import Classification, ClassificationStatus, Recommendation

STATUS_TAGS = {
    ClassificationStatus.RECOMMENDED: "OK",
    ClassificationStatus.NEEDS_VERIFICATION: "VERIFICAR",
    ClassificationStatus.LOW_CONFIDENCE: "INCIERTO",
    ClassificationStatus.NO_RULE: "NO RECONOCIDO",
}
INDENT = "    "

NO_DETECTIONS_MESSAGE = "No se detectaron objetos con la confianza mínima del detector."
NOT_RECOGNIZED_MESSAGE = "No se reconoció ningún residuo clasificable en la imagen."


def format_recommendation(recommendation: Recommendation) -> str:
    """Convierte una recomendación en un texto corto de varias líneas, en español."""
    tag = STATUS_TAGS[recommendation.status]
    names = ", ".join(recommendation.names)
    lines = [
        f"[{tag}] {names} → {recommendation.category.name}",
        f"{INDENT}{recommendation.message}",
    ]
    lines.extend(f"{INDENT}⚠ {warning}" for warning in recommendation.warnings)
    follow_up = recommendation.follow_up
    if follow_up is not None:
        lines.append(f"{INDENT}? {follow_up.question}")
        lines.append(f"{INDENT}  Si la respuesta es sí: {follow_up.yes_recommendation}")
    return "\n".join(lines)


def format_debug_line(result: Classification) -> str:
    """Una línea técnica por detección, con la etiqueta original del modelo.

    Solo para calibrar y depurar: es la única salida que muestra las etiquetas
    del modelo en inglés.
    """
    detection = result.detection
    return (
        f"{detection.label:<14} {detection.confidence:>4.0%}  "
        f"{result.status.value:<18} {result.rule_id or '-'}"
    )
