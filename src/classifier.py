"""Clasificación determinista de detecciones en recomendaciones de disposición.

El modelo solo dice *qué ve*. Este módulo decide qué recomendar y se niega a dar
una respuesta definitiva cuando la evidencia no alcanza.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.knowledge_base import Category, KnowledgeBase
from src.vision import Detection

# Valor provisional: debe calibrarse con imágenes de prueba reales (ver docs/test-plan).
DEFAULT_MIN_CONFIDENCE = 0.50

LOW_CONFIDENCE_MESSAGE = (
    "No se pudo identificar el objeto con suficiente seguridad "
    "({label}, {confidence:.0%}). Acércalo, mejora la iluminación "
    "o verifica manualmente."
)
NO_RULE_MESSAGE = (
    "Se detectó «{label}», pero no hay una recomendación automática para este "
    "objeto. Consulta las indicaciones de tu servicio de aseo."
)


class ClassificationStatus(Enum):
    """Resultado posible de clasificar una detección."""

    RECOMMENDED = "recommended"
    NEEDS_VERIFICATION = "needs_verification"
    LOW_CONFIDENCE = "low_confidence"
    NO_RULE = "no_rule"


@dataclass(frozen=True)
class Classification:
    """Resultado para una detección, listo para mostrarse al usuario."""

    status: ClassificationStatus
    detection: Detection
    message: str
    warnings: tuple[str, ...] = ()
    rule_id: str | None = None
    category: Category | None = None


class WasteClassifier:
    """Aplica las reglas de la base de conocimiento a las detecciones."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in (0, 1], got {min_confidence}"
            )
        self._knowledge_base = knowledge_base
        self._min_confidence = min_confidence

    def classify(self, detection: Detection) -> Classification:
        """Clasifica una detección. Nunca devuelve una categoría sin evidencia suficiente."""
        # Primero se revisa la confianza: con una predicción dudosa no se da
        # ninguna categoría, aunque la clase tenga regla.
        if detection.confidence < self._min_confidence:
            return Classification(
                status=ClassificationStatus.LOW_CONFIDENCE,
                detection=detection,
                message=LOW_CONFIDENCE_MESSAGE.format(
                    label=detection.label, confidence=detection.confidence
                ),
            )

        rule = self._knowledge_base.find_rule(detection.label)
        # Objeto reconocido pero sin regla: se informa en vez de inventar una.
        if rule is None:
            return Classification(
                status=ClassificationStatus.NO_RULE,
                detection=detection,
                message=NO_RULE_MESSAGE.format(label=detection.label),
            )

        status = (
            ClassificationStatus.NEEDS_VERIFICATION
            if rule.requires_verification
            else ClassificationStatus.RECOMMENDED
        )
        return Classification(
            status=status,
            detection=detection,
            message=rule.recommendation,
            warnings=rule.warnings,
            rule_id=rule.id,
            category=self._knowledge_base.get_category(rule.category_id),
        )

    def classify_all(self, detections: list[Detection]) -> list[Classification]:
        """Clasifica varias detecciones conservando su orden."""
        return [self.classify(detection) for detection in detections]
