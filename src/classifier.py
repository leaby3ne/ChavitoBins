"""Clasificación determinista de detecciones en recomendaciones de disposición.

El modelo solo dice *qué ve*. Este módulo decide qué recomendar y se niega a dar
una respuesta definitiva cuando la evidencia no alcanza.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from src.knowledge_base import Category, FollowUp, KnowledgeBase
from src.vision import Detection

# Valor provisional: debe calibrarse con imágenes de prueba reales (ver docs/test-plan).
DEFAULT_MIN_CONFIDENCE = 0.50

# Los objetos sin regla NO se nombran: están fuera del alcance de ChavitoBins y
# mostrar la etiqueta del modelo (en inglés) los haría parecer reconocidos.
NO_RULE_MESSAGE = (
    "Objeto no reconocido o fuera del alcance de ChavitoBins. "
    "Consulta las indicaciones de tu servicio de aseo."
)
LOW_CONFIDENCE_MESSAGE = (
    "No se pudo identificar «{name}» con suficiente seguridad ({confidence:.0%}). "
    "Acércalo, mejora la iluminación o verifica manualmente."
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
    # Nombre en español; None cuando el objeto no tiene regla (no reconocido).
    display_name: str | None = None
    follow_up: FollowUp | None = None


@dataclass(frozen=True)
class Recommendation:
    """Recomendación única para una regla, agrupando los objetos que la comparten."""

    rule_id: str
    status: ClassificationStatus
    category: Category
    message: str
    warnings: tuple[str, ...]
    names: tuple[str, ...]  # nombres en español, sin repetir
    # Pregunta pendiente; None si no aplica o si el usuario ya la respondió.
    follow_up: FollowUp | None = None


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
        rule = self._knowledge_base.find_rule(detection.label)
        # Sin regla, el objeto está fuera de alcance sea cual sea la confianza.
        if rule is None:
            return Classification(
                status=ClassificationStatus.NO_RULE,
                detection=detection,
                message=NO_RULE_MESSAGE,
            )

        name = rule.names[detection.label]
        # Con una predicción dudosa no se da ninguna categoría, aunque la
        # clase tenga regla.
        if detection.confidence < self._min_confidence:
            return Classification(
                status=ClassificationStatus.LOW_CONFIDENCE,
                detection=detection,
                message=LOW_CONFIDENCE_MESSAGE.format(
                    name=name, confidence=detection.confidence
                ),
                display_name=name,
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
            display_name=name,
            follow_up=rule.follow_up,
        )

    def answer_follow_up(
        self, recommendation: Recommendation, yes: bool
    ) -> Recommendation:
        """Aplica la respuesta del usuario a la pregunta de verificación.

        Con "sí" la recomendación pasa a la categoría indicada por la regla
        (por ejemplo, el recipiente negro para un empaque metalizado). Con "no"
        se conserva la recomendación original y solo se marca como respondida.
        """
        follow_up = recommendation.follow_up
        if follow_up is None:
            raise ValueError("This recommendation has no pending follow-up question.")
        if not yes:
            return replace(recommendation, follow_up=None)
        return replace(
            recommendation,
            status=ClassificationStatus.RECOMMENDED,
            category=self._knowledge_base.get_category(follow_up.yes_category_id),
            message=follow_up.yes_recommendation,
            warnings=follow_up.yes_warnings,
            follow_up=None,
        )

    def classify_all(self, detections: list[Detection]) -> list[Classification]:
        """Clasifica varias detecciones conservando su orden."""
        return [self.classify(detection) for detection in detections]


def group_recommendations(results: list[Classification]) -> list[Recommendation]:
    """Agrupa los resultados accionables en una recomendación por regla.

    Solo cuentan los resultados con recomendación (definitiva o por verificar).
    Los no reconocidos y los de confianza baja se descartan, y los objetos que
    comparten regla (por ejemplo, banana y manzana) se muestran juntos en vez
    de repetir el mismo mensaje. No se cuenta cuántos hay de cada uno, porque
    el detector puede duplicar un mismo objeto.
    """
    actionable = (
        ClassificationStatus.RECOMMENDED,
        ClassificationStatus.NEEDS_VERIFICATION,
    )
    grouped: dict[str, Recommendation] = {}
    for result in results:
        if result.status not in actionable or result.rule_id is None:
            continue
        existing = grouped.get(result.rule_id)
        if existing is None:
            grouped[result.rule_id] = Recommendation(
                rule_id=result.rule_id,
                status=result.status,
                category=result.category,
                message=result.message,
                warnings=result.warnings,
                names=(result.display_name,),
                follow_up=result.follow_up,
            )
        elif result.display_name not in existing.names:
            grouped[result.rule_id] = Recommendation(
                rule_id=existing.rule_id,
                status=existing.status,
                category=existing.category,
                message=existing.message,
                warnings=existing.warnings,
                names=existing.names + (result.display_name,),
                follow_up=existing.follow_up,
            )
    return list(grouped.values())
