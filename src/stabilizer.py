"""Estabilización de recomendaciones entre fotogramas consecutivos.

Con cámara en vivo, el detector puede "parpadear": un objeto aparece en un
fotograma y desaparece en el siguiente, o aparece un falso positivo aislado.
Abrir un pop-up por cada aparición sería inusable. Este módulo avisa de una
recomendación solo cuando se ha visto suficientes fotogramas, y no la repite
mientras el objeto siga en escena.
"""

from __future__ import annotations

from src.classifier import Recommendation


class RecommendationStabilizer:
    """Decide cuándo una recomendación es estable y debe mostrarse al usuario."""

    def __init__(self, confirm_frames: int = 5, release_frames: int = 10) -> None:
        if confirm_frames < 1 or release_frames < 1:
            raise ValueError("confirm_frames and release_frames must be at least 1")
        self._confirm_frames = confirm_frames
        self._release_frames = release_frames
        self._seen: dict[str, int] = {}  # fotogramas en que se vio cada regla
        self._missing: dict[str, int] = {}  # fotogramas seguidos sin verla
        self._announced: set[str] = set()  # reglas que ya se mostraron

    def update(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        """Procesa un fotograma y devuelve las recomendaciones recién confirmadas."""
        present = {rec.rule_id: rec for rec in recommendations}

        for rule_id in present:
            self._seen[rule_id] = self._seen.get(rule_id, 0) + 1
            self._missing[rule_id] = 0

        # Una ausencia breve no reinicia el conteo (tolera el parpadeo); solo
        # se olvida la regla cuando falta varios fotogramas seguidos.
        for rule_id in [rid for rid in self._seen if rid not in present]:
            self._missing[rule_id] += 1
            if self._missing[rule_id] >= self._release_frames:
                self._forget(rule_id)

        confirmed = [
            rec
            for rule_id, rec in present.items()
            if self._seen[rule_id] >= self._confirm_frames
            and rule_id not in self._announced
        ]
        self._announced.update(rec.rule_id for rec in confirmed)
        return confirmed

    @property
    def active_rule_ids(self) -> frozenset[str]:
        """Reglas ya confirmadas que siguen en escena (las que deben mostrarse)."""
        return frozenset(self._announced)

    def reset(self) -> None:
        """Olvida todo el estado (por ejemplo, al detener la cámara)."""
        self._seen.clear()
        self._missing.clear()
        self._announced.clear()

    def _forget(self, rule_id: str) -> None:
        self._seen.pop(rule_id, None)
        self._missing.pop(rule_id, None)
        self._announced.discard(rule_id)
