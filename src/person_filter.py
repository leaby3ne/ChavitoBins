"""Detector híbrido: modelo de residuos + modelo COCO.

El modelo de residuos tiene dos debilidades que el modelo COCO compensa:

1. Siempre elige uno de sus materiales, así que ante una persona puede inventar
   "papel" o "vidrio". El detector COCO sí reconoce la clase ``person``, y se
   descartan las detecciones de residuos que corresponden a una persona.
2. Reconoce mal algunos objetos cotidianos (una botella transparente en la mano
   salió con 0.24 de confianza para "plástico"), mientras que COCO los detecta
   con seguridad. Por eso se aceptan además ciertas clases COCO
   (``DEFAULT_COMPLEMENT_LABELS``: botellas, frutas y verduras) como residuos.

Regla de descarte (una detección de residuo se elimina si cumple alguna):
  1. Su caja se solapa con la de una persona: IoU mayor que ``iou_threshold``.
  2. Su caja contiene casi toda la caja de una persona: el residuo "cubre" a la
     persona (caso típico del falso positivo gigante sobre el cuerpo).
Un objeto sostenido frente a una persona queda dentro de la caja de ella pero es
mucho más pequeño, por lo que normalmente no se descarta. Si el objeto es grande
respecto a la persona (muy cerca de la cámara), la regla 1 puede descartarlo:
por eso los umbrales son configurables.
"""

from __future__ import annotations

import numpy as np

from src.vision import Detection, ObjectDetector

PERSON_LABEL = "person"
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_CONTAINMENT_THRESHOLD = 0.8  # fracción de la persona cubierta por el residuo
DEFAULT_REFRESH_FRAMES = 3  # cada cuántos fotogramas se vuelve a buscar personas
PERSON_SCORE_THRESHOLD = 0.35  # confianza mínima para considerar que hay una persona
# Clases COCO que se aceptan como residuos porque COCO las detecta con fiabilidad.
DEFAULT_COMPLEMENT_LABELS = ("bottle", "banana", "apple", "orange", "broccoli", "carrot")
# Si una detección del modelo de residuos con al menos esta confianza coincide con
# una de COCO (IoU > DUPLICATE_IOU), se conserva la del modelo de residuos.
DEFAULT_COMPLEMENT_FLOOR = 0.5
DUPLICATE_IOU = 0.5

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def box_iou(a: Box, b: Box) -> float:
    """Intersección sobre unión entre dos cajas."""
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def covered_fraction(inner: Box, outer: Box) -> float:
    """Qué fracción del área de ``inner`` queda dentro de ``outer``."""
    area = _area(inner)
    return _intersection(inner, outer) / area if area > 0 else 0.0


def is_person_false_positive(
    box: Box,
    person_boxes: list[Box],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> bool:
    """True si la caja de un residuo corresponde a una persona (ver reglas arriba)."""
    for person in person_boxes:
        if box_iou(box, person) > iou_threshold:
            return True
        if covered_fraction(person, box) >= containment_threshold:
            return True
    return False


class PersonFilteredDetector:
    """Detector de residuos que descarta personas y se complementa con COCO.

    Sin clases complementarias, para no duplicar el costo en cada fotograma:
      * el detector COCO solo se ejecuta cuando el de residuos encontró algo
        (si no hay nada que filtrar, no hace falta);
      * sus resultados se reutilizan durante ``refresh_frames`` fotogramas.
    Con clases complementarias, COCO se ejecuta en cada fotograma, porque sus
    detecciones también son resultados y sus cajas deben estar al día.
    """

    def __init__(
        self,
        waste_detector: ObjectDetector,
        person_detector: ObjectDetector,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
        refresh_frames: int = DEFAULT_REFRESH_FRAMES,
        complement_labels: tuple[str, ...] = (),
        complement_floor: float = DEFAULT_COMPLEMENT_FLOOR,
    ) -> None:
        if refresh_frames < 1:
            raise ValueError("refresh_frames must be at least 1")
        missing = [label for label in complement_labels if label not in person_detector.labels]
        if missing:
            raise ValueError(f"Complement labels not in the COCO model: {missing}")
        self._complement_labels = tuple(complement_labels)
        self._complement_floor = complement_floor
        self._waste = waste_detector
        self._person = person_detector
        self._iou_threshold = iou_threshold
        self._containment_threshold = containment_threshold
        self._refresh_frames = refresh_frames
        # Se inicia "vencido" para que la primera detección sí busque personas.
        self._frames_since_refresh = refresh_frames
        self._person_boxes: list[Box] = []
        self.last_discarded = 0  # cuántas detecciones descartó la última llamada

    @property
    def labels(self) -> tuple[str, ...]:
        """Etiquetas que puede producir: las del modelo de residuos y las complementarias."""
        extra = tuple(l for l in self._complement_labels if l not in self._waste.labels)
        return self._waste.labels + extra

    def invalidate_cache(self) -> None:
        """Obliga a buscar personas de nuevo en la próxima detección.

        Hay que llamarlo cuando la siguiente imagen no es continuación de la
        anterior (por ejemplo, otra foto o al encender la cámara): las personas
        guardadas pertenecen a una escena distinta.
        """
        self._frames_since_refresh = self._refresh_frames

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Detecta residuos, elimina los que son personas y añade los de COCO."""
        waste = self._waste.detect(image_bgr)
        self._frames_since_refresh += 1
        self.last_discarded = 0

        refresh_due = self._frames_since_refresh >= self._refresh_frames
        coco: list[Detection] = []
        if self._complement_labels or (waste and refresh_due):
            coco = self._person.detect(image_bgr)
            self._person_boxes = [d.box for d in coco if d.label == PERSON_LABEL]
            self._frames_since_refresh = 0

        kept = [
            detection
            for detection in waste
            if not is_person_false_positive(
                detection.box,
                self._person_boxes,
                self._iou_threshold,
                self._containment_threshold,
            )
        ]
        self.last_discarded = len(waste) - len(kept)
        if not self._complement_labels:
            return kept
        return self._merge_complements(kept, coco)

    def _merge_complements(
        self, waste: list[Detection], coco: list[Detection]
    ) -> list[Detection]:
        """Añade las detecciones COCO complementarias evitando duplicar objetos.

        Si el modelo de residuos ya reconoce el mismo objeto con confianza
        suficiente, se conserva esa detección. Si solo tiene una conjetura débil
        sobre el mismo objeto, se prefiere la de COCO y se retira la débil.
        """
        confident = [d for d in waste if d.confidence >= self._complement_floor]
        complements = [
            d
            for d in coco
            if d.label in self._complement_labels
            and not any(box_iou(d.box, w.box) > DUPLICATE_IOU for w in confident)
        ]
        weak_duplicates = [
            w
            for w in waste
            if w.confidence < self._complement_floor
            and any(box_iou(w.box, c.box) > DUPLICATE_IOU for c in complements)
        ]
        merged = [w for w in waste if w not in weak_duplicates] + complements
        return sorted(merged, key=lambda d: -d.confidence)
