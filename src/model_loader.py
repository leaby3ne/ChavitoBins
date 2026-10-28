"""Elige y arma los detectores según el formato del archivo ONNX."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

from src.person_filter import (
    DEFAULT_COMPLEMENT_LABELS,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_REFRESH_FRAMES,
    PERSON_LABEL,
    PERSON_SCORE_THRESHOLD,
    PersonFilteredDetector,
)
from src.vision import DEFAULT_INPUT_SIZE, ModelLoadError, ObjectDetector, YoloxDetector
from src.vision_yolo import YoloDetector, is_ultralytics_export


def create_detector(model_path: Path, score_threshold: float = 0.25) -> ObjectDetector:
    """Crea el detector que corresponde al modelo indicado.

    Si el ONNX trae el metadato ``names`` (exportación de Ultralytics, como el
    modelo de residuos) se usa ``YoloDetector``; si no, se asume YOLOX con
    las clases COCO. La decisión se basa en el contenido del archivo, no en su nombre.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelLoadError(f"Model file not found: {model_path}")
    try:
        # Se abre una vez solo para leer los metadatos; el detector vuelve a
        # cargarlo. El costo es menor que mantener sesiones compartidas.
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime lanza varios tipos de error
        raise ModelLoadError(f"Cannot load model '{model_path}': {exc}") from exc

    # El tamaño de entrada se lee del modelo: YOLOX-nano usa 416 y los demás 640.
    shape = session.get_inputs()[0].shape
    input_size = shape[2] if isinstance(shape[2], int) else DEFAULT_INPUT_SIZE

    if is_ultralytics_export(session.get_modelmeta().custom_metadata_map):
        return YoloDetector(model_path, input_size=input_size, score_threshold=score_threshold)
    return YoloxDetector(model_path, input_size=input_size, score_threshold=score_threshold)


def allowed_rule_labels(detector: ObjectDetector) -> tuple[str, ...]:
    """Etiquetas que pueden aparecer en el archivo de reglas.

    Son las del detector más las clases COCO complementarias por defecto: así el
    mismo archivo de reglas sirve con el complemento activado o desactivado
    (sin complemento esas reglas simplemente no se aplican), y una etiqueta mal
    escrita sigue detectándose como error.
    """
    extra = tuple(l for l in DEFAULT_COMPLEMENT_LABELS if l not in detector.labels)
    return tuple(detector.labels) + extra


def create_detector_stack(
    waste_model: Path,
    person_model: Path | None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    refresh_frames: int = DEFAULT_REFRESH_FRAMES,
    complement_labels: tuple[str, ...] = DEFAULT_COMPLEMENT_LABELS,
) -> ObjectDetector:
    """Detector de residuos, con filtro de personas y complemento si se da un modelo COCO.

    Con ``person_model=None`` se devuelve solo el detector de residuos. Las
    ``complement_labels`` son clases COCO que también se aceptan como residuos
    (una tupla vacía desactiva el complemento y deja solo el filtro).
    """
    waste_detector = create_detector(waste_model)
    if person_model is None:
        return waste_detector

    person_detector = create_detector(person_model, score_threshold=PERSON_SCORE_THRESHOLD)
    # El filtro solo sirve si el segundo modelo realmente reconoce personas.
    if PERSON_LABEL not in person_detector.labels:
        raise ModelLoadError(
            f"The person-filter model '{person_model}' has no '{PERSON_LABEL}' class."
        )
    return PersonFilteredDetector(
        waste_detector,
        person_detector,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
        refresh_frames=refresh_frames,
        complement_labels=tuple(complement_labels),
    )
