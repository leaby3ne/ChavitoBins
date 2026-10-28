"""Detección local con modelos YOLO exportados por Ultralytics (v8/v11) a ONNX.

Formato de salida (verificado con el modelo de residuos ``waste_yolo11n.onnx``):
un tensor (1, 4 + C, 8400) para una entrada de 640x640. A diferencia de YOLOX,
no hay columna de "objectness" ni hay que aplicar rejilla: las primeras 4 filas
ya son ``cx, cy, w, h`` en píxeles de la imagen de entrada y las C siguientes
son el puntaje de cada clase. Los nombres de clase viajan dentro del propio
archivo ONNX (metadato ``names``), así que no se escriben a mano.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.preprocessing import letterbox
from src.vision import (
    BOX_VALUES,
    Detection,
    InferenceError,
    ModelLoadError,
    non_max_suppression,
)

NAMES_METADATA_KEY = "names"


def parse_class_names(raw_names: str) -> tuple[str, ...]:
    """Convierte el metadato ``names`` (por ejemplo "{0: 'PAPER', 1: 'GLASS'}") en una tupla.

    Se usa ``ast.literal_eval`` (solo acepta literales, no ejecuta código) y se
    exige que los índices sean 0..N-1 sin huecos. Las etiquetas se pasan a
    minúsculas para que coincidan con las del archivo de reglas.
    """
    try:
        names = ast.literal_eval(raw_names)
    except (ValueError, SyntaxError) as exc:
        raise ModelLoadError(f"Cannot parse the model class names: {exc}") from exc
    if not isinstance(names, dict) or not names:
        raise ModelLoadError("The model class names must be a non-empty mapping.")
    if sorted(names) != list(range(len(names))):
        raise ModelLoadError("The model class indices must be 0..N-1 without gaps.")
    if not all(isinstance(name, str) and name.strip() for name in names.values()):
        raise ModelLoadError("Every model class name must be a non-empty string.")
    return tuple(names[index].strip().lower() for index in range(len(names)))


def is_ultralytics_export(metadata: Mapping[str, str]) -> bool:
    """True si los metadatos del ONNX indican un modelo exportado por Ultralytics."""
    return NAMES_METADATA_KEY in metadata and metadata.get("task", "detect") == "detect"


def decode_predictions(
    raw: np.ndarray,
    ratio: float,
    original_size: tuple[int, int],
    score_threshold: float,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte la salida cruda (4 + C, N) en cajas, puntajes e ids de clase.

    Devuelve arreglos ya filtrados por ``score_threshold``. Las cajas son
    (x1, y1, x2, y2) en píxeles de la imagen original, recortadas a sus bordes.
    """
    expected_rows = BOX_VALUES + class_count
    if raw.ndim != 2 or raw.shape[0] != expected_rows:
        raise InferenceError(
            f"Unexpected model output shape {raw.shape}; "
            f"expected ({expected_rows}, N)."
        )

    predictions = raw.T  # (N, 4 + C): una fila por celda candidata
    class_scores = predictions[:, BOX_VALUES:]
    class_ids = class_scores.argmax(axis=1)
    # No hay "objectness": la confianza es directamente el mejor puntaje de clase.
    confidences = class_scores[np.arange(predictions.shape[0]), class_ids]

    keep = confidences >= score_threshold
    centers_sizes = predictions[keep, :BOX_VALUES]
    confidences, class_ids = confidences[keep], class_ids[keep]

    centers, sizes = centers_sizes[:, :2], centers_sizes[:, 2:]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    # Se deshace el redimensionado del letterbox para volver a la imagen original.
    boxes /= ratio
    height, width = original_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes, confidences, class_ids


class YoloDetector:
    """Ejecuta un modelo YOLO (formato Ultralytics) en ONNX, sin conexión a internet."""

    def __init__(
        self,
        model_path: Path,
        input_size: int = 640,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._session = self._load_session(Path(model_path))
        self._input_name = self._session.get_inputs()[0].name
        self._labels = self._read_labels(self._session)
        self._check_output_shape(self._session, len(self._labels))

    @property
    def labels(self) -> tuple[str, ...]:
        """Etiquetas del modelo, leídas de sus metadatos y en minúsculas."""
        return self._labels

    def _load_session(self, model_path: Path) -> ort.InferenceSession:
        if not model_path.is_file():
            raise ModelLoadError(f"Model file not found: {model_path}")
        try:
            # Solo CPU: es lo que garantiza que funcione en cualquier equipo.
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime lanza varios tipos de error
            raise ModelLoadError(f"Cannot load model '{model_path}': {exc}") from exc

        expected_shape = [1, 3, self._input_size, self._input_size]
        actual_shape = session.get_inputs()[0].shape
        if actual_shape != expected_shape:
            raise ModelLoadError(
                f"Model input shape is {actual_shape}, expected {expected_shape}. "
                "Use a model exported for the same input size."
            )
        return session

    @staticmethod
    def _read_labels(session: ort.InferenceSession) -> tuple[str, ...]:
        metadata = session.get_modelmeta().custom_metadata_map
        if NAMES_METADATA_KEY not in metadata:
            raise ModelLoadError(
                "The model has no 'names' metadata; it does not look like an "
                "Ultralytics export."
            )
        return parse_class_names(metadata[NAMES_METADATA_KEY])

    @staticmethod
    def _check_output_shape(session: ort.InferenceSession, class_count: int) -> None:
        """Comprueba al cargar que las filas de salida coinciden con las clases."""
        shape = session.get_outputs()[0].shape
        expected_rows = BOX_VALUES + class_count
        if len(shape) != 3 or shape[1] != expected_rows:
            raise ModelLoadError(
                f"Model output shape is {shape}; expected (1, {expected_rows}, N) "
                f"for {class_count} classes."
            )

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Detecta objetos en una imagen BGR y los devuelve de mejor a peor puntaje."""
        prepared = letterbox(image_bgr, self._input_size, normalize_rgb=True)
        try:
            raw = self._session.run(None, {self._input_name: prepared.tensor})[0]
        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc

        # raw[0]: se descarta la dimensión de lote (siempre hay una sola imagen).
        boxes, scores, class_ids = decode_predictions(
            raw[0],
            prepared.ratio,
            prepared.original_size,
            self._score_threshold,
            len(self._labels),
        )
        keep = non_max_suppression(boxes, scores, class_ids, self._iou_threshold)
        return [
            Detection(
                class_id=int(class_ids[i]),
                label=self._labels[int(class_ids[i])],
                confidence=float(scores[i]),
                box=tuple(float(v) for v in boxes[i]),
            )
            for i in keep
        ]
