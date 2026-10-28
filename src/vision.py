"""Detección local de objetos con un modelo YOLOX exportado a ONNX.

Formato de salida (verificado con el ``yolox_s.onnx`` oficial): un tensor
crudo de forma (1, 8400, 85) para una entrada de 640x640. Cada fila es
``[cx, cy, w, h, objectness, 80 puntajes de clase]`` y todavía está codificada
respecto a la rejilla de anclas, así que hay que decodificarla con la rejilla
y los strides antes de usarla.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from src.coco_labels import COCO_CLASSES
from src.preprocessing import letterbox

# YOLOX predice en tres escalas; cada stride es cuántos píxeles de la imagen
# de entrada cubre una celda de la rejilla en esa escala.
STRIDES = (8, 16, 32)
DEFAULT_INPUT_SIZE = 640  # tamaño de entrada de la mayoría de los modelos YOLO
BOX_VALUES = 4  # cx, cy, w, h
OBJECTNESS_VALUES = 1  # probabilidad de que la celda contenga algún objeto


class ModelLoadError(RuntimeError):
    """Se lanza si el archivo del modelo falta, no se lee o es incompatible."""


class InferenceError(RuntimeError):
    """Se lanza si falla la inferencia o la salida del modelo no es la esperada."""


@dataclass(frozen=True)
class Detection:
    """Un objeto detectado, con su caja en píxeles de la imagen original."""

    class_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


class ObjectDetector(Protocol):
    """Lo que el resto de la app necesita de un detector, sea del modelo que sea."""

    @property
    def labels(self) -> tuple[str, ...]:
        """Etiquetas que el modelo puede producir."""

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Detecta objetos en una imagen BGR."""


def build_grids(input_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve las rejillas y los strides que corresponden a las filas de salida.

    Con entrada de 640: 80x80 + 40x40 + 20x20 = 8400 celdas, que coincide con
    las 8400 filas que produce el modelo.
    """
    grids, strides = [], []
    for stride in STRIDES:
        cells = input_size // stride
        xs, ys = np.meshgrid(np.arange(cells), np.arange(cells))
        grid = np.stack((xs, ys), axis=2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((grid.shape[0], 1), stride))
    return (
        np.concatenate(grids).astype(np.float32),
        np.concatenate(strides).astype(np.float32),
    )


def decode_predictions(
    raw: np.ndarray,
    grid: np.ndarray,
    strides: np.ndarray,
    ratio: float,
    original_size: tuple[int, int],
    score_threshold: float,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte la salida cruda (N, 5+C) en cajas, puntajes e ids de clase.

    Devuelve arreglos ya filtrados por ``score_threshold``. Las cajas son
    (x1, y1, x2, y2) en píxeles de la imagen original, recortadas a sus bordes.
    """
    expected = BOX_VALUES + OBJECTNESS_VALUES + len(labels)
    # Validar la forma evita que un modelo distinto produzca resultados
    # absurdos sin avisar.
    if raw.ndim != 2 or raw.shape[1] != expected or raw.shape[0] != grid.shape[0]:
        raise InferenceError(
            f"Unexpected model output shape {raw.shape}; "
            f"expected ({grid.shape[0]}, {expected})."
        )

    # El modelo predice el centro como desplazamiento dentro de la celda, y el
    # tamaño en escala logarítmica; así se pasan a píxeles del tensor de entrada.
    centers = (raw[:, :2] + grid) * strides
    sizes = np.exp(raw[:, 2:4]) * strides

    # Confianza final = probabilidad de objeto x probabilidad de la mejor clase.
    class_scores = raw[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    confidences = raw[:, 4] * class_scores[np.arange(raw.shape[0]), class_ids]

    # Se filtra antes de armar las cajas: de 8400 filas casi todas se descartan.
    keep = confidences >= score_threshold
    centers, sizes = centers[keep], sizes[keep]
    confidences, class_ids = confidences[keep], class_ids[keep]

    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    # Se deshace el redimensionado del letterbox para volver a la imagen original.
    boxes /= ratio
    height, width = original_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes, confidences, class_ids


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU (intersección sobre unión) entre una caja y un arreglo de cajas."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + other_areas - intersection
    # El máximo evita dividir entre cero con cajas de área nula.
    return intersection / np.maximum(union, 1e-9)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """NMS por clase: elimina cajas duplicadas del mismo objeto.

    Devuelve los índices a conservar, de mayor a menor puntaje. Solo compara
    cajas de la misma clase, por lo que un mismo objeto puede seguir apareciendo
    con clases distintas (por ejemplo, "apple" y "orange" sobre una naranja).
    """
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        indices = indices[np.argsort(-scores[indices])]
        while indices.size:
            # Se conserva la mejor caja y se descartan las que se le solapan.
            best = indices[0]
            kept.append(int(best))
            if indices.size == 1:
                break
            overlaps = _iou(boxes[best], boxes[indices[1:]])
            indices = indices[1:][overlaps < iou_threshold]
    kept.sort(key=lambda index: -scores[index])
    return kept


class YoloxDetector:
    """Ejecuta un modelo YOLOX en ONNX sobre imágenes BGR, sin conexión a internet."""

    def __init__(
        self,
        model_path: Path,
        input_size: int = DEFAULT_INPUT_SIZE,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        labels: tuple[str, ...] = COCO_CLASSES,
    ) -> None:
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._labels = labels
        # La rejilla solo depende del tamaño de entrada: se calcula una vez.
        self._grid, self._strides = build_grids(input_size)
        self._session = self._load_session(Path(model_path))
        self._input_name = self._session.get_inputs()[0].name

    @property
    def labels(self) -> tuple[str, ...]:
        """Etiquetas que produce este modelo (las 80 clases COCO)."""
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

        # Si el modelo se exportó con otro tamaño, la rejilla no coincidiría.
        expected_shape = [1, 3, self._input_size, self._input_size]
        actual_shape = session.get_inputs()[0].shape
        if actual_shape != expected_shape:
            raise ModelLoadError(
                f"Model input shape is {actual_shape}, expected {expected_shape}. "
                "Use a model exported for the same input size."
            )
        return session

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Detecta objetos en una imagen BGR y los devuelve de mejor a peor puntaje."""
        prepared = letterbox(image_bgr, self._input_size)
        try:
            raw = self._session.run(None, {self._input_name: prepared.tensor})[0]
        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc

        # raw[0]: se descarta la dimensión de lote (siempre hay una sola imagen).
        boxes, scores, class_ids = decode_predictions(
            raw[0],
            self._grid,
            self._strides,
            prepared.ratio,
            prepared.original_size,
            self._score_threshold,
            self._labels,
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
