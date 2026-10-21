"""Preparación de imágenes para el modelo de detección.

Convención interna: las imágenes son arreglos ``uint8`` de forma
(alto, ancho, 3) con los canales en orden BGR (la convención de OpenCV).
YOLOX se entrenó con imágenes BGR y valores crudos 0-255, así que NO se
intercambian canales ni se normaliza.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# Valor de gris con el que YOLOX rellena los bordes; es el que se usó en el
# entrenamiento, por eso no conviene cambiarlo.
PAD_VALUE = 114


class ImageLoadError(RuntimeError):
    """Se lanza cuando no se puede leer un archivo de imagen."""


@dataclass(frozen=True)
class PreprocessedImage:
    """Tensor listo para el modelo y los datos para volver a las coordenadas originales."""

    tensor: np.ndarray  # float32, forma (1, 3, tamaño, tamaño)
    ratio: float  # tamaño redimensionado / tamaño original
    original_size: tuple[int, int]  # (alto, ancho) de la imagen original


def load_image_bgr(path: Path) -> np.ndarray:
    """Lee un archivo de imagen y lo devuelve como arreglo BGR uint8."""
    try:
        with Image.open(path) as image:
            # convert("RGB") unifica formatos (PNG con transparencia, escala
            # de grises, etc.) para que siempre haya exactamente 3 canales.
            rgb = np.asarray(image.convert("RGB"))
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(f"Cannot read image '{path}': {exc}") from exc
    # Pillow entrega RGB; se invierte el último eje para obtener BGR.
    return np.ascontiguousarray(rgb[:, :, ::-1])


def letterbox(image_bgr: np.ndarray, input_size: int) -> PreprocessedImage:
    """Redimensiona conservando la proporción, rellena hasta un cuadrado y arma el tensor.

    Se usa "letterbox" en lugar de estirar la imagen para no deformar los
    objetos, porque el modelo reconoce peor las formas distorsionadas.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected an image of shape (H, W, 3), got {image_bgr.shape}"
        )
    height, width = image_bgr.shape[:2]
    # La proporción más pequeña garantiza que la imagen quepa completa.
    ratio = min(input_size / height, input_size / width)
    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))

    # Redimensionar por canal no depende del orden de colores, por eso se
    # puede pasar un arreglo BGR a Pillow sin convertirlo.
    resized = np.asarray(
        Image.fromarray(image_bgr).resize((new_width, new_height), Image.BILINEAR)
    )
    # La imagen queda pegada arriba a la izquierda; el relleno va abajo/derecha.
    # Así las coordenadas se recuperan solo dividiendo entre ``ratio``.
    canvas = np.full((input_size, input_size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:new_height, :new_width] = resized

    # (alto, ancho, canal) -> (1, canal, alto, ancho), como espera el modelo.
    tensor = canvas.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return PreprocessedImage(
        tensor=np.ascontiguousarray(tensor),
        ratio=ratio,
        original_size=(height, width),
    )
