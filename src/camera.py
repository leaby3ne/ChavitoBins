"""Captura de fotogramas desde una cámara web con OpenCV."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import TracebackType
from typing import Any

import cv2
import numpy as np

DEFAULT_CAMERA_INDEX = 0


class CameraError(RuntimeError):
    """Se lanza si la cámara no se puede abrir o deja de entregar imágenes."""


def default_backend() -> int:
    """Backend de OpenCV por defecto.

    En Windows se usa DSHOW, que funcionó en las pruebas (scripts/check_camera.py);
    en otros sistemas se deja que OpenCV elija.
    """
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class Camera:
    """Cámara web con apertura explícita y liberación garantizada.

    Se recomienda usarla con ``with`` para que la cámara se libere siempre,
    incluso si ocurre un error durante el procesamiento.
    """

    def __init__(
        self,
        index: int = DEFAULT_CAMERA_INDEX,
        backend: int | None = None,
        # La fábrica permite sustituir cv2.VideoCapture por un doble en las
        # pruebas, para no depender de una cámara física.
        capture_factory: Callable[[int, int], Any] = cv2.VideoCapture,
    ) -> None:
        self._index = index
        self._backend = default_backend() if backend is None else backend
        self._capture_factory = capture_factory
        self._capture: Any | None = None

    def open(self) -> None:
        """Abre la cámara. Lanza CameraError si no está disponible."""
        capture = self._capture_factory(self._index, self._backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"No se pudo abrir la cámara (índice {self._index}). Revisa que esté "
                "conectada, que Windows permita su uso y que otra aplicación no la "
                "esté usando."
            )
        self._capture = capture

    def read(self) -> np.ndarray:
        """Devuelve el siguiente fotograma como arreglo BGR uint8."""
        if self._capture is None:
            raise CameraError("La cámara no está abierta; llama a open() primero.")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise CameraError("La cámara no entregó ningún fotograma.")
        return frame

    def release(self) -> None:
        """Libera la cámara. Es seguro llamarlo varias veces."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
