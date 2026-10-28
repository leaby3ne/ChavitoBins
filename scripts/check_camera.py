"""Diagnóstico de cámara: prueba qué índices y backends de OpenCV funcionan.

Uso (desde la raíz del proyecto):
    python -m scripts.check_camera

Requiere opencv-python. Guarda una foto de la primera combinación que funcione
en ``camera_test.jpg`` para comprobar que la imagen se ve bien.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# En Windows, DSHOW y MSMF son los dos backends habituales; AUTO deja que
# OpenCV elija. Probar los tres ayuda a encontrar el que funciona en cada equipo.
BACKENDS = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "AUTO": cv2.CAP_ANY}
MAX_CAMERA_INDEX = 3  # se prueban los índices 0..3
WARMUP_FRAMES = 5  # los primeros fotogramas suelen salir oscuros (autoexposición)
MEASURED_FRAMES = 30
SNAPSHOT_PATH = Path("camera_test.jpg")


@dataclass(frozen=True)
class CameraProbe:
    """Resultado de una combinación índice + backend que sí funcionó."""

    index: int
    backend: str
    width: int
    height: int
    measured_fps: float
    frame: np.ndarray


def probe_camera(index: int, backend_name: str, backend_id: int) -> CameraProbe | None:
    """Intenta abrir una cámara y leer fotogramas. Devuelve None si no funciona."""
    capture = cv2.VideoCapture(index, backend_id)
    try:
        if not capture.isOpened():
            return None

        for _ in range(WARMUP_FRAMES):
            capture.read()

        frame = None
        read_ok = 0
        start = time.perf_counter()
        for _ in range(MEASURED_FRAMES):
            success, current = capture.read()
            if success and current is not None:
                frame = current
                read_ok += 1
        elapsed = time.perf_counter() - start

        if frame is None:
            return None
        height, width = frame.shape[:2]
        return CameraProbe(
            index=index,
            backend=backend_name,
            width=width,
            height=height,
            measured_fps=read_ok / elapsed if elapsed > 0 else 0.0,
            frame=frame,
        )
    finally:
        capture.release()  # siempre se libera la cámara, aunque falle algo


def main() -> int:
    """Prueba todas las combinaciones y recomienda la primera que funciona."""
    print(f"OpenCV {cv2.__version__}")
    print(f"Probando índices 0..{MAX_CAMERA_INDEX} con {', '.join(BACKENDS)}\n")

    working: list[CameraProbe] = []
    for index in range(MAX_CAMERA_INDEX + 1):
        for backend_name, backend_id in BACKENDS.items():
            result = probe_camera(index, backend_name, backend_id)
            if result is None:
                print(f"  índice {index} + {backend_name:<5}: no disponible")
            else:
                working.append(result)
                print(
                    f"  índice {index} + {backend_name:<5}: OK  "
                    f"{result.width}x{result.height}  ~{result.measured_fps:.0f} fps"
                )

    if not working:
        print(
            "\nNo se encontró ninguna cámara. Revisa que esté conectada, que Windows "
            "permita a las aplicaciones de escritorio usarla (Configuración > "
            "Privacidad y seguridad > Cámara) y que otra app (Teams, Zoom, el "
            "navegador) no la esté usando."
        )
        return 1

    best = working[0]
    cv2.imwrite(str(SNAPSHOT_PATH), best.frame)
    print(
        f"\nUsar: índice {best.index} con {best.backend}. "
        f"Foto de prueba guardada en {SNAPSHOT_PATH}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
