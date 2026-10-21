"""Preparación inicial: descarga el modelo YOLOX-s ONNX oficial y las imágenes de prueba.

Requiere internet. Después de esto, ChavitoBins funciona sin conexión.

Uso (desde la raíz del proyecto):
    python -m scripts.download_model
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_s.onnx"
)
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
MODEL_PATH = Path("models/yolox_s.onnx")

# Imágenes de prueba: la imagen de demostración oficial de YOLOX y dos fotos de frutas de los ejemplos de OpenCV.
SAMPLES = {
    Path("samples/dog.jpg"): (
        "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/"
        "main/assets/dog.jpg"
    ),
    Path("samples/apple.jpg"): (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/apple.jpg"
    ),
    Path("samples/orange.jpg"): (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/orange.jpg"
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)


def main() -> None:
    if not MODEL_PATH.is_file():
        download(MODEL_URL, MODEL_PATH)
    actual = sha256_of(MODEL_PATH)
    # Comprueba la integridad: detecta descargas incompletas o archivos alterados.
    if actual != MODEL_SHA256:
        raise SystemExit(
            f"Checksum mismatch for {MODEL_PATH}.\n"
            f"  expected: {MODEL_SHA256}\n  actual:   {actual}\n"
            "Delete the file and run this script again."
        )
    print(f"Model OK: {MODEL_PATH} (SHA-256 verified)")

    for path, url in SAMPLES.items():
        if not path.is_file():
            download(url, path)
        print(f"Sample image OK: {path}")


if __name__ == "__main__":
    main()
