"""Preparación inicial: descarga el modelo de residuos y las imágenes de prueba.

Requiere internet. Después de esto, ChavitoBins funciona sin conexión.

Uso (desde la raíz del proyecto):
    python -m scripts.download_model                   # residuos + filtro de personas + imágenes
    python -m scripts.download_model --con-yolox-s     # además, YOLOX-s (35 MB, más lento)
    python -m scripts.download_model --solo-residuos   # sin filtro de personas
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    """Un modelo descargable: dónde guardarlo, de dónde bajarlo y su suma SHA-256."""

    path: Path
    url: str
    sha256: str


# Modelo de residuos (YOLO11n de Ultralytics, 6 clases). La URL apunta a un
# commit concreto, no a la rama principal, para que el archivo no cambie sin avisar.
WASTE_MODEL = ModelFile(
    path=Path("models/waste_yolo11n.onnx"),
    url=(
        "https://raw.githubusercontent.com/shshodiyev162-max/Waste-detector/"
        "f5fa284c794c60325a88d739c83558e052fd7fa3/best.onnx"
    ),
    sha256="e38c11cdd4fc23ff9f6f2408df3f11df4908311fc24ab1a29da6933869e423cf",
)
# Modelo COCO ligero (80 clases, entrada 416): detecta personas para el filtro.
# Es mucho más rápido que YOLOX-s, por eso es el predeterminado del filtro.
YOLOX_NANO_MODEL = ModelFile(
    path=Path("models/yolox_nano.onnx"),
    url=(
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        "0.1.1rc0/yolox_nano.onnx"
    ),
    sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
)
# Modelo COCO más preciso y más lento: alternativa para el filtro y para comparar.
YOLOX_MODEL = ModelFile(
    path=Path("models/yolox_s.onnx"),
    url=(
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        "0.1.1rc0/yolox_s.onnx"
    ),
    sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
)

# Imágenes de prueba: ejemplos oficiales de YOLOX y Ultralytics (zidane.jpg
# tiene personas, útil para ver falsos positivos) y dos fotos de frutas de OpenCV.
SAMPLES = {
    Path("samples/dog.jpg"): (
        "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg"
    ),
    Path("samples/zidane.jpg"): (
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    ),
    Path("samples/apple.jpg"): (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/apple.jpg"
    ),
    Path("samples/orange.jpg"): (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/orange.jpg"
    ),
}


def sha256_of(path: Path) -> str:
    """Calcula la suma SHA-256 de un archivo leyéndolo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Descarga una URL a un archivo, creando las carpetas necesarias."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {url}")
    urllib.request.urlretrieve(url, destination)


def ensure_model(model: ModelFile) -> None:
    """Descarga el modelo si falta y comprueba su integridad."""
    if not model.path.is_file():
        download(model.url, model.path)
    actual = sha256_of(model.path)
    # Detecta descargas incompletas o archivos alterados.
    if actual != model.sha256:
        raise SystemExit(
            f"La suma SHA-256 de {model.path} no coincide.\n"
            f"  esperada: {model.sha256}\n  obtenida: {actual}\n"
            "Borra el archivo y vuelve a ejecutar este script."
        )
    print(f"Modelo OK: {model.path} (SHA-256 verificado)")


def main() -> None:
    """Descarga los modelos pedidos y las imágenes de prueba."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solo-residuos", action="store_true",
        help="No descarga ningún modelo COCO (se perderá el filtro de personas).",
    )
    parser.add_argument(
        "--con-yolox-s", action="store_true",
        help="Descarga también YOLOX-s (35 MB), alternativa más precisa y lenta.",
    )
    args = parser.parse_args()

    ensure_model(WASTE_MODEL)
    if not args.solo_residuos:
        ensure_model(YOLOX_NANO_MODEL)
    if args.con_yolox_s:
        ensure_model(YOLOX_MODEL)

    for path, url in SAMPLES.items():
        if not path.is_file():
            download(url, path)
        print(f"Imagen de prueba OK: {path}")


if __name__ == "__main__":
    main()
