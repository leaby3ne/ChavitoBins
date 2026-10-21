# Modelo de visión

- **Modelo:** YOLOX-s (preentrenado en COCO, 80 clases), formato ONNX.
- **Origen:** release oficial `0.1.1rc0` de https://github.com/Megvii-BaseDetection/YOLOX
- **Licencia:** Apache-2.0 (según el repositorio oficial). Revisa el LICENSE del repositorio antes de redistribuir los pesos.
- **SHA-256 del archivo:** `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`
- **Entrada verificada:** `images`, `[1, 3, 640, 640]`, float32, BGR, valores 0–255 sin normalizar.
- **Salida verificada:** `output`, `[1, 8400, 85]`, sin decodificar (requiere rejilla y strides).

El archivo `.onnx` no se versiona en Git. Para obtenerlo:

    python -m scripts.download_model
