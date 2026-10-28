# Modelos de visión

ChavitoBins ejecuta **dos modelos ONNX locales**: uno de residuos (principal) y
uno COCO para el filtro de personas. Los archivos `.onnx` no se versionan en Git:

    python -m scripts.download_model                 # residuos + YOLOX-nano
    python -m scripts.download_model --con-yolox-s   # además, YOLOX-s
    python -m scripts.download_model --solo-residuos # sin filtro de personas

La descarga verifica la suma SHA-256 de cada archivo.

| Archivo | Rol | Clases | Entrada | Tamaño |
|---|---|---|---|---|
| `waste_yolo11n.onnx` | Detección de residuos (principal) | 6 materiales | 640x640, RGB, 0-1 | 10 MB |
| `yolox_nano.onnx` | Filtro de personas (predeterminado) | 80 COCO | 416x416, BGR, 0-255 | 3.6 MB |
| `yolox_s.onnx` | Filtro de personas (alternativa) y modelo COCO anterior | 80 COCO | 640x640, BGR, 0-255 | 36 MB |

## `waste_yolo11n.onnx` — residuos

- **Qué es:** YOLO11n entrenado para detectar materiales de residuos.
- **Etiquetas** (leídas del propio ONNX): `biodegradable`, `cardboard`, `glass`,
  `metal`, `paper`, `plastic`. En las reglas se traducen a: orgánico, cartón,
  vidrio, metal, papel, plástico.
- **Origen:** repositorio `shshodiyev162-max/Waste-detector`, commit
  `f5fa284c794c60325a88d739c83558e052fd7fa3`, archivo `best.onnx`.
- **SHA-256:** `e38c11cdd4fc23ff9f6f2408df3f11df4908311fc24ab1a29da6933869e423cf`
- **Salida verificada:** `output0`, `[1, 10, 8400]` = 4 valores de caja + 6
  clases, sin columna de objectness (formato Ultralytics).
- **Métricas declaradas por el autor** (validación, no pruebas reales): mAP50 de
  0.899 en total; `biodegradable` es la clase más débil (recall 0.56).

### Licencia: discrepancia que debes conocer

El repositorio declara **licencia MIT**, pero los metadatos del propio archivo
ONNX dicen **AGPL-3.0 (Ultralytics)**, porque el modelo se entrenó y exportó con
Ultralytics. Para este proyecto (uso **académico y local**, sin distribuir el
modelo ni ofrecerlo como servicio en red) no impide su uso, pero **la licencia
efectiva no está confirmada**: antes de publicar o redistribuir el modelo, hay
que consultar con el docente o revisar las licencias de los conjuntos de datos
de Roboflow con los que se entrenó. Se cita la procedencia en esta página.

### Limitaciones verificadas

- **No existe la clase "no aprovechable".** El recipiente negro se alcanza con la
  pregunta de verificación, no por detección.
- **Siempre elige un material.** Con fotos de un perro y de personas inventó
  `glass` (0.69) y `paper` (0.77–0.88): de ahí el filtro de personas.
- **Sensible a la distancia:** se entrenó con objetos a distancia media.
- **Falla con plásticos transparentes en la mano:** en una botella de limpiapisos
  sostenida a mano dio `plastic` 0.24 (COCO la detectó con 0.75 y 0.90). Ver
  `docs/test-plan.md`, sección 5.2.
- **Confunde vasos de vidrio con metal** (0.88 `metal`, 0.00 `glass` en un vaso de
  prueba), aunque reconoce bien botellas y frascos (119 de 120). Ver
  `docs/test-plan.md`, sección 5.1.

## `yolox_nano.onnx` y `yolox_s.onnx` — COCO

- **Qué son:** YOLOX preentrenado en COCO (80 clases). Aquí cumplen dos funciones en
  el detector híbrido: detectar `person` (ID 0) para descartar falsos positivos del
  modelo de residuos, y **aportar** las clases `bottle`, `banana`, `apple`, `orange`,
  `broccoli` y `carrot`, que COCO detecta mejor que el modelo de residuos.
  `yolox_s` también permite usar COCO como detector principal
  (`data/waste_rules_coco.json`).
- **Origen:** release `0.1.1rc0` de https://github.com/Megvii-BaseDetection/YOLOX
- **Licencia:** Apache-2.0 según el repositorio oficial.
- **SHA-256:**
  - `yolox_nano.onnx`: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
  - `yolox_s.onnx`: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`
- **Salida:** `[1, 3549, 85]` (nano) y `[1, 8400, 85]` (s), sin decodificar:
  requiere rejilla y strides (implementado en `src/vision.py`).
- **Por qué nano es el predeterminado:** ambos encuentran a las personas de las
  imágenes de prueba, pero nano es mucho más rápido (ver `docs/test-plan.md`).
  `yolox_s` es más preciso en general; se puede usar con
  `--modelo-personas models/yolox_s.onnx`.

## Ámbito académico

Todos los modelos se usan localmente, sin enviar imágenes a servidores. Se
declara la procedencia de cada uno y las licencias tal como las indican sus
fuentes; donde hay discrepancia (modelo de residuos), se señala expresamente.
