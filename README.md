# ChavitoBins

Sistema local de reconocimiento visual de residuos que orienta al usuario sobre
en qué caneca depositarlos, según el código de colores de la **Resolución 2184
de 2019** del Ministerio de Ambiente de Colombia (verde: orgánicos
aprovechables, blanca: aprovechables, negra: no aprovechables).

Una cámara web (o una imagen) alimenta un modelo de visión artificial que se
ejecuta **en el propio equipo**; la decisión de en qué caneca va cada objeto la
toma un motor de **reglas explícitas** (archivo JSON), no una IA. El sistema
**no usa APIs externas, no usa LLMs y no envía imágenes a ningún servidor**.

> **Alcance.** ChavitoBins es un prototipo académico (MVP). Reconoce 6 materiales
> y no pretende identificar cualquier residuo. Ver [Limitaciones](#limitaciones).

## Contenido

1. [Cómo funciona](#cómo-funciona)
2. [Estado real de la implementación](#estado-real-de-la-implementación)
3. [Arquitectura](#arquitectura)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Tecnologías y requisitos](#tecnologías-y-requisitos)
6. [Instalación](#instalación)
7. [Ejecución y uso](#ejecución-y-uso)
8. [Reglas y código de colores](#reglas-y-código-de-colores)
9. [Funcionamiento sin conexión](#funcionamiento-sin-conexión)
10. [Pruebas](#pruebas)
11. [Personalización](#personalización)
12. [Limitaciones](#limitaciones)
13. [Modelos, licencias y fuentes](#modelos-licencias-y-fuentes)
14. [Guion sugerido para la demostración](#guion-sugerido-para-la-demostración)

## Cómo funciona

```
 Captura          Visión              Filtro y complemento  Clasificador        Interfaz
 (camera.py) ──►  (vision_yolo.py) ──► COCO (person_filter) ──► (classifier.py) ──► (gui.py / HUD)
 Webcam o         YOLO11n de          YOLOX-nano (COCO):     Reglas JSON         Cajas + panel
 imagen           residuos            descarta personas y    (knowledge_base)    con la caneca,
                  (6 materiales)      añade botellas,        + preguntas         advertencias y
                                      frutas y verduras      de verificación     preguntas sí/no
                                      ▲                                          ▲
                                      └── stabilizer.py: la recomendación aparece tras 5
                                          fotogramas seguidos y no se repite mientras el
                                          objeto siga en escena
```

1. **Captura.** `Camera` abre la webcam con OpenCV (o se carga una imagen).
2. **Visión.** El modelo de residuos (`waste_yolo11n.onnx`, ONNX Runtime en CPU)
   devuelve cajas, material (`biodegradable`, `cardboard`, `glass`, `metal`,
   `paper`, `plastic`) y confianza.
3. **Filtro y complemento COCO.** El modelo de residuos siempre elige un material,
   incluso frente a una persona: un segundo detector (YOLOX-nano con clases COCO)
   busca personas y se descartan las detecciones de residuos que se solapan con
   una (IoU > 0.3) o que cubren casi toda la persona. Además, el modelo de
   residuos falla con algunos objetos cotidianos (una botella transparente en la
   mano salió con 0.24 de confianza), mientras que COCO los detecta con
   seguridad: por eso también se aceptan las clases COCO `bottle`, `banana`,
   `apple`, `orange`, `broccoli` y `carrot`. Si el modelo de residuos ya
   reconoce el mismo objeto con confianza suficiente, se conserva su detección.
4. **Clasificación.** `WasteClassifier` busca la regla del material en
   `data/waste_rules.json`. Con confianza insuficiente o sin regla, **no da
   categoría**: informa de la incertidumbre.
5. **Recipiente negro.** La cámara no puede ver si un plástico es un empaque
   metalizado de snacks o está sucio con comida. Para esos casos las reglas
   incluyen una **pregunta de sí/no**; con "Sí" la recomendación pasa al
   recipiente negro.
6. **Interfaz.** Ventana Tkinter con la imagen y un panel HUD al costado (no tapa
   el encuadre ni las cajas) con la franja del color de la caneca.

## Estado real de la implementación

| Componente | Estado | Verificación |
|---|---|---|
| Detección con modelo de residuos (imagen y cámara) | Implementado | Probado con imágenes reales; cámara validada con fotogramas simulados |
| Filtro híbrido de personas y complemento COCO (botella, frutas, verduras) | Implementado | Probado con imágenes y pruebas automáticas; **falta probarlo con la webcam** |
| Reglas JSON validadas + clasificador | Implementado | Pruebas automáticas |
| Recipiente negro (por pregunta de verificación) | Implementado | Pruebas automáticas e interfaz; **falta probarlo con empaques reales** |
| Estabilización (5 fotogramas) | Implementado | Pruebas automáticas e interfaz con cámara simulada |
| Interfaz gráfica con HUD | Implementado | Verificada en Linux con servidor gráfico virtual y cámara simulada; **falta verla con la webcam y en Windows** |
| Funcionamiento sin conexión | Implementado | Verificado ejecutando la app sin acceso a red |
| Pruebas automáticas | 135 pruebas | Pasan en el entorno de desarrollo (Linux, Python 3.12). Con la instalación por defecto se ejecutan 132 y se omiten 3 que usan YOLOX-s (se activan con `--con-yolox-s`) |

Hechos verificados en el equipo del desarrollador en etapas anteriores: la
webcam funciona con OpenCV en Windows (backend DSHOW, 640x480, ~30 fps) y la
versión inicial del proyecto pasó sus pruebas en Python 3.14.7 (Linux) y 3.12
(Windows). La **versión actual** debe validarse en el equipo de la
demostración siguiendo [`docs/test-plan.md`](docs/test-plan.md).

## Arquitectura

| Módulo | Responsabilidad |
|---|---|
| `app.py` | Punto de entrada: arranca la interfaz o analiza una imagen por terminal; gestiona errores esperados |
| `src/camera.py` | Apertura de la webcam, lectura de fotogramas y liberación garantizada |
| `src/preprocessing.py` | Carga de imágenes y *letterbox* (escalado con relleno) para el modelo |
| `src/vision_yolo.py` | Detector para modelos Ultralytics (residuos): lee las etiquetas del propio ONNX y decodifica la salida |
| `src/vision.py` | Detector YOLOX (COCO), tipos comunes (`Detection`) y supresión de duplicados (NMS) |
| `src/model_loader.py` | Elige el detector según el contenido del ONNX y arma la cascada residuos + filtro |
| `src/person_filter.py` | Detector híbrido: descarta residuos que coinciden con una persona y añade botellas, frutas y verduras de COCO |
| `src/knowledge_base.py` | Carga y validación del JSON de reglas (errores claros) |
| `src/classifier.py` | Aplica las reglas, gestiona la incertidumbre y las preguntas de verificación |
| `src/stabilizer.py` | Decide cuándo una recomendación es estable (5 fotogramas) |
| `src/display.py` | Presentación en texto (terminal) |
| `src/gui.py` | Interfaz Tkinter: cajas, panel HUD, preguntas y pop-up opcional |
| `data/waste_rules.json` | Reglas y mensajes en español |
| `scripts/` | Descarga de modelos, diagnóstico de cámara, vista previa, benchmark y evaluación |

Decisiones de diseño: la IA solo reconoce; las reglas son deterministas y
editables sin tocar código; la lógica de negocio no depende de la interfaz; los
errores esperados (cámara, modelo, reglas, imagen) se informan con mensajes
claros en vez de ocultarse.

## Estructura del proyecto

```
ChavitoBins/
├── app.py
├── requirements.txt
├── data/
│   ├── waste_rules.json          # reglas del modelo de residuos (las que se usan)
│   └── waste_rules_coco.json     # reglas del modelo COCO anterior (alternativa)
├── models/
│   ├── README.md                 # procedencia, etiquetas y licencias de los modelos
│   └── *.onnx                    # se descargan con scripts.download_model (no están en Git)
├── src/                          # código de la aplicación (ver tabla de módulos)
├── scripts/
│   ├── download_model.py         # descarga y verifica (SHA-256) los modelos y las imágenes de prueba
│   ├── check_camera.py           # diagnostica qué índice y backend de cámara funcionan
│   ├── live_preview.py           # vista previa en vivo con ms por fotograma
│   ├── benchmark.py              # latencia y FPS con/sin filtro de personas
│   ├── evaluate.py               # precisión sobre carpetas de imágenes etiquetadas
│   └── validate_model.py         # prueba un modelo sobre una imagen
├── tests/                        # pruebas automáticas (pytest)
└── docs/
    └── test-plan.md              # plan de pruebas y matriz de casos
```

## Tecnologías y requisitos

- **Python** 3.12 o superior (probado con 3.12; la versión inicial también con 3.14.7).
- **ONNX Runtime** (CPU) para ejecutar los modelos; **NumPy**; **Pillow**.
- **OpenCV** (`opencv-python`) solo para la webcam.
- **Tkinter** para la interfaz: viene con Python en Windows. En Linux:
  `sudo apt install python3-tk` (Debian/Ubuntu) o `sudo pacman -S tk` (Arch).
- **pytest** para las pruebas.
- Una webcam, si se quiere el modo en vivo. No requiere GPU.
- Linux o Windows. Internet solo para la instalación inicial.

## Instalación

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate          # en fish: source .venv/bin/activate.fish
pip install -r requirements.txt
python -m scripts.download_model   # descarga los modelos (con comprobación SHA-256) y las imágenes de prueba
python -m pytest -q                # comprueba la instalación
```

Windows (PowerShell o Git Bash; se llama al Python del entorno sin activarlo):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m scripts.download_model
.venv\Scripts\python.exe -m pytest -q
```

`download_model` descarga el modelo de residuos y el detector de personas
(YOLOX-nano). Opciones: `--con-yolox-s` (alternativa más precisa y lenta) y
`--solo-residuos` (sin filtro de personas).

## Ejecución y uso

```bash
python app.py                                   # interfaz gráfica
python app.py --image samples/apple.jpg         # una imagen, resultado en la terminal
python app.py --image samples/apple.jpg --detalle   # con las detecciones originales del modelo
python app.py --camara 1                        # otro índice de cámara
python app.py --sin-filtro-personas             # desactiva el filtro y el complemento (para comparar)
python app.py --sin-complemento-coco            # solo filtra personas, sin aceptar botellas ni frutas de COCO
python app.py --min-confidence 0.6              # confianza mínima para recomendar
```

(En Windows, sustituye `python` por `.venv\Scripts\python.exe`.)

**Uso de la cámara.** Si no sabes qué índice o backend funciona, ejecuta
`python -m scripts.check_camera`. En Windows revisa *Configuración → Privacidad y
seguridad → Cámara* y cierra otras aplicaciones que la usen.

**Interfaz.**
- *Abrir imagen…* analiza una imagen; *Iniciar cámara* arranca el análisis en vivo.
- Las **cajas** muestran el material en español y la confianza (verde: recomendado,
  ámbar: por verificar, gris: confianza baja).
- El **panel de la derecha (HUD)** muestra la franja del color de la caneca, qué se
  detectó, el mensaje y las advertencias. En vivo, una recomendación aparece cuando
  el mismo objeto se ve **5 fotogramas seguidos** y desaparece al retirarlo.
- Si la cámara no puede decidir sola (plástico, metal o papel), el panel hace una
  **pregunta de sí/no**, por ejemplo *¿Es un empaque de snack o papas fritas?*. Al
  responder **Sí**, la franja cambia a **negro**.
- *Ventana emergente* (opcional) abre además un pop-up por recomendación.

**Herramientas.** `python -m scripts.live_preview` (ms por fotograma en vivo),
`python -m scripts.benchmark` (latencia), `python -m scripts.evaluate CARPETA`
(precisión). Ver [`docs/test-plan.md`](docs/test-plan.md).

## Reglas y código de colores

| Caneca | Residuos | Cómo lo decide ChavitoBins |
|---|---|---|
| Verde | Orgánicos aprovechables (restos de comida, cáscaras) | Material `biodegradable`; frutas y verduras detectadas por COCO |
| Blanca | Aprovechables limpios y secos: plástico, vidrio, metales, multicapa, papel y cartón | Materiales `plastic`, `glass`, `metal`, `paper`, `cardboard` (vidrio y metal se muestran juntos como "vidrio o metal"; ver Limitaciones); botellas detectadas por COCO (con aviso de verificar) |
| Negra | No aprovechables: papel higiénico, servilletas, papeles y cartones contaminados con comida, papeles metalizados, entre otros | **Solo por pregunta de verificación** (la cámara no puede verlo) |

Fuente: Resolución 2184 de 2019 y comunicado del Ministerio de Ambiente
(consultados en octubre de 2026). Dos matices: la norma nombra "papeles
metalizados" y los empaques de snacks se tratan como tales (interpretación
habitual, no literal); y "multicapa" figura entre los aprovechables, por lo que
ante un empaque concreto la advertencia remite al servicio de aseo local. Los
servicios municipales pueden tener indicaciones adicionales.

Las reglas están en `data/waste_rules.json`. El archivo se **valida al cargar**:
etiquetas mal escritas, categorías inexistentes, reglas duplicadas o
incompletas producen un error que lista todos los problemas.

## Funcionamiento sin conexión

La instalación y la descarga de modelos requieren internet. Después, la
ejecución normal es **totalmente local**: no hay APIs externas ni nube. Se
verificó ejecutando la aplicación en un entorno sin acceso a red.

## Pruebas

```bash
python -m pytest -q
```

135 pruebas de comportamiento (3 se omiten si no está YOLOX-s): reglas (casos normales y límite, JSON malformado,
reglas incompletas, categorías sin correspondencia), clasificador (desconocido,
confianza baja, preguntas de verificación), decodificación del modelo y
coordenadas, filtro de personas (solapamiento, objeto sostenido, caché),
estabilizador, cámara con dobles de prueba (sin hardware) y cierre de recursos.
Algunas pruebas usan los modelos reales y se omiten si no están descargados. La
interfaz se verificó aparte con un servidor gráfico virtual. El plan completo
está en [`docs/test-plan.md`](docs/test-plan.md).

## Personalización

| Qué cambiar | Dónde |
|---|---|
| Mensajes, advertencias, preguntas y nombres de canecas | `data/waste_rules.json` |
| Colores de franja, cajas, tamaños y textos de botones | Constantes al inicio de `src/gui.py` |
| Cuántos fotogramas hacen falta para mostrar una recomendación | `confirm_frames` / `release_frames` en `src/stabilizer.py` |
| Umbrales del filtro de personas y clases COCO complementarias | `--iou-personas`, `--contencion-personas`, `--intervalo-personas` o `DEFAULT_COMPLEMENT_LABELS` en `src/person_filter.py` (si agregas una clase, agrega su regla al JSON) |
| Confianza mínima | `--min-confidence` o `DEFAULT_MIN_CONFIDENCE` en `src/classifier.py` |

## Limitaciones

- **Solo 6 materiales.** El modelo reconoce `biodegradable`, `cardboard`, `glass`,
  `metal`, `paper` y `plastic`. No reconoce pilas, medicamentos, residuos
  electrónicos ni peligrosos.
- **No hay clase "no aprovechable".** El recipiente negro solo se alcanza
  respondiendo la pregunta de verificación. Si el modelo no reconoce un empaque,
  no habrá pregunta.
- **Siempre elige un material.** Fuera de su dominio puede inventar uno. El filtro
  de personas reduce el caso más frecuente, pero otros objetos (un perro, por
  ejemplo) pueden recibir una etiqueta de residuo.
- **Plásticos transparentes en la mano.** El modelo de residuos dio solo 0.24 a 0.43
  de confianza para `plastic` en una botella transparente sostenida a mano con fondo
  desordenado (por debajo del umbral). Las botellas se rescatan con el complemento
  COCO (`botella`, con aviso de verificar material y contenido), pero otros envases
  plásticos en condiciones similares pueden no reconocerse.
- **Vidrio y metal se confunden.** El modelo reconoce bien botellas y frascos de vidrio
  (119 de 120 imágenes de prueba), pero un vaso de vidrio se clasificó como metal
  (88 %) en todas las variaciones probadas. Como ambos materiales van al recipiente
  blanco, la regla los muestra juntos como "vidrio o metal" en vez de afirmar un
  material que puede estar equivocado. El recipiente recomendado es el correcto.
- **Orgánicos menos fiables.** El autor del modelo reporta `biodegradable` como su
  clase más débil (recall 0.56), y en pruebas una naranja se reconoció con baja
  confianza.
- **Filtro de personas.** Un objeto muy grande y muy cerca de la cámara puede
  confundirse con la persona y descartarse; los umbrales son ajustables.
- **No se ve la suciedad ni el contenido.** Por eso las recomendaciones piden que
  el residuo esté limpio y seco.
- **Sensible a distancia e iluminación.** El modelo se entrenó con objetos a
  distancia media.
- **Validación parcial.** Ver [Estado real](#estado-real-de-la-implementación).

## Modelos, licencias y fuentes

Detalle completo (procedencia, SHA-256, etiquetas y licencias) en
[`models/README.md`](models/README.md). Resumen:

- **Residuos:** `waste_yolo11n.onnx` (YOLO11n, repositorio `shshodiyev162-max/Waste-detector`).
  El repositorio declara MIT, pero los metadatos del ONNX indican AGPL-3.0 de
  Ultralytics: **la licencia efectiva no está confirmada**. Uso académico y local.
- **Filtro de personas:** `yolox_nano.onnx` (y opcionalmente `yolox_s.onnx`),
  YOLOX de Megvii, Apache-2.0 según el repositorio oficial.
- **Normativa:** Resolución 2184 de 2019 del Ministerio de Ambiente y Desarrollo
  Sostenible de Colombia.

## Guion sugerido para la demostración

1. Abrir `python app.py`, mostrar el HUD vacío y pulsar *Iniciar cámara*.
2. **Verde:** una fruta entera frente a la cámara → franja verde.
3. **Blanca:** una botella o lata limpia → franja blanca; mostrar la pregunta.
4. **Negra:** un empaque de papas o snack vacío → el panel pregunta; responder
   **Sí** → franja negra. *Probar antes con el empaque real* (ver `docs/test-plan.md`).
5. **Personas:** quedar frente a la cámara sin objeto → no aparece ninguna recomendación.
6. Explicar las limitaciones: seis materiales, el negro por pregunta y el
   funcionamiento sin conexión.
