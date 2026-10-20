# ChavitoBins — Sistema de Reconocimiento y Clasificación Visual de Residuos

Desarrollado por **Nekkro**. Un sistema de visión artificial y motor de reglas determinista diseñado para funcionar 100% en local. Utiliza procesamiento de imágenes en tiempo real a través de cámara web para detectar residuos y sugerir su disposición según el código unificado de colores en Colombia (Resolución 2184 de 2019 / 591 de 2024), gestionando las canecas Verde, Roja y Negra de forma normativamente rigurosa.

---

## Mapeo y Lógica de Disposición (Normativa Colombia)

> **Nota Técnica:** En Colombia, el código de colores ordinario asigna la caneca Blanca a los reciclables limpios. Para fines de este proyecto (enfocado en Verde, Negra y Roja):
> - **Verde:** Orgánicos aprovechables (restos de comida, cáscaras, desechos agrícolas).
> - **Negra:** No aprovechables (servilletas sucias, papel higiénico, cartón/plástico contaminado, residuos ordinarios).
> - **Roja:** Residuos peligrosos e infecciosos (material con riesgo biológico, jeringas, elementos cortopunzantes, tapabocas o guantes de látex). Los residuos peligrosos como pilas o RAEES son marcados bajo condición especial o ruta selectiva, nunca arrojados por defecto a la caneca convencional.

---

## Stack Tecnológico y Justificación

- **Python 3.10+**: Lenguaje principal por su robusto ecosistema en procesamiento numérico y visión artificial.
- **OpenCV (`opencv-python`)**: Manipulación de fotogramas, procesamiento visual en tiempo real y renderizado del canvas interactivo sin sobrecarga de GUI.
- **ONNX Runtime (`onnxruntime`)**: Motor de inferencia ultrarrápido y ligero para ejecutar modelos de visión en local usando CPU, evitando la dependencia de frameworks pesados como PyTorch o CUDA en producción.
- **YOLOv8 / MobileNet**: Arquitecturas evaluadas para detección y localización de objetos con alto rendimiento en tiempo real.
- **NumPy**: Operaciones matriciales de alta velocidad sobre los canales de imagen.
- **JSON**: Formato ligero y desacoplado para almacenar la base de reglas de negocio (`reglas_residuos.json`).
- **Pytest**: Entorno de pruebas unitarias para validar las decisiones del motor de reglas sin necesidad de invocar la cámara web.

---

# Arquitectura del Sistema

```text
                        +-----------------------+
                        |      Cámara Web       |
                        +-----------+-----------+
                                    |
                                    v [Fotogramas]
                        +-----------------------+
                        |      `camara.py`      |
                        +-----------+-----------+
                                    |
                                    v
                        +-----------------------+
                        |   `preprocesamiento.py`|
                        +-----------+-----------+
                                    |
                                    v
                        +-----------------------+
                        |      `vision.py`      |
                        | (YOLOv8 / ONNX Local) |
                        +-----------+-----------+
                                    |
                                    v [Detecciones + Cajas Delimitadoras + Confianza]
                        +-----------------------+
                        |    `clasificador.py`  |
                        +-----------+-----------+
                                    |
                                    +<--- Consultas <---+
                                    |                   |
                                    v                   |
                        +-----------------------+  +----+-------------------------+
                        | `base_conocimiento.py`|--| data/reglas_residuos.json   |
                        +-----------------------+  +------------------------------+
                                    |
                                    v [Objeto Clasificado / Estado / Advertencias]
                        +-----------------------+
                        |    `visualizador.py`  |
                        |    (OpenCV Canvas)    |
                        +-----------------------+

---

## Mapeo y Lógica de Disposición (Normativa Colombia)

| Caneca / Categoria | Tipo de Residuo Cubierto | Ejemplos Comunes |
| --- | --- | --- |
| **Verde** | Orgánicos aprovechables | Restos de comida, cáscaras de frutas/verduras, desechos agrícolas y residuos de jardinería. |
| **Negra** | No aprovechables / Ordinarios | Servilletas usadas, papel higiénico, cartón/plástico contaminado con comida, empaques metalizados de snacks. |
| **Roja** | Peligrosos e Infecciosos | Elementos con riesgo biológico, jeringas, elementos cortopunzantes, tapabocas o guantes de látex. |
| **Manejo Especial** | RAEES / Pilas / Químicos | Pilas usadas, baterías, medicamentos vencidos, aceite de cocina usado (requieren ruta selectiva). |

> **Nota Técnica:** En Colombia, el código de colores ordinario asigna la caneca **Blanca** a los reciclables limpios y secos. Para el alcance de este proyecto (enfocado en Verde, Negra y Roja), los residuos reciclables que se encuentren sucios o contaminados son dirigidos a la **Caneca Negra** por criterio de precaución.

---

## Por qué un motor de reglas separado del modelo visual

* **Flexibilidad normativa:** Las regulaciones ambientales cambian por municipio o legislación sin necesidad de reentrenar la red neuronal.
* **Seguridad en residuos peligrosos:** Previene que un objeto detectado con alta confianza visual (ej. una jeringa) sea clasificado incorrectamente por falta de contexto normativo.
* **Manejo de ambigüedad:** Si el modelo detecta un empaque sucio con baja confianza, el motor de reglas asigna por defecto la **Caneca Negra** por criterio de precaución.

---

## Stack

* **Lenguaje:** Python 3.10+
* **Visión Artificial:** OpenCV (`opencv-python`) para captura y renderizado en tiempo real.
* **Inferencia Local:** ONNX Runtime (`onnxruntime`) para ejecutar modelos YOLOv8/MobileNet en CPU sin depender de GPU ni CUDA.
* **Procesamiento Numérico:** NumPy para manipulación matricial de fotogramas.
* **Base de Conocimiento:** JSON estructurado (`data/waste_rules.json`) desacoplado de la lógica de visión.
* **Pruebas:** `pytest` para verificación unitaria de las reglas de negocio.

---

## Estructura del Proyecto

* **`/chavitobins`**
* **`app.py`**: Coordinador del flujo principal y bucle de la cámara.
* **`camera.py`**: Inicialización, lectura y liberación de la cámara web.
* **`vision.py`**: Carga del modelo (YOLOv8 / ONNX) e inferencia visual.
* **`preprocessing.py`**: Redimensionamiento, normalización y adaptaciones para el modelo.
* **`knowledge_base.py`**: Carga, indexación y validación del motor de reglas.
* **`classifier.py`**: Cruce entre detecciones visuales y base de conocimiento.
* **`models.py`**: Estructuras de datos (`DetectionResult`, `WasteRule`, `ClassificationOutput`).
* **`display.py`**: Dibujado de cajas delimitadoras, etiquetas y avisos en OpenCV.
* **`data/`**
* **`waste_rules.json`**: Mapeo de clases, sinónimos, canecas y normatividad.


* **`models_weights/`**
* **`yolov8n_waste.onnx`**: Pesos del modelo local (fuera del control de versiones).


* **`tests/`**
* **`test_rules.py`**: Pruebas unitarias para el motor de reglas.
* **`test_vision.py`**: Pruebas de inferencia con imágenes estáticas.


* **`README.md`**: Documentación principal del proyecto.



---

## Contrato de Datos (Contrato de Detección)

Cada iteración entre el módulo de visión y el clasificador genera una estructura estandarizada:

```json
{
  "nombre_clase": "cascara_banano",
  "confianza": 0.92,
  "caja_delimitadora": [120, 80, 340, 420],
  "regla_aplicada": "REGLA_ORG_001",
  "estado": "CLASIFICADO",
  "categoria_recomendada": "Verde",
  "advertencias": []
}

```

---

## Requisitos y Configuración Local

### Prerrequisitos

* **Python 3.10+**
* **Cámara web integrada o USB** (o un directorio con imágenes estáticas para pruebas).

### 1. Clonar el repositorio

```bash
git clone [https://github.com/nekkro/chavitobins.git](https://github.com/nekkro/chavitobins.git)
cd chavitobins

```

### 2. Crear un entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install opencv-python onnxruntime numpy pytest

```

### 3. Descarga de Pesos del Modelo Local

Descarga los pesos localmente en la carpeta **`models_weights/`** antes de ejecutar la aplicación:

```bash
mkdir -p models_weights
# Descargar el modelo en formato ONNX / PyTorch localmente

```

---

## Ejecución del Sistema

### Modo Tiempo Real (Cámara Web)

```bash
python app.py

```

*Para salir de la aplicación, presiona la tecla `q` en la ventana de la cámara.*

### Modo de Prueba con Imágenes Estáticas

```bash
python app.py --image tests/samples/test_apple.jpg

```

---

## Pruebas

Antes de ejecutar la aplicación con la cámara encendida, ejecuta la suite de pruebas para verificar el motor de reglas:

```bash
pytest tests/

```

Prueba la preservación de reglas de negocio, resolución de ambigüedades y comportamientos ante objetos desconocidos sin sobrecarga de hardware.

---

## Limitaciones Conocidas

1. **Variaciones de Iluminación:** Rendimiento dependiente de la intensidad lumínica ambiental sobre el objeto.
2. **Residuos Ocultos o Compuestos:** Empaques multilaminados (ej. empaques metalizados de snacks) son redirigidos a la caneca negra debido a su baja reciclabilidad.
