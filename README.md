# ChavitoBins 

## Sistema Local de Reconocimiento y Clasificación Visual de Residuos

**Estado:** Planificación y diseño técnico  
**Fecha de planificación:** Octubre de 2026  
**Desarrollador:** Nekkro  
**Lenguaje principal:** Python 3.10+

---

## 1. Descripción del proyecto

ChavitoBins es un sistema de reconocimiento visual de residuos que busca facilitar su correcta separación mediante una cámara y un modelo de inteligencia artificial ejecutado localmente.

El sistema analizará imágenes capturadas en tiempo real, detectará objetos mediante un modelo de visión artificial preentrenado y utilizará un motor de reglas deterministas para generar recomendaciones sobre su disposición.

La aplicación estará diseñada para ejecutarse en el equipo del usuario, sin depender de APIs externas de inteligencia artificial, modelos de lenguaje (LLMs) ni servicios en la nube durante su funcionamiento normal.

La arquitectura separa el reconocimiento visual de la lógica de clasificación, lo que permitirá actualizar las reglas y recomendaciones sin modificar directamente el funcionamiento interno del modelo.

> **Nota sobre el estado:** ChavitoBins se encuentra en fase de planificación. La arquitectura, las tecnologías y la estructura de archivos descritas en este documento representan el diseño propuesto, no funcionalidades ya implementadas.

## 2. Problema que busca resolver

La separación de residuos puede resultar confusa cuando las personas desconocen la categoría a la que pertenece un objeto o no saben cómo disponer de él correctamente.

ChavitoBins pretende ofrecer orientación visual inmediata, mostrando el objeto detectado, el nivel de confianza de la predicción y una recomendación de disposición basada en reglas definidas.

El sistema será una herramienta de apoyo educativo. No reemplazará las indicaciones oficiales para la gestión de residuos ni garantizará la identificación de cualquier material.

## 3. Objetivos

### 3.1. Objetivo general

Desarrollar un prototipo de reconocimiento visual que detecte un conjunto limitado de objetos mediante una cámara web y genere recomendaciones de separación de residuos utilizando un modelo de visión artificial local y un motor de reglas en Python.

### 3.2. Objetivos específicos

- Capturar imágenes en tiempo real mediante una cámara web.
- Ejecutar localmente un modelo de visión artificial preentrenado.
- Detectar objetos pertenecientes a un conjunto de clases previamente definido.
- Procesar las predicciones y sus niveles de confianza.
- Consultar una base de conocimiento de reglas almacenadas en JSON.
- Generar recomendaciones a partir de las detecciones y las reglas aplicables.
- Mostrar los resultados mediante una interfaz visual sencilla.
- Gestionar objetos desconocidos y predicciones de baja confianza.
- Verificar la lógica de clasificación mediante pruebas automatizadas.
- Documentar los requisitos, las limitaciones y los resultados de las pruebas.

## 4. Alcance del MVP

El producto mínimo viable (MVP) priorizará un flujo funcional de principio a fin sobre la cantidad de objetos reconocidos o la complejidad de la interfaz.

### 4.1. Funcionalidades incluidas

1. Captura de imágenes mediante cámara web.
2. Procesamiento local de los fotogramas.
3. Detección de un conjunto reducido de objetos compatibles con el modelo seleccionado.
4. Obtención de etiquetas, confianza y coordenadas de las detecciones, cuando el modelo las proporcione.
5. Clasificación mediante reglas deterministas.
6. Consulta de reglas desde un archivo JSON.
7. Visualización de las detecciones y las recomendaciones.
8. Manejo de objetos desconocidos o predicciones inciertas.
9. Modo opcional de procesamiento de imágenes estáticas para facilitar las pruebas.
10. Pruebas unitarias del motor de reglas.
11. Ejecución sin conexión a internet una vez instaladas las dependencias y disponibles los archivos necesarios.

### 4.2. Funcionalidades fuera del alcance inicial

- Entrenamiento de un modelo desde cero.
- Reconocimiento garantizado de todos los residuos.
- Identificación de la composición química de los materiales.
- Determinación fiable de contaminación a partir de una imagen únicamente.
- Clasificación exhaustiva de residuos peligrosos, hospitalarios o industriales.
- Aplicación móvil o interfaz web.
- Registro de usuarios o almacenamiento remoto.
- Integración con APIs externas de IA o LLMs.
- Automatización física de contenedores.
- Sistemas de seguimiento, estadísticas o análisis en la nube.

Estas funcionalidades podrán considerarse en futuras versiones si los resultados del MVP justifican su desarrollo.

## 5. Arquitectura del sistema

ChavitoBins utilizará una arquitectura modular que separa la captura de imágenes, la inferencia del modelo, la toma de decisiones y la presentación de resultados.

### 5.1. Flujo general

```text
             ┌───────────────────┐
             │    Cámara web     │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Captura de imagen  │
             │      OpenCV       │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │  Preprocesamiento │
             │     de imagen     │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Modelo de visión  │
             │   ejecutado local │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Procesamiento de  │
             │    detecciones    │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Motor de reglas   │◄──── Base de conocimiento
             │     Python        │      waste_rules.json
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Recomendaciones e │
             │ interfaz visual   │
             └───────────────────┘
```

### 5.2. Componentes principales

| Componente | Responsabilidad |
|---|---|
| Captura de cámara | Obtener fotogramas mediante OpenCV. |
| Preprocesamiento | Ajustar las imágenes al formato requerido por el modelo. |
| Visión artificial | Ejecutar el modelo local y procesar sus salidas. |
| Motor de clasificación | Aplicar reglas de decisión a las detecciones. |
| Base de conocimiento | Almacenar y consultar las categorías y recomendaciones. |
| Interfaz visual | Mostrar el objeto detectado, la confianza y los mensajes correspondientes. |
| Aplicación principal | Coordinar los componentes y gestionar el ciclo de ejecución. |

### 5.3. Principios de diseño

- **Modularidad:** cada módulo tendrá una responsabilidad definida.
- **Procesamiento local:** las imágenes y predicciones se procesarán en el equipo del usuario.
- **Separación de responsabilidades:** la IA detectará objetos; las reglas determinarán las recomendaciones.
- **Trazabilidad:** las recomendaciones podrán relacionarse con las detecciones y reglas que las originaron.
- **Tolerancia a errores:** una predicción incierta no se convertirá automáticamente en una clasificación definitiva.
- **Configurabilidad:** las reglas podrán modificarse desde JSON.
- **Alcance verificable:** solo se anunciarán como compatibles las clases probadas.

## 6. Tecnologías previstas

| Tecnología | Utilidad |
|---|---|
| Python 3.10+ | Lenguaje principal de desarrollo. |
| OpenCV | Captura de cámara y visualización de resultados. |
| ONNX Runtime | Ejecución local del modelo si se selecciona un modelo compatible con ONNX. |
| NumPy | Operaciones y transformación de matrices de imágenes. |
| JSON | Almacenamiento de reglas y recomendaciones. |
| pytest | Pruebas automatizadas. |
| Git | Control de versiones del código fuente. |

Las versiones concretas de las dependencias se definirán y verificarán durante la implementación, para después registrarlas en `requirements.txt`.

La elección de ONNX Runtime depende de que el modelo seleccionado sea compatible con ese formato. No se presupone que cualquier modelo de visión pueda ejecutarse directamente sin adaptar su entrada, sus salidas o su postprocesamiento.

## 7. Inteligencia artificial y reconocimiento visual

### 7.1. Enfoque seleccionado

El MVP utilizará un modelo de visión artificial preentrenado que se ejecute localmente. Esta estrategia busca reducir el tiempo de desarrollo y evitar la necesidad de recopilar y etiquetar un conjunto de datos propio.

La selección del modelo deberá priorizar la compatibilidad con las clases de objetos necesarias para la demostración, la capacidad de ejecutarse en el hardware disponible y la posibilidad de utilizarlo conforme a su licencia.

### 7.2. Requisitos del modelo

El modelo seleccionado deberá cumplir los siguientes requisitos:

- Ejecutarse sin solicitar servicios externos durante la inferencia.
- Reconocer objetos relevantes para el alcance definido.
- Permitir interpretar las clases y los resultados que genera.
- Proporcionar información de confianza o un mecanismo equivalente, si está disponible.
- Tener una licencia compatible con el uso previsto.
- Ser compatible con el entorno de ejecución local.
- Disponer de información suficiente para implementar el preprocesamiento y el postprocesamiento.

### 7.3. Validación previa a la integración

Antes de integrarlo en la aplicación se verificará:

1. La existencia y procedencia del modelo.
2. Las clases que realmente puede reconocer.
3. El formato y las dimensiones de entrada.
4. La estructura de las salidas.
5. El procedimiento de postprocesamiento requerido.
6. La ejecución con imágenes de prueba.
7. El rendimiento en el equipo disponible.
8. La licencia y las condiciones de distribución.

La selección definitiva del modelo permanece pendiente de esta validación.

## 8. Motor de clasificación y base de conocimiento

El motor de clasificación será desarrollado en Python y utilizará reglas explícitas almacenadas en `data/waste_rules.json`.

Su responsabilidad será relacionar las detecciones del modelo con las categorías y recomendaciones definidas para el proyecto.

La clasificación no dependerá exclusivamente de la etiqueta visual. También se tendrán en cuenta las limitaciones de la detección y la información que realmente pueda obtenerse de la imagen.

### 8.1. Responsabilidades del motor de reglas

- Recibir las detecciones procesadas por el módulo de visión.
- Verificar si la detección cumple las condiciones mínimas para evaluarse.
- Consultar las reglas correspondientes.
- Determinar si existe una recomendación aplicable.
- Identificar situaciones que requieren una advertencia.
- Devolver un resultado estructurado a la interfaz visual.
- Informar cuando no exista una regla válida para el objeto detectado.

### 8.2. Estructura conceptual de las reglas

Cada regla podrá contener campos como los siguientes:

| Campo | Descripción |
|---|---|
| `id` | Identificador único de la regla. |
| `clases` | Clases de objetos a las que puede aplicarse. |
| `categoria` | Categoría de disposición recomendada. |
| `recomendacion` | Mensaje que se mostrará al usuario. |
| `advertencias` | Precauciones o restricciones relevantes. |
| `condiciones` | Condiciones necesarias para aplicar la regla. |

Esta estructura es conceptual y podrá ajustarse durante la implementación para mantener un formato JSON sencillo y validable.

### 8.3. Separación entre reconocimiento y disposición

Reconocer un objeto no siempre permite determinar cómo debe desecharse.

Por ejemplo, identificar visualmente una botella no permite saber con certeza si contiene sustancias peligrosas o si está contaminada. Por ello, el sistema no deberá inferir propiedades que el modelo no pueda observar.

Cuando falte información, mostrará una recomendación de verificación manual en lugar de presentar una clasificación definitiva.

## 9. Esquema de separación de residuos en Colombia

Las recomendaciones deberán distinguir entre el código general de separación de residuos y las situaciones que requieren un tratamiento especial.

Como referencia general para la separación de residuos en Colombia se contempla:

- **Blanco:** residuos aprovechables que cumplan las condiciones correspondientes.
- **Verde:** residuos orgánicos aprovechables.
- **Negro:** residuos no aprovechables.

La aplicación no deberá tratar el recipiente rojo como una categoría general de todos los residuos peligrosos domésticos. Cualquier manejo especial deberá delimitarse según el tipo de residuo y las disposiciones aplicables.

La clasificación final dependerá de las reglas que se definan y verifiquen para el MVP. El sistema no afirmará que puede reconocer automáticamente todas las condiciones necesarias para determinar la disposición correcta de cualquier objeto.

## 10. Manejo de confianza e incertidumbre

La confianza generada por el modelo será utilizada como una señal de apoyo para decidir si una detección puede procesarse. No se interpretará como garantía de exactitud.

### Comportamiento previsto

- **Detección aceptable:** consultar las reglas aplicables y mostrar la recomendación correspondiente.
- **Confianza insuficiente:** indicar que el objeto no pudo identificarse con suficiente seguridad.
- **Objeto sin regla asociada:** informar que no existe una recomendación automática disponible.
- **Posible residuo peligroso:** evitar recomendaciones definitivas que no estén respaldadas por una regla validada.
- **Error de cámara o modelo:** mostrar un mensaje comprensible y finalizar o permitir reintentar de forma controlada.

El umbral de confianza se establecerá durante las pruebas con el modelo seleccionado.

## 11. Estructura de directorios prevista

La siguiente estructura representa la organización propuesta para el repositorio:

```text
chavitobins/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── camera.py
│   ├── preprocessing.py
│   ├── vision.py
│   ├── classifier.py
│   ├── knowledge_base.py
│   └── display.py
│
├── data/
│   └── waste_rules.json
│
├── models/
│   └── README.md
│
└── tests/
    ├── test_rules.py
    └── test_vision.py
```

La estructura podrá modificarse cuando se conozcan las necesidades reales de integración. No será necesario crear módulos vacíos que no aporten valor al MVP.

### Responsabilidad de los archivos

| Archivo | Función prevista |
|---|---|
| `app.py` | Punto de entrada y coordinación general. |
| `src/camera.py` | Captura y gestión de la cámara. |
| `src/preprocessing.py` | Preparación de imágenes para el modelo. |
| `src/vision.py` | Inferencia y procesamiento de detecciones. |
| `src/classifier.py` | Evaluación de reglas y generación de resultados. |
| `src/knowledge_base.py` | Carga y consulta de las reglas JSON. |
| `src/display.py` | Visualización de detecciones y recomendaciones. |
| `data/waste_rules.json` | Base de conocimiento de clasificación. |
| `models/` | Ubicación prevista para el modelo y su documentación. |
| `tests/` | Pruebas automatizadas del sistema. |

Los pesos del modelo se gestionarán localmente y no se incluirán en el repositorio por defecto si su tamaño o licencia hacen inconveniente distribuirlos. Se documentarán su procedencia, versión y procedimiento de instalación.

## 12. Plan de pruebas

La validación se dividirá en pruebas unitarias, pruebas del modelo y pruebas de integración.

### 12.1. Pruebas unitarias

- Carga correcta de la base de conocimiento.
- Validación de la estructura de las reglas.
- Aplicación de reglas para clases conocidas.
- Gestión de objetos sin reglas asociadas.
- Manejo de detecciones con confianza insuficiente.
- Generación de resultados y advertencias esperadas.

### 12.2. Pruebas de visión artificial

- Ejecución del modelo con imágenes estáticas.
- Comprobación de las clases detectadas.
- Verificación del formato de las salidas.
- Pruebas con objetos incluidos en el alcance.
- Pruebas con objetos no contemplados.
- Manejo de errores durante la inferencia.

### 12.3. Pruebas de integración

- Apertura y cierre de la cámara.
- Flujo completo desde la captura hasta la recomendación.
- Visualización de las detecciones y los mensajes.
- Manejo de errores de ejecución.
- Comprobación del funcionamiento sin conexión después de preparar el entorno.

## 13. Criterios de aceptación del MVP

La primera versión se considerará entregable cuando:

- La aplicación pueda iniciarse en el entorno documentado.
- La cámara pueda capturar imágenes correctamente.
- El modelo seleccionado pueda ejecutarse localmente.
- Se reconozcan las clases concretas incluidas en el alcance de la demostración.
- El motor de reglas genere las recomendaciones esperadas para los casos de prueba.
- Los objetos desconocidos o las predicciones inciertas no generen clasificaciones definitivas injustificadas.
- Los resultados sean visibles en la interfaz.
- Las pruebas principales puedan ejecutarse y sus resultados estén documentados.
- La aplicación funcione sin conexión una vez instaladas las dependencias y disponibles los archivos del modelo.
- Las instrucciones de instalación y ejecución correspondan con el código implementado.

El criterio de éxito será demostrar un flujo completo y verificable, no reconocer todos los residuos posibles.

## 14. Privacidad y funcionamiento sin conexión

ChavitoBins priorizará el procesamiento local.

Durante la ejecución normal:

- Las imágenes no se enviarán a APIs externas.
- No se dependerá de servicios de IA en la nube.
- No se almacenarán imágenes de forma persistente salvo que esa función se incorpore y documente expresamente.
- Las dependencias y el modelo deberán estar disponibles localmente antes de iniciar la ejecución sin conexión.

La instalación inicial podrá requerir conexión a internet para descargar dependencias y obtener el modelo. Una vez preparado el entorno, la inferencia y la clasificación deberán funcionar sin conexión.

## 15. Limitaciones previstas

- La capacidad de reconocimiento dependerá de las clases y del entrenamiento del modelo seleccionado.
- La iluminación, el ángulo, el fondo y la oclusión pueden afectar los resultados.
- La apariencia visual no siempre permite determinar la composición o el estado de un residuo.
- Un modelo genérico puede no distinguir materiales relevantes para su separación.
- La calidad de las recomendaciones dependerá de la cobertura y validez de las reglas.
- El rendimiento dependerá del procesador, la memoria y el hardware disponible.
- La aplicación será una herramienta de orientación y no sustituirá las instrucciones oficiales para la gestión de residuos peligrosos.

## 16. Plan de desarrollo

El trabajo se organizará por etapas para reducir riesgos y priorizar el funcionamiento integral.

1. **Definición del alcance:** seleccionar las clases objetivo y establecer los criterios de aceptación.
2. **Validación del modelo:** comprobar compatibilidad, licencia y capacidad de detección.
3. **Captura e inferencia:** implementar la cámara y la ejecución local del modelo.
4. **Motor de reglas:** crear la base de conocimiento y la lógica de clasificación.
5. **Integración visual:** conectar los módulos y mostrar las recomendaciones.
6. **Pruebas y documentación:** verificar el flujo completo, corregir errores y preparar la demostración.

Se priorizará un sistema pequeño que funcione de principio a fin antes de ampliar la cantidad de clases o incorporar funcionalidades adicionales.

## 17. Estado actual del proyecto

En esta etapa, ChavitoBins se encuentra en planificación y diseño técnico.

- [x] Definición preliminar del objetivo.
- [x] Propuesta de arquitectura modular.
- [x] Selección preliminar de tecnologías.
- [x] Definición inicial del alcance del MVP.
- [ ] Selección y validación del modelo.
- [ ] Creación de la estructura de directorios.
- [ ] Implementación de la captura de cámara.
- [ ] Integración de la inferencia local.
- [ ] Implementación del motor de clasificación.
- [ ] Integración de la interfaz visual.
- [ ] Ejecución de las pruebas de aceptación.

Las casillas describen el estado previsto al redactar esta especificación y deberán actualizarse a medida que avance el desarrollo.

---

## 18. Resultado esperado

ChavitoBins busca demostrar que un modelo de visión artificial ejecutado localmente, combinado con un motor de reglas explícitas, puede ofrecer orientación práctica para la separación de un conjunto limitado de residuos.

El diseño prioriza la privacidad, el funcionamiento sin conexión, la separación de responsabilidades, la trazabilidad de las recomendaciones y un alcance técnicamente verificable.

El objetivo del MVP es entregar una aplicación funcional, documentada y demostrable, sobre la cual puedan construirse futuras mejoras.
