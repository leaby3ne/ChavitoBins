# Plan de pruebas de ChavitoBins

Este documento define **cómo** se prueba el sistema y registra los resultados
**medidos**. Lo que aún no se ha medido figura como *Pendiente*; no se
estiman ni se suponen resultados.

## 1. Entornos de prueba

| Entorno | Descripción | Qué se verificó allí |
|---|---|---|
| **A. Desarrollo (Linux)** | Python 3.12.3, ONNX Runtime 1.24.4, NumPy 2.4.4, Pillow 12.1.1, OpenCV 4.13.0, CPU Intel Xeon 2.1 GHz de **1 núcleo** (servidor, sin cámara) | Pruebas automáticas, rendimiento y precisión con imágenes, interfaz con servidor gráfico virtual y cámara simulada, ejecución sin red |
| **B. Demostración (Windows + webcam)** | PC de la demostración | **Pendiente:** completar las secciones marcadas con ☐ |

Los tiempos del entorno A **no son los del equipo de la demostración**: sirven
para comparar configuraciones entre sí, no como cifra absoluta.

## 2. Pruebas automáticas

```bash
python -m pytest -q
```

**Resultado en el entorno A: 135 pruebas, todas pasan.** Con la instalación por defecto (sin `--con-yolox-s`) se ejecutan 132 y se omiten 3, que necesitan `yolox_s.onnx`. ☐ Repetir en el entorno B.

| Área | Qué comprueban |
|---|---|
| Reglas (`test_knowledge_base`) | Regla válida; JSON malformado; archivo ausente; regla incompleta; categoría sin correspondencia; clases duplicadas; etiqueta que el modelo no tiene; preguntas de verificación válidas e inválidas; el archivo real tiene reglas para todas las clases del modelo |
| Clasificador (`test_classifier`) | Clase conocida; desconocida (no se nombra); confianza insuficiente (sin categoría); umbral exacto; agrupación por regla; respuesta Sí/No a la pregunta y paso al recipiente negro |
| Visión (`test_vision`, `test_vision_yolo`, `test_preprocessing`) | Letterbox y coordenadas; decodificación de las salidas de los dos formatos de modelo; filtrado por confianza; recorte a los bordes; metadatos de etiquetas (sin ejecutar código); errores de carga e inferencia; modelos reales |
| Filtro y complemento COCO (`test_person_filter`) | Solapamiento y contención; objeto sostenido que se conserva; el modelo COCO no corre sin necesidad; reutilización y reinicio de la memoria; complemento (botella rescatada, sin duplicados, solo clases permitidas, COCO en cada fotograma); casos reales con personas |
| Estabilizador (`test_stabilizer`) | Aparece tras N fotogramas; no se repite; tolera parpadeos; olvida lo que se fue; falso positivo aislado nunca aparece |
| Cámara (`test_camera`) | Apertura, lectura, errores y liberación (con cámara simulada, sin hardware) |
| Presentación y evaluación | Formato de texto; criterios del script de evaluación |

## 3. Pruebas de rendimiento

```bash
python -m scripts.benchmark                      # filtro con YOLOX-nano (predeterminado)
python -m scripts.benchmark --modelo-personas models/yolox_s.onnx
python -m scripts.live_preview                   # ms por fotograma en vivo (ventana)
```

Se mide el flujo completo: preprocesamiento, inferencia, decodificación y reglas.
"Imágenes sueltas" es el peor caso (el filtro siempre busca personas);
"Cámara continua" reproduce una escena estable como un video.

**Resultados medidos en el entorno A** (4 imágenes: personas, fruta, sin residuos y
una botella en la mano; 12 repeticiones; CPU de **1 núcleo**):

| Configuración | Imágenes sueltas: media / p95 (ms) | Cámara continua: media / p95 (ms) | FPS (cámara continua) |
|---|---|---|---|
| Solo modelo de residuos | 116 / 128 | 112 / 120 | 8.9 |
| + filtro de personas (YOLOX-nano, cada 3 fotogramas) | 129 / 160 | 121 / 147 | 8.3 |
| + filtro y complemento COCO (**predeterminado**) | 148 / 168 | 147 / 156 | 6.8 |

Medición previa con YOLOX-s en lugar de nano para el filtro: añadía de 55 a 180 ms
por fotograma (media 156 ms y p95 384 ms con el filtro cada 3 fotogramas; 278 ms
con el filtro en cada fotograma), por eso nano es el predeterminado.

**Lectura.** El filtro de personas con YOLOX-nano cuesta unos 10 ms. El
complemento COCO obliga a ejecutar el modelo COCO en cada fotograma (sus cajas
deben estar al día), lo que sube la media a unos 147 ms (6.8 FPS en este
servidor de 1 núcleo). Si el equipo de la demostración no alcanza el criterio,
`--sin-complemento-coco` vuelve a la configuración de 8.3 FPS (a costa de perder
las botellas y frutas de COCO).

**Criterio propuesto de aceptación:** ≥ 6 FPS y p95 < 250 ms en el equipo de la demostración (una cámara de 30 fps no exige más: el panel solo necesita que la recomendación aparezca en pocos segundos).

| Medición en el entorno B (☐ completar) | Resultado |
|---|---|
| CPU / RAM del equipo | ☐ |
| `check_camera`: índice, backend y resolución | ☐ (previo: DSHOW, 640x480, ~30 fps) |
| Solo residuos: media / p95 / FPS | ☐ |
| Con filtro y complemento (predeterminado): media / p95 / FPS | ☐ |
| ¿La ventana responde con fluidez? | ☐ |

## 4. Pruebas de precisión de clasificación

```bash
python -m scripts.evaluate eval/
```

Carpetas dentro de `eval/` (el nombre indica lo esperado):

| Carpeta | Contenido | Se considera correcto si… |
|---|---|---|
| `verde/` | Residuos orgánicos | recomienda el recipiente verde |
| `blanca/` | Plástico, vidrio, metal, papel o cartón limpios | recomienda el recipiente blanco |
| `negra/` | Empaques de snacks, papel/cartón con grasa, servilletas usadas | **ofrece la pregunta que lleva al negro** (el usuario responde Sí) |
| `ninguna/` | Personas, objetos fuera de alcance | no recomienda nada |

La carpeta `negra` se mide así porque la cámara no puede ver si un empaque es
metalizado o está sucio: el sistema acierta cuando **reconoce un objeto de
material dudoso y pregunta**.

**Resultados preliminares en el entorno A** (imágenes de ejemplo, **no** objetos
reales de la demostración):

| Carpeta | Imágenes | Correctas | Aciertos |
|---|---|---|---|
| `verde` (2 fotos de frutas) | 2 | 1 | 50 % (la naranja se reconoció con 46 % y no se recomendó) |
| `blanca` (8 imágenes de TrashNet por cada material) | 40 | 36 | 90 % |
| `negra` (40 imágenes de la categoría *trash* de TrashNet) | 40 | 29 | 72 % |
| `ninguna` (personas, perro) | 2 | 2 | 100 % (con filtro) |
| **Total** | 84 | 68 | 81 % |

**Advertencias sobre estos números.** TrashNet son fotos de un objeto sobre
fondo simple y podrían formar parte de los datos con los que se entrenó el
modelo, así que los resultados son optimistas. La carpeta *trash* de TrashNet no
equivale a empaques de snacks. Las carpetas `verde` y `ninguna` son muy
pequeñas. **No sustituyen la prueba con los objetos reales de la demostración.**

| Medición en el entorno B (☐ completar con fotos propias, ≥ 20 por carpeta) | Resultado |
|---|---|
| `verde` | ☐ |
| `blanca` | ☐ |
| `negra` (con los empaques reales de la demo) | ☐ |
| `ninguna` (usuario frente a la cámara, con el filtro) | ☐ |

Para ajustar la confianza mínima: `python app.py --image FOTO --detalle` muestra
la confianza de cada detección.

## 5. Matriz de casos de prueba

Estados: **Medido** (resultado registrado), **Auto** (cubierto por pruebas
automáticas), **Pendiente** (requiere cámara o equipo B).

### Caneca verde
| ID | Escenario | Resultado esperado | Estado |
|---|---|---|---|
| V-01 | Manzana frente a la cámara | Franja verde con "orgánico"; aviso de verificar | Medido en imagen (91 %). ☐ En vivo |
| V-02 | Banana o verdura cruda | Franja verde | ☐ Pendiente |
| V-03 | Naranja | Verde o, si la confianza < 0.5, ninguna recomendación (riesgo conocido) | Medido en imagen: baja confianza, sin recomendación |

### Caneca blanca
| ID | Escenario | Resultado esperado | Estado |
|---|---|---|---|
| B-01 | Botella de plástico (también en la mano) | Franja blanca con "botella" y aviso de verificar | Medido en imagen: botella 75 % (ver 5.2); ☐ en vivo |
| B-01b | Otro envase plástico (no botella) | Franja blanca; pregunta de empaque de snack | Medido en imágenes (TrashNet); ☐ en vivo, puede quedar bajo el umbral |
| B-02 | Botella o frasco de vidrio | Franja blanca; se muestra como "vidrio o metal" | Medido en imágenes (TrashNet: 119/120 reconocidos como vidrio); ☐ en vivo |
| B-02b | **Vaso de vidrio** (hallazgo del desarrollador) | Franja blanca; se muestra como "vidrio o metal" | Medido: el modelo da *metal* 0.88 y *vidrio* 0.00, estable ante espejo, brillo, contraste, recorte y escala. La caneca es correcta; el nombre se muestra combinado |
| B-03 | Lata de metal limpia | Franja blanca; pregunta sobre empaque metalizado | Medido en imágenes; ☐ en vivo |
| B-04 | Papel o cartón limpio y seco | Franja blanca; pregunta sobre suciedad | Medido en imágenes; ☐ en vivo |

### Caneca negra
| ID | Escenario | Resultado esperado | Estado |
|---|---|---|---|
| N-01 | **Bolsa de papas o snack vacía** (muestra principal) | Se reconoce como plástico, metal o papel y el panel pregunta; **Sí → franja negra** | ☐ **Pendiente: no se ha comprobado qué clase asigna el modelo a un empaque real** |
| N-02 | Papel o cartón con grasa o comida | Pregunta; Sí → negro | Auto (flujo de la pregunta); ☐ en vivo |
| N-03 | Servilleta usada | Se reconoce como papel; pregunta; Sí → negro | ☐ Pendiente (puede no detectarse) |
| N-04 | Envase plástico sucio con comida | Pregunta; Sí → negro | Auto (flujo); ☐ en vivo |

### Personas (filtro híbrido)
| ID | Escenario | Resultado esperado | Estado |
|---|---|---|---|
| P-01 | Usuario frente a la cámara, sin objeto | Ninguna recomendación | Medido en imagen: sin filtro "papel 88 %"; con filtro, descartado. ☐ En vivo |
| P-02 | Persona sosteniendo una botella a un brazo de distancia | Se conserva la botella | Auto (objeto sostenido); ☐ en vivo |
| P-03 | Objeto grande muy cerca de la cámara | Riesgo: puede descartarse por IoU > 0.3; si ocurre, subir `--iou-personas` | ☐ Pendiente |
| P-04 | Dos personas | Ninguna recomendación | Medido en imagen |

### Estabilización e interfaz
| ID | Escenario | Resultado esperado | Estado |
|---|---|---|---|
| S-01 | Mostrar un objeto | La recomendación aparece tras 5 fotogramas | Medido con cámara simulada; ☐ en vivo |
| S-02 | Mantener el objeto | No se repite ni parpadea | Medido con cámara simulada |
| S-03 | Retirar el objeto | El panel se vacía | Medido con cámara simulada |
| S-04 | Responder la pregunta | La tarjeta se actualiza; se conserva mientras el objeto siga | Medido (interfaz virtual) |
| S-05 | Panel y cajas | El panel no tapa la imagen ni las cajas | Medido (captura de pantalla); ☐ en el monitor de la demo |

### Incertidumbre y errores
| ID | Escenario | Resultado esperado | Estado |
|---|---|---|---|
| E-01 | Objeto fuera de alcance (taza, perro) | "No reconocido"; sin categoría; nunca se nombra en inglés | Auto |
| E-02 | Poca luz o objeto lejano | Baja confianza: sin recomendación definitiva | ☐ Pendiente |
| E-03 | Cámara desconectada u ocupada | Aviso claro; la app sigue abierta | Auto (cámara simulada); ☐ real |
| E-04 | Archivo de modelo ausente | Error claro; no se inventan detecciones | Auto |
| E-05 | `waste_rules.json` malformado o con clases inexistentes | Error que lista todos los problemas | Auto |
| E-06 | Sin conexión a internet | Funciona igual | Medido (entorno sin red) |
| E-07 | Cerrar la ventana con la cámara activa | La cámara se libera | Medido con cámara simulada |

## 5.1 Hallazgo: confusión entre vidrio y metal

Al probar con imágenes sueltas se observó que un **vaso de vidrio** se detectaba como
*metal* (82-88 %). Se investigó:

| Prueba | Resultado |
|---|---|
| 120 imágenes de vidrio de TrashNet (botellas, frascos) | 119 `glass`, 1 `metal` |
| Vaso de vidrio: original, espejo, +brillo, +contraste, recorte, 50 % de tamaño | `metal` 0.86-0.89 en todas; `glass` 0.00 |
| Preprocesamiento con relleno centrado o arriba-izquierda | Mismo resultado: no es un fallo del código |

Conclusión: el modelo reconoce envases de vidrio, pero un vaso cilíndrico se
parece por forma a una lata. No se corrige con un umbral. Como ambos materiales
van al recipiente blanco, se unieron en la regla `glass_metal_001` y se muestran
como "vidrio o metal". Una solución de fondo exigiría reentrenar con vasos de
vidrio, fuera del alcance de este prototipo.

## 5.2 Hallazgo: botella transparente en la mano

Una botella de limpiapisos transparente, sostenida a mano, con fondo desordenado,
**no se reconocía**: la interfaz decía "no se detectaron objetos".

| Medición | Resultado |
|---|---|
| Modelo de residuos, mejor puntaje de `plastic` | 0.24 (por debajo del umbral de 0.25) |
| Con la imagen girada ±90°, más brillo o contraste, recortada o sin la mano | 0.27 a 0.43, siempre por debajo de 0.5 |
| YOLOX-nano (COCO) | `bottle` 0.75 |
| YOLOX-s (COCO) | `bottle` 0.90 |

Causa: el modelo de residuos falla con plásticos transparentes sostenidos a mano
(otro dominio que sus datos de entrenamiento). Solución: el detector híbrido acepta
las clases COCO `bottle`, `banana`, `apple`, `orange`, `broccoli` y `carrot` como
residuos; ahora se muestra "botella 75 %" con la franja blanca y el aviso de
verificar material y contenido. Si el modelo de residuos reconoce el mismo objeto
con confianza ≥ 0.5, se conserva su detección y no se duplica.

Una observación del mismo análisis: YOLOX-s dibujó una caja de persona que cubría
toda la imagen (mano y cuerpo), con la que un residuo grande tendría IoU > 0.3. Con
YOLOX-nano las cajas fueron más pequeñas. Vigilar este riesgo (caso P-03).

## 6. Riesgos conocidos para la demostración

1. **Empaque de snack no reconocido** (N-01): si el modelo no lo detecta, no hay
   pregunta. Probar los empaques exactos antes de la exposición y, si falla,
   mostrarlo más cerca o con mejor luz; es la limitación principal del modelo.
2. **Falsos positivos** sobre objetos que no son residuos (el modelo siempre
   elige un material). Mantener el encuadre despejado y fondo simple.
3. **Descarte indebido** de un objeto grande por el filtro de personas (P-03).
4. **Rendimiento** del equipo de la demostración (sección 3).
5. **Licencia del modelo** de residuos sin confirmar (ver `models/README.md`).
