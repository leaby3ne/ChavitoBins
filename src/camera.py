
"""Captura de cámara de baja latencia y carga de modelo ONNX."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from collections.abc import Callable
from types import TracebackType
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


# Configuracion


DEFAULT_CAMERA_INDEX = 0
DEFAULT_INPUT_SIZE = 640
DEFAULT_MODEL_PATH = Path("modelo.onnx")


# EXCEPCIONES


class CameraError(RuntimeError):
    """Error al abrir la cámara o recibir fotogramas."""


class ModelLoadError(RuntimeError):
    """Error al cargar o configurar el modelo ONNX."""



# CÁMARA OPTIMIZADA


def default_backend() -> int:
    """Selecciona el backend de captura según el sistema operativo."""
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class Camera:
    """Cámara web con captura continua en segundo plano."""

    def __init__(
        self,
        index: int = DEFAULT_CAMERA_INDEX,
        backend: int | None = None,
        capture_factory: Callable[[int, int], Any] = cv2.VideoCapture,
    ) -> None:
        self._index = index
        self._backend = (
            default_backend() if backend is None else backend
        )
        self._capture_factory = capture_factory
        self._capture: Any | None = None

        self._frame: np.ndarray | None = None
        self._success = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def open(self) -> None:
        """Abre la cámara e inicia el hilo de captura."""

        if self._running:
            return

        capture = self._capture_factory(
            self._index,
            self._backend,
        )

        # Solicitar MJPEG para reducir el tráfico USB.
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        # Resolución y FPS solicitados.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Desactivar el enfoque automático si el controlador lo permite.
        try:
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass

        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"No se pudo abrir la cámara (índice {self._index}). "
                "Comprueba la conexión, los permisos y que otra "
                "aplicación no la esté utilizando."
            )

        self._capture = capture

        # Obtener el primer fotograma.
        success, frame = capture.read()

        if not success or frame is None:
            capture.release()
            self._capture = None
            raise CameraError(
                "La cámara abrió, pero no entregó un fotograma inicial."
            )

        with self._lock:
            self._frame = frame
            self._success = True

        self._running = True
        self._thread = threading.Thread(
            target=self._update_loop,
            name="CameraCapture",
            daemon=True,
        )
        self._thread.start()

    def _update_loop(self) -> None:
        """Mantiene actualizado el fotograma más reciente."""

        while self._running and self._capture is not None:
            success, frame = self._capture.read()

            with self._lock:
                if success and frame is not None:
                    self._frame = frame
                    self._success = True
                else:
                    self._success = False

    def read(self) -> np.ndarray:
        """Devuelve una copia del fotograma más reciente."""

        if not self._running or self._capture is None:
            raise CameraError(
                "La cámara no está abierta. Llama a open() primero."
            )

        with self._lock:
            if not self._success or self._frame is None:
                raise CameraError(
                    "La cámara no está entregando fotogramas."
                )

            return self._frame.copy()

    def release(self) -> None:
        """Detiene el hilo y libera los recursos de la cámara."""

        self._running = False

        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            self._thread.join(timeout=2.0)

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._thread = None

        with self._lock:
            self._frame = None
            self._success = False

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


# MODELO ONNX OPTIMIZADO PARA CPU


class Detector:
    """Carga el modelo ONNX y permite ejecutar inferencias."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        self._input_size = input_size
        self.session = self._load_session(Path(model_path))

        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

    def _load_session(
        self,
        model_path: Path,
    ) -> ort.InferenceSession:
        """Carga el modelo con optimizaciones de ONNX Runtime."""

        if not model_path.is_file():
            raise ModelLoadError(
                f"No se encontró el modelo: {model_path.resolve()}"
            )

        try:
            options = ort.SessionOptions()

            # Configuración de rendimiento de CPU.
            options.intra_op_num_threads = os.cpu_count() or 4
            options.inter_op_num_threads = 2
            options.execution_mode = (
                ort.ExecutionMode.ORT_SEQUENTIAL
            )
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

        except Exception as exc:
            raise ModelLoadError(
                f"No se pudo cargar el modelo '{model_path}': {exc}"
            ) from exc

        # Comprobar que la entrada sea compatible con NCHW.
        model_input = session.get_inputs()[0]
        actual_shape = model_input.shape
        expected_shape = [
            1,
            3,
            self._input_size,
            self._input_size,
        ]

        # Comprobar dimensiones estáticas sin rechazar las dinámicas.
        if (
            len(actual_shape) != 4
            or actual_shape[1] not in (3, "3")
        ):
            raise ModelLoadError(
                f"Entrada incompatible: {actual_shape}. "
                "Se esperaba una entrada NCHW con 3 canales."
            )

        if all(isinstance(dim, int) for dim in actual_shape):
            if actual_shape != expected_shape:
                raise ModelLoadError(
                    f"Entrada del modelo: {actual_shape}; "
                    f"tamaño esperado: {expected_shape}. "
                    "Ajusta input_size al tamaño de entrada del modelo."
                )

        return session

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convierte el fotograma BGR en un tensor NCHW normalizado."""

        resized = cv2.resize(
            frame,
            (self._input_size, self._input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return np.ascontiguousarray(tensor)

    def predict(self, frame: np.ndarray) -> list[np.ndarray]:
        """Ejecuta la inferencia y devuelve las salidas originales."""

        input_tensor = self.preprocess(frame)

        outputs = self.session.run(
            self._output_names,
            {self._input_name: input_tensor},
        )

        return outputs



# PROGRAMA PRINCIPAL


def main() -> None:
    print("Cargando modelo ONNX...")

    try:
        detector = Detector(
            model_path=DEFAULT_MODEL_PATH,
            input_size=DEFAULT_INPUT_SIZE,
        )

        print("Modelo cargado correctamente.")
        print("Abriendo cámara...")

        with Camera(index=DEFAULT_CAMERA_INDEX) as camera:
            print("Cámara activa. Pulsa Q o ESC para salir.")

            while True:
                try:
                    frame = camera.read()
                except CameraError as exc:
                    print(f"Error de captura: {exc}")
                    break

                try:
                    outputs = detector.predict(frame)

                    # Aquí se integran decode_predictions(),
                    # build_grids() y el postprocesamiento
                    # específico de tu modelo.

                    # Visualizar temporalmente las salidas.
                    # Elimina este bloque cuando integres
                    # tu función de decodificación.
                    if outputs:
                        cv2.putText(
                            frame,
                            "Inferencia ONNX activa",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                except Exception as exc:
                    print(f"Error durante la inferencia: {exc}")
                    break

                cv2.imshow("Deteccion en tiempo real", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except (CameraError, ModelLoadError) as exc:
        print(f"Error: {exc}")

    finally:
        cv2.destroyAllWindows()
        print("Programa finalizado.")


if __name__ == "__main__":
    main()
