import numpy as np
import pytest

from src.camera import Camera, CameraError


class FakeCapture:
    """Doble de cv2.VideoCapture para probar sin cámara física."""

    def __init__(self, opened: bool = True, frames=None) -> None:
        self._opened = opened
        self._frames = list(frames) if frames is not None else []
        self.released = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released += 1


def make_camera(capture: FakeCapture) -> Camera:
    return Camera(index=0, backend=0, capture_factory=lambda index, backend: capture)


def test_open_and_read_returns_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = make_camera(FakeCapture(frames=[frame]))
    camera.open()

    assert camera.read().shape == (480, 640, 3)


def test_open_failure_raises_and_releases_device():
    capture = FakeCapture(opened=False)

    with pytest.raises(CameraError, match="No se pudo abrir"):
        make_camera(capture).open()

    assert capture.released == 1


def test_read_before_open_raises():
    with pytest.raises(CameraError, match="no está abierta"):
        make_camera(FakeCapture()).read()


def test_read_without_frame_raises():
    camera = make_camera(FakeCapture(frames=[]))
    camera.open()

    with pytest.raises(CameraError, match="ningún fotograma"):
        camera.read()


def test_context_manager_releases_even_when_body_fails():
    capture = FakeCapture(frames=[np.zeros((2, 2, 3), dtype=np.uint8)])

    with pytest.raises(ValueError):
        with make_camera(capture):
            raise ValueError("fallo durante el procesamiento")

    assert capture.released == 1


def test_release_is_idempotent():
    capture = FakeCapture()
    camera = make_camera(capture)
    camera.open()

    camera.release()
    camera.release()

    assert capture.released == 1
