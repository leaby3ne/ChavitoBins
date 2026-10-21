import numpy as np
import pytest
from PIL import Image

from src.preprocessing import PAD_VALUE, ImageLoadError, letterbox, load_image_bgr


def test_letterbox_keeps_aspect_ratio_and_pads_bottom():
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # wide image

    result = letterbox(image, input_size=640)

    assert result.tensor.shape == (1, 3, 640, 640)
    assert result.tensor.dtype == np.float32
    assert result.ratio == pytest.approx(3.2)
    assert result.original_size == (100, 200)
    # Resized content is 640x320; rows below it must be padding.
    assert np.all(result.tensor[0, :, 330:, :] == PAD_VALUE)
    assert np.all(result.tensor[0, :, :300, :] == 0)


def test_letterbox_rejects_non_color_images():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), dtype=np.uint8), input_size=640)


def test_load_image_bgr_swaps_channels(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(path)  # pure red in RGB

    image = load_image_bgr(path)

    assert image.shape == (4, 4, 3)
    assert tuple(image[0, 0]) == (0, 0, 255)  # red is the last channel in BGR


def test_load_image_bgr_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image_bgr(tmp_path / "missing.png")


def test_load_image_bgr_invalid_file_raises(tmp_path):
    path = tmp_path / "not_an_image.png"
    path.write_text("this is not an image")

    with pytest.raises(ImageLoadError):
        load_image_bgr(path)
