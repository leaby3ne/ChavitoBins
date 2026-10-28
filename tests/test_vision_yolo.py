from pathlib import Path

import numpy as np
import pytest

from src.model_loader import create_detector
from src.preprocessing import load_image_bgr
from src.vision import InferenceError, ModelLoadError, YoloxDetector
from src.vision_yolo import (
    YoloDetector,
    decode_predictions,
    is_ultralytics_export,
    parse_class_names,
)

WASTE_MODEL = Path("models/waste_yolo11n.onnx")
YOLOX_MODEL = Path("models/yolox_s.onnx")
APPLE_IMAGE = Path("samples/apple.jpg")
CLASS_COUNT = 6


def empty_raw() -> np.ndarray:
    return np.zeros((4 + CLASS_COUNT, 8400), dtype=np.float32)


def test_parse_class_names_lowercases_and_orders_by_index():
    names = parse_class_names("{0: 'BIODEGRADABLE', 1: 'CARDBOARD', 2: 'Glass'}")

    assert names == ("biodegradable", "cardboard", "glass")


@pytest.mark.parametrize(
    "raw_names",
    ["no es un diccionario", "[1, 2]", "{}", "{0: 'A', 2: 'B'}", "{0: ''}", "{0: 5}"],
)
def test_parse_class_names_rejects_invalid_metadata(raw_names):
    with pytest.raises(ModelLoadError):
        parse_class_names(raw_names)


def test_parse_class_names_does_not_execute_code():
    with pytest.raises(ModelLoadError):
        parse_class_names("__import__('os').system('echo hola')")


def test_is_ultralytics_export_depends_on_metadata():
    assert is_ultralytics_export({"names": "{0: 'a'}", "task": "detect"})
    assert not is_ultralytics_export({})
    assert not is_ultralytics_export({"names": "{0: 'a'}", "task": "segment"})


def test_decode_maps_box_and_class_back_to_original_pixels():
    raw = empty_raw()
    raw[0:4, 0] = (100, 100, 40, 20)  # cx, cy, w, h en píxeles del tensor
    raw[4 + 2, 0] = 0.9  # clase 2

    boxes, scores, class_ids = decode_predictions(
        raw, ratio=0.5, original_size=(1280, 1280), score_threshold=0.25,
        class_count=CLASS_COUNT,
    )

    assert len(boxes) == 1
    assert class_ids[0] == 2
    assert scores[0] == pytest.approx(0.9)
    # (80, 90, 120, 110) dividido entre ratio 0.5
    assert boxes[0] == pytest.approx([160, 180, 240, 220])


def test_decode_filters_low_scores():
    raw = empty_raw()
    raw[0:4, 0] = (100, 100, 40, 20)
    raw[4, 0] = 0.2

    boxes, _, _ = decode_predictions(raw, 1.0, (640, 640), 0.25, CLASS_COUNT)

    assert len(boxes) == 0


def test_decode_clips_boxes_to_image_borders():
    raw = empty_raw()
    raw[0:4, 0] = (5, 5, 100, 100)  # sobresale por arriba y a la izquierda
    raw[4, 0] = 0.9

    boxes, _, _ = decode_predictions(raw, 1.0, (200, 300), 0.25, CLASS_COUNT)

    assert boxes[0][0] == 0 and boxes[0][1] == 0


def test_decode_rejects_unexpected_output_shape():
    with pytest.raises(InferenceError):
        decode_predictions(np.zeros((85, 8400), dtype=np.float32), 1.0, (640, 640), 0.25, CLASS_COUNT)


def test_detector_missing_model_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        YoloDetector(tmp_path / "missing.onnx")


def test_detector_invalid_model_file_raises(tmp_path):
    fake = tmp_path / "fake.onnx"
    fake.write_bytes(b"not a model")

    with pytest.raises(ModelLoadError):
        YoloDetector(fake)


def test_create_detector_missing_file_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        create_detector(tmp_path / "missing.onnx")


@pytest.mark.skipif(not WASTE_MODEL.is_file(), reason="Needs models/waste_yolo11n.onnx")
def test_waste_model_reads_its_labels_from_metadata():
    labels = YoloDetector(WASTE_MODEL).labels

    assert labels == ("biodegradable", "cardboard", "glass", "metal", "paper", "plastic")


@pytest.mark.skipif(
    not (WASTE_MODEL.is_file() and APPLE_IMAGE.is_file()),
    reason="Needs models/waste_yolo11n.onnx and samples/apple.jpg",
)
def test_waste_model_detects_organic_in_apple_photo():
    detections = YoloDetector(WASTE_MODEL).detect(load_image_bgr(APPLE_IMAGE))

    assert detections[0].label == "biodegradable"
    assert detections[0].confidence > 0.5


@pytest.mark.skipif(not WASTE_MODEL.is_file(), reason="Needs models/waste_yolo11n.onnx")
def test_create_detector_picks_ultralytics_detector_for_waste_model():
    assert isinstance(create_detector(WASTE_MODEL), YoloDetector)


@pytest.mark.skipif(not YOLOX_MODEL.is_file(), reason="Needs models/yolox_s.onnx")
def test_create_detector_picks_yolox_detector_for_coco_model():
    assert isinstance(create_detector(YOLOX_MODEL), YoloxDetector)
