from pathlib import Path

import numpy as np
import pytest

from src.coco_labels import COCO_CLASSES
from src.preprocessing import load_image_bgr
from src.vision import (
    InferenceError,
    ModelLoadError,
    YoloxDetector,
    build_grids,
    decode_predictions,
    non_max_suppression,
)

MODEL_PATH = Path("models/yolox_s.onnx")
SAMPLE_IMAGE = Path("samples/dog.jpg")
BOTTLE = COCO_CLASSES.index("bottle")


def test_build_grids_matches_model_output_rows():
    grid, strides = build_grids(640)

    assert grid.shape == (8400, 2)
    assert strides.shape == (8400, 1)
    assert set(np.unique(strides)) == {8.0, 16.0, 32.0}


def _empty_raw() -> np.ndarray:
    return np.zeros((8400, 5 + len(COCO_CLASSES)), dtype=np.float32)


def test_decode_maps_box_back_to_original_pixels():
    grid, strides = build_grids(640)
    raw = _empty_raw()
    raw[0, :2] = (0.5, 0.5)  # center (4, 4) on the 640 canvas, stride 8
    raw[0, 2:4] = np.log(2.0)  # width = height = 16
    raw[0, 4] = 0.9
    raw[0, 5 + BOTTLE] = 0.8

    boxes, scores, class_ids = decode_predictions(
        raw, grid, strides, ratio=0.5, original_size=(1280, 1280),
        score_threshold=0.25, labels=COCO_CLASSES,
    )

    assert len(boxes) == 1
    assert class_ids[0] == BOTTLE
    assert scores[0] == pytest.approx(0.72)
    # (-4, -4, 12, 12) clipped at 0, then divided by ratio 0.5.
    assert boxes[0] == pytest.approx([0, 0, 24, 24])


def test_decode_filters_low_confidence():
    grid, strides = build_grids(640)
    raw = _empty_raw()
    raw[0, 4] = 0.5
    raw[0, 5 + BOTTLE] = 0.4  # confidence 0.2 < threshold

    boxes, _, _ = decode_predictions(
        raw, grid, strides, 1.0, (640, 640), 0.25, COCO_CLASSES
    )

    assert len(boxes) == 0


def test_decode_rejects_unexpected_output_shape():
    grid, strides = build_grids(640)
    with pytest.raises(InferenceError):
        decode_predictions(
            np.zeros((8400, 10), dtype=np.float32), grid, strides,
            1.0, (640, 640), 0.25, COCO_CLASSES,
        )


def test_nms_suppresses_overlap_only_within_same_class():
    boxes = np.array(
        [[0, 0, 100, 100], [5, 5, 105, 105], [5, 5, 105, 105]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    class_ids = np.array([1, 1, 2])

    kept = non_max_suppression(boxes, scores, class_ids, iou_threshold=0.45)

    assert kept == [0, 2]  # box 1 is a duplicate of box 0; box 2 is another class


def test_detector_missing_model_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        YoloxDetector(tmp_path / "missing.onnx")


def test_detector_invalid_model_file_raises(tmp_path):
    fake = tmp_path / "fake.onnx"
    fake.write_bytes(b"not a model")

    with pytest.raises(ModelLoadError):
        YoloxDetector(fake)


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and SAMPLE_IMAGE.is_file()),
    reason="Needs models/yolox_s.onnx and samples/dog.jpg (see models/README.md)",
)
def test_real_model_detects_known_objects():
    detector = YoloxDetector(MODEL_PATH)

    labels = {d.label for d in detector.detect(load_image_bgr(SAMPLE_IMAGE))}

    assert {"dog", "bicycle"} <= labels


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Needs models/yolox_s.onnx")
def test_yolox_detector_exposes_coco_labels():
    assert YoloxDetector(MODEL_PATH).labels == COCO_CLASSES
