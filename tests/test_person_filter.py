from pathlib import Path

import pytest

from src.model_loader import create_detector_stack
from src.person_filter import (
    PersonFilteredDetector,
    box_iou,
    covered_fraction,
    is_person_false_positive,
)
from src.preprocessing import load_image_bgr
from src.vision import Detection

WASTE_MODEL = Path("models/waste_yolo11n.onnx")
PERSON_MODEL = Path("models/yolox_nano.onnx")
ZIDANE = Path("samples/zidane.jpg")
APPLE = Path("samples/apple.jpg")

PERSON_BOX = (100.0, 100.0, 500.0, 700.0)  # persona grande en la escena


def detection(label: str, box, confidence: float = 0.9) -> Detection:
    return Detection(class_id=0, label=label, confidence=confidence, box=tuple(box))


class FakeDetector:
    """Detector falso que devuelve siempre lo mismo y cuenta sus llamadas."""

    def __init__(self, detections, labels=("paper",)):
        self._detections = detections
        self._labels = labels
        self.calls = 0

    @property
    def labels(self):
        return self._labels

    def detect(self, image_bgr):
        self.calls += 1
        return list(self._detections)


def make_filter(waste_dets, person_dets, **kwargs):
    waste = FakeDetector(waste_dets)
    person = FakeDetector(person_dets, labels=("person", "dog"))
    return PersonFilteredDetector(waste, person, **kwargs), waste, person


def test_box_iou_and_covered_fraction():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert covered_fraction((0, 0, 10, 10), (0, 0, 20, 20)) == pytest.approx(1.0)
    assert covered_fraction((0, 0, 10, 10), (5, 0, 20, 10)) == pytest.approx(0.5)


def test_waste_box_overlapping_a_person_is_a_false_positive():
    box = (100, 100, 450, 650)  # casi la misma caja que la persona

    assert is_person_false_positive(box, [PERSON_BOX])


def test_waste_box_covering_a_whole_person_is_a_false_positive():
    box = (0, 0, 1280, 720)  # caja gigante que contiene a la persona

    assert is_person_false_positive(box, [PERSON_BOX])


def test_small_object_held_in_front_of_a_person_is_kept():
    box = (250, 300, 330, 420)  # dentro de la persona, pero mucho más pequeño

    assert not is_person_false_positive(box, [PERSON_BOX])


def test_object_far_from_any_person_is_kept():
    assert not is_person_false_positive((600, 100, 700, 200), [PERSON_BOX])


def test_no_persons_means_nothing_is_discarded():
    assert not is_person_false_positive((0, 0, 1280, 720), [])


def test_thresholds_are_configurable():
    box = (250, 100, 700, 700)  # IoU moderado con la persona
    assert is_person_false_positive(box, [PERSON_BOX], iou_threshold=0.1)
    assert not is_person_false_positive(box, [PERSON_BOX], iou_threshold=0.9)


def test_filter_discards_only_waste_detections_on_a_person():
    on_person = detection("paper", (100, 100, 450, 650))
    held_object = detection("plastic", (250, 300, 330, 420))
    detector, _, _ = make_filter(
        [on_person, held_object], [detection("person", PERSON_BOX)]
    )

    kept = detector.detect(None)

    assert kept == [held_object]
    assert detector.last_discarded == 1


def test_filter_ignores_non_person_detections_from_the_coco_model():
    waste_det = detection("paper", (100, 100, 450, 650))
    detector, _, _ = make_filter([waste_det], [detection("dog", PERSON_BOX)])

    assert detector.detect(None) == [waste_det]


def test_labels_come_from_the_waste_detector():
    detector, _, _ = make_filter([], [])

    assert detector.labels == ("paper",)


def test_person_model_is_not_run_when_there_is_nothing_to_filter():
    detector, _, person = make_filter([], [detection("person", PERSON_BOX)])

    detector.detect(None)

    assert person.calls == 0


def test_person_boxes_are_reused_between_refreshes():
    detector, _, person = make_filter(
        [detection("paper", (0, 0, 10, 10))], [], refresh_frames=3
    )

    for _ in range(3):
        detector.detect(None)

    assert person.calls == 1


def test_refresh_frames_one_runs_person_model_every_time():
    detector, _, person = make_filter(
        [detection("paper", (0, 0, 10, 10))], [], refresh_frames=1
    )

    for _ in range(3):
        detector.detect(None)

    assert person.calls == 3


def test_invalidate_cache_forces_a_new_person_search():
    detector, _, person = make_filter(
        [detection("paper", (0, 0, 10, 10))], [], refresh_frames=10
    )
    detector.detect(None)
    detector.detect(None)
    assert person.calls == 1

    detector.invalidate_cache()
    detector.detect(None)

    assert person.calls == 2


def test_invalid_refresh_frames_is_rejected():
    with pytest.raises(ValueError):
        make_filter([], [], refresh_frames=0)


@pytest.mark.skipif(
    not (WASTE_MODEL.is_file() and PERSON_MODEL.is_file() and ZIDANE.is_file() and APPLE.is_file()),
    reason="Needs both models and samples/zidane.jpg, samples/apple.jpg",
)
def test_real_models_discard_waste_false_positive_on_people_but_keep_fruit():
    stack = create_detector_stack(WASTE_MODEL, PERSON_MODEL, refresh_frames=1)
    zidane = load_image_bgr(ZIDANE)
    apple = load_image_bgr(APPLE)

    without_filter = create_detector_stack(WASTE_MODEL, None).detect(zidane)
    with_filter = stack.detect(zidane)
    stack.invalidate_cache()

    assert len(without_filter) > 0  # el modelo de residuos sí se equivoca sin filtro
    assert with_filter == []
    assert stack.detect(apple)[0].label == "biodegradable"


# ---- Complemento COCO: clases que COCO detecta mejor que el modelo de residuos ----

BOTTLE_BOX = (123.0, 68.0, 268.0, 457.0)


def make_hybrid(waste_dets, coco_dets, **kwargs):
    waste = FakeDetector(waste_dets, labels=("plastic", "glass"))
    coco = FakeDetector(coco_dets, labels=("person", "bottle", "banana", "dog"))
    detector = PersonFilteredDetector(
        waste, coco, complement_labels=("bottle", "banana"), **kwargs
    )
    return detector, waste, coco


def test_complement_adds_coco_bottle_when_waste_model_is_unsure():
    weak_plastic = detection("plastic", BOTTLE_BOX, confidence=0.24)
    coco_bottle = detection("bottle", BOTTLE_BOX, confidence=0.75)
    detector, _, _ = make_hybrid([weak_plastic], [coco_bottle])

    result = detector.detect(None)

    # Se prefiere la botella de COCO y se retira la conjetura débil del otro modelo.
    assert [d.label for d in result] == ["bottle"]


def test_complement_keeps_confident_waste_detection_instead_of_duplicating():
    plastic = detection("plastic", BOTTLE_BOX, confidence=0.8)
    coco_bottle = detection("bottle", BOTTLE_BOX, confidence=0.9)
    detector, _, _ = make_hybrid([plastic], [coco_bottle])

    assert [d.label for d in detector.detect(None)] == ["plastic"]


def test_complement_ignores_coco_labels_that_are_not_listed():
    detector, _, _ = make_hybrid([], [detection("dog", BOTTLE_BOX), detection("person", PERSON_BOX)])

    assert detector.detect(None) == []


def test_complement_works_even_when_the_waste_model_finds_nothing():
    detector, _, _ = make_hybrid([], [detection("banana", (10, 10, 90, 90), 0.8)])

    assert [d.label for d in detector.detect(None)] == ["banana"]


def test_complement_runs_coco_every_frame():
    detector, _, coco = make_hybrid([], [], refresh_frames=5)

    for _ in range(4):
        detector.detect(None)

    assert coco.calls == 4


def test_complement_still_discards_waste_false_positives_on_people():
    on_person = detection("plastic", (100, 100, 450, 650), confidence=0.9)
    detector, _, _ = make_hybrid([on_person], [detection("person", PERSON_BOX)])

    assert detector.detect(None) == []
    assert detector.last_discarded == 1


def test_labels_include_the_complement_labels():
    detector, _, _ = make_hybrid([], [])

    assert detector.labels == ("plastic", "glass", "bottle", "banana")


def test_unknown_complement_label_is_rejected():
    waste = FakeDetector([], labels=("plastic",))
    coco = FakeDetector([], labels=("person",))

    with pytest.raises(ValueError, match="not in the COCO model"):
        PersonFilteredDetector(waste, coco, complement_labels=("bottle",))


@pytest.mark.skipif(
    not (WASTE_MODEL.is_file() and PERSON_MODEL.is_file()),
    reason="Needs models/waste_yolo11n.onnx and models/yolox_nano.onnx",
)
def test_allowed_rule_labels_work_with_and_without_the_complement():
    from src.model_loader import allowed_rule_labels
    from src.person_filter import DEFAULT_COMPLEMENT_LABELS

    plain = create_detector_stack(WASTE_MODEL, PERSON_MODEL, complement_labels=())
    hybrid = create_detector_stack(WASTE_MODEL, PERSON_MODEL)

    assert "bottle" not in plain.labels and "bottle" in hybrid.labels
    for detector in (plain, hybrid):
        allowed = allowed_rule_labels(detector)
        assert all(label in allowed for label in DEFAULT_COMPLEMENT_LABELS)
        assert "bananna" not in allowed  # una errata no se acepta
