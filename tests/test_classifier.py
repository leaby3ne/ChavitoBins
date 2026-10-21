import pytest

from src.classifier import (
    DEFAULT_MIN_CONFIDENCE,
    ClassificationStatus,
    WasteClassifier,
)
from src.knowledge_base import KnowledgeBase
from src.vision import Detection

DATA = {
    "schema_version": 1,
    "categories": {
        "green_bin": {"name": "Recipiente verde", "description": "Orgánicos."},
        "white_bin": {"name": "Recipiente blanco", "description": "Aprovechables."},
    },
    "rules": [
        {
            "id": "organic_001",
            "classes": ["banana"],
            "category": "green_bin",
            "recommendation": "Va en el recipiente verde.",
        },
        {
            "id": "bottle_001",
            "classes": ["bottle"],
            "category": "white_bin",
            "recommendation": "Va en el recipiente blanco si está limpia y seca.",
            "warnings": ["Verifica el material."],
            "requires_verification": True,
        },
    ],
}


def make_detection(label: str, confidence: float) -> Detection:
    return Detection(class_id=0, label=label, confidence=confidence, box=(0, 0, 10, 10))


@pytest.fixture
def classifier() -> WasteClassifier:
    return WasteClassifier(KnowledgeBase.from_dict(DATA), min_confidence=0.5)


def test_known_class_gets_recommendation_and_category(classifier):
    result = classifier.classify(make_detection("banana", 0.9))

    assert result.status is ClassificationStatus.RECOMMENDED
    assert result.rule_id == "organic_001"
    assert result.category.id == "green_bin"
    assert result.message == "Va en el recipiente verde."
    assert result.warnings == ()


def test_rule_requiring_verification_keeps_warnings(classifier):
    result = classifier.classify(make_detection("bottle", 0.8))

    assert result.status is ClassificationStatus.NEEDS_VERIFICATION
    assert result.category.id == "white_bin"
    assert result.warnings == ("Verifica el material.",)


def test_unknown_class_gets_no_category(classifier):
    result = classifier.classify(make_detection("dog", 0.95))

    assert result.status is ClassificationStatus.NO_RULE
    assert result.category is None
    assert result.rule_id is None
    assert "dog" in result.message


def test_low_confidence_is_never_definitive_even_for_known_class(classifier):
    result = classifier.classify(make_detection("banana", 0.3))

    assert result.status is ClassificationStatus.LOW_CONFIDENCE
    assert result.category is None
    assert result.rule_id is None
    assert "30%" in result.message


def test_confidence_exactly_at_threshold_is_accepted(classifier):
    result = classifier.classify(make_detection("banana", 0.5))

    assert result.status is ClassificationStatus.RECOMMENDED


def test_low_confidence_unknown_class_reports_low_confidence_first(classifier):
    result = classifier.classify(make_detection("dog", 0.3))

    assert result.status is ClassificationStatus.LOW_CONFIDENCE


def test_classify_all_preserves_order(classifier):
    detections = [
        make_detection("bottle", 0.9),
        make_detection("dog", 0.9),
        make_detection("banana", 0.9),
    ]

    results = classifier.classify_all(detections)

    assert [r.detection.label for r in results] == ["bottle", "dog", "banana"]
    assert classifier.classify_all([]) == []


@pytest.mark.parametrize("value", [0, -0.1, 1.5])
def test_invalid_min_confidence_is_rejected(value):
    with pytest.raises(ValueError):
        WasteClassifier(KnowledgeBase.from_dict(DATA), min_confidence=value)


def test_default_min_confidence_is_valid():
    assert 0 < DEFAULT_MIN_CONFIDENCE <= 1
