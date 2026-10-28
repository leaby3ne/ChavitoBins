import pytest

from src.classifier import (
    DEFAULT_MIN_CONFIDENCE,
    ClassificationStatus,
    WasteClassifier,
    group_recommendations,
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
            "classes": ["banana", "apple"],
            "names": {"banana": "banana", "apple": "manzana"},
            "category": "green_bin",
            "recommendation": "Va en el recipiente verde.",
        },
        {
            "id": "bottle_001",
            "classes": ["bottle"],
            "names": {"bottle": "botella"},
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
    assert result.display_name == "banana"
    assert result.message == "Va en el recipiente verde."
    assert result.warnings == ()


def test_display_name_is_spanish_not_model_label(classifier):
    assert classifier.classify(make_detection("apple", 0.9)).display_name == "manzana"


def test_rule_requiring_verification_keeps_warnings(classifier):
    result = classifier.classify(make_detection("bottle", 0.8))

    assert result.status is ClassificationStatus.NEEDS_VERIFICATION
    assert result.category.id == "white_bin"
    assert result.warnings == ("Verifica el material.",)


def test_unknown_class_is_unrecognized_and_not_named(classifier):
    result = classifier.classify(make_detection("dog", 0.95))

    assert result.status is ClassificationStatus.NO_RULE
    assert result.category is None
    assert result.rule_id is None
    assert result.display_name is None
    assert "dog" not in result.message  # nunca se muestra la etiqueta en inglés


def test_unknown_class_stays_unrecognized_even_with_low_confidence(classifier):
    result = classifier.classify(make_detection("dog", 0.3))

    assert result.status is ClassificationStatus.NO_RULE


def test_low_confidence_is_never_definitive_even_for_known_class(classifier):
    result = classifier.classify(make_detection("bottle", 0.3))

    assert result.status is ClassificationStatus.LOW_CONFIDENCE
    assert result.category is None
    assert result.rule_id is None
    assert "botella" in result.message  # nombre en español
    assert "bottle" not in result.message
    assert "30%" in result.message


def test_confidence_exactly_at_threshold_is_accepted(classifier):
    result = classifier.classify(make_detection("banana", 0.5))

    assert result.status is ClassificationStatus.RECOMMENDED


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


def test_group_merges_objects_sharing_a_rule_without_repeating_names(classifier):
    results = classifier.classify_all([
        make_detection("banana", 0.9),
        make_detection("apple", 0.8),
        make_detection("banana", 0.7),  # duplicado del detector
    ])

    groups = group_recommendations(results)

    assert len(groups) == 1
    assert groups[0].rule_id == "organic_001"
    assert groups[0].names == ("banana", "manzana")


def test_group_keeps_one_recommendation_per_rule_in_order(classifier):
    results = classifier.classify_all([
        make_detection("bottle", 0.9),
        make_detection("banana", 0.9),
        make_detection("bottle", 0.6),
    ])

    groups = group_recommendations(results)

    assert [g.rule_id for g in groups] == ["bottle_001", "organic_001"]
    assert groups[0].status is ClassificationStatus.NEEDS_VERIFICATION


def test_group_ignores_unrecognized_and_low_confidence(classifier):
    results = classifier.classify_all([
        make_detection("dog", 0.95),
        make_detection("banana", 0.2),
    ])

    assert group_recommendations(results) == []


DATA_WITH_FOLLOW_UP = {
    **DATA,
    "categories": {
        **DATA["categories"],
        "black_bin": {"name": "Recipiente negro", "description": "No aprovechables."},
    },
    "rules": [
        *DATA["rules"],
        {
            "id": "plastic_001",
            "classes": ["plastic"],
            "names": {"plastic": "plástico"},
            "category": "white_bin",
            "recommendation": "Plástico limpio al blanco.",
            "warnings": ["Debe estar limpio."],
            "follow_up": {
                "question": "¿Es un empaque de snack?",
                "yes_category": "black_bin",
                "yes_recommendation": "Los empaques metalizados van al negro.",
                "yes_warnings": ["Consulta a tu servicio de aseo."],
            },
        },
    ],
}


@pytest.fixture
def follow_up_classifier() -> WasteClassifier:
    return WasteClassifier(KnowledgeBase.from_dict(DATA_WITH_FOLLOW_UP), min_confidence=0.5)


def plastic_recommendation(classifier):
    results = classifier.classify_all([make_detection("plastic", 0.9)])
    return group_recommendations(results)[0]


def test_recommendation_carries_the_pending_question(follow_up_classifier):
    recommendation = plastic_recommendation(follow_up_classifier)

    assert recommendation.category.id == "white_bin"
    assert recommendation.follow_up.question == "¿Es un empaque de snack?"


def test_answering_yes_moves_the_recommendation_to_the_black_bin(follow_up_classifier):
    recommendation = plastic_recommendation(follow_up_classifier)

    answered = follow_up_classifier.answer_follow_up(recommendation, yes=True)

    assert answered.category.id == "black_bin"
    assert answered.message == "Los empaques metalizados van al negro."
    assert answered.warnings == ("Consulta a tu servicio de aseo.",)
    assert answered.status is ClassificationStatus.RECOMMENDED
    assert answered.follow_up is None
    assert answered.names == recommendation.names


def test_answering_no_keeps_the_original_recommendation(follow_up_classifier):
    recommendation = plastic_recommendation(follow_up_classifier)

    answered = follow_up_classifier.answer_follow_up(recommendation, yes=False)

    assert answered.category.id == "white_bin"
    assert answered.message == recommendation.message
    assert answered.follow_up is None


def test_answering_a_recommendation_without_question_raises(follow_up_classifier):
    results = follow_up_classifier.classify_all([make_detection("banana", 0.9)])
    recommendation = group_recommendations(results)[0]

    with pytest.raises(ValueError):
        follow_up_classifier.answer_follow_up(recommendation, yes=True)


def test_shipped_rules_show_glass_and_metal_with_the_same_name_and_bin():
    # El modelo confunde vidrio con metal (por ejemplo, un vaso de vidrio); ambos
    # van al recipiente blanco, así que no se debe afirmar uno solo de los dos.
    from pathlib import Path

    rules_file = Path(__file__).resolve().parents[1] / "data" / "waste_rules.json"
    classifier = WasteClassifier(KnowledgeBase.load(rules_file), min_confidence=0.5)

    as_glass = classifier.classify(make_detection("glass", 0.9))
    as_metal = classifier.classify(make_detection("metal", 0.9))

    assert as_glass.display_name == as_metal.display_name == "vidrio o metal"
    assert as_glass.rule_id == as_metal.rule_id
    assert as_glass.category.id == as_metal.category.id == "white_bin"
    groups = group_recommendations([as_glass, as_metal])
    assert len(groups) == 1 and groups[0].names == ("vidrio o metal",)
