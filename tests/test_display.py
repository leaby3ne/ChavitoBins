from src.classifier import Classification, ClassificationStatus
from src.display import format_classification
from src.knowledge_base import Category
from src.vision import Detection

DETECTION = Detection(class_id=39, label="bottle", confidence=0.8, box=(0, 0, 1, 1))


def test_format_includes_status_category_message_and_warnings():
    result = Classification(
        status=ClassificationStatus.NEEDS_VERIFICATION,
        detection=DETECTION,
        message="Va en el recipiente blanco.",
        warnings=("Verifica el material.",),
        rule_id="bottle_001",
        category=Category("white_bin", "Recipiente blanco", "Aprovechables."),
    )

    text = format_classification(result)

    assert text.startswith("[VERIFICAR] bottle (80%) → Recipiente blanco")
    assert "Va en el recipiente blanco." in text
    assert "⚠ Verifica el material." in text


def test_format_without_category_omits_arrow():
    result = Classification(
        status=ClassificationStatus.NO_RULE, detection=DETECTION, message="Sin regla."
    )

    assert "→" not in format_classification(result)
