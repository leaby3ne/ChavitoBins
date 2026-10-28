from src.classifier import Classification, ClassificationStatus, Recommendation
from src.display import format_debug_line, format_recommendation
from src.knowledge_base import Category
from src.vision import Detection

CATEGORY = Category("white_bin", "Recipiente blanco", "Aprovechables.")


def test_format_recommendation_includes_tag_names_category_and_warnings():
    recommendation = Recommendation(
        rule_id="bottle_001",
        status=ClassificationStatus.NEEDS_VERIFICATION,
        category=CATEGORY,
        message="Va en el recipiente blanco.",
        warnings=("Verifica el material.",),
        names=("botella", "libro"),
    )

    text = format_recommendation(recommendation)

    assert text.startswith("[VERIFICAR] botella, libro → Recipiente blanco")
    assert "Va en el recipiente blanco." in text
    assert "⚠ Verifica el material." in text


def test_debug_line_shows_original_model_label():
    result = Classification(
        status=ClassificationStatus.NO_RULE,
        detection=Detection(class_id=16, label="dog", confidence=0.91, box=(0, 0, 1, 1)),
        message="Objeto no reconocido.",
    )

    line = format_debug_line(result)

    assert "dog" in line
    assert "91%" in line
    assert "no_rule" in line


def test_format_recommendation_shows_follow_up_question_and_black_bin_answer():
    from src.knowledge_base import FollowUp

    recommendation = Recommendation(
        rule_id="plastic_001",
        status=ClassificationStatus.RECOMMENDED,
        category=CATEGORY,
        message="Plástico limpio al blanco.",
        warnings=(),
        names=("plástico",),
        follow_up=FollowUp(
            question="¿Es un empaque de snack?",
            yes_category_id="black_bin",
            yes_recommendation="Va al negro.",
        ),
    )

    text = format_recommendation(recommendation)

    assert "? ¿Es un empaque de snack?" in text
    assert "Si la respuesta es sí: Va al negro." in text
