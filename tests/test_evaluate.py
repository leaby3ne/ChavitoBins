from src.classifier import ClassificationStatus, Recommendation
from src.knowledge_base import Category, FollowUp
from scripts.evaluate import (
    immediate_categories,
    meets_expectation,
    reachable_categories,
)

WHITE = Category("white_bin", "Recipiente blanco", "Aprovechables.")
GREEN = Category("green_bin", "Recipiente verde", "Orgánicos.")


def rec(rule_id, category, follow_up=None) -> Recommendation:
    return Recommendation(
        rule_id=rule_id, status=ClassificationStatus.RECOMMENDED, category=category,
        message="m", warnings=(), names=(rule_id,), follow_up=follow_up,
    )


PLASTIC = rec(
    "plastic", WHITE,
    FollowUp(question="¿Snack?", yes_category_id="black_bin", yes_recommendation="negro"),
)


def test_immediate_vs_reachable_categories():
    assert immediate_categories([PLASTIC]) == {"white_bin"}
    assert reachable_categories([PLASTIC]) == {"white_bin", "black_bin"}


def test_green_and_white_need_the_immediate_recommendation():
    assert meets_expectation("green_bin", [rec("organic", GREEN)])
    assert meets_expectation("white_bin", [PLASTIC])
    assert not meets_expectation("green_bin", [PLASTIC])


def test_black_bin_is_met_when_a_question_leads_to_it():
    assert meets_expectation("black_bin", [PLASTIC])
    assert not meets_expectation("black_bin", [rec("organic", GREEN)])


def test_none_expected_means_no_recommendations_at_all():
    assert meets_expectation(None, [])
    assert not meets_expectation(None, [PLASTIC])
