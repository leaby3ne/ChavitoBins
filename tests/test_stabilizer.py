import pytest

from src.classifier import ClassificationStatus, Recommendation
from src.knowledge_base import Category
from src.stabilizer import RecommendationStabilizer

CATEGORY = Category("green_bin", "Recipiente verde", "Orgánicos.")


def rec(rule_id: str) -> Recommendation:
    return Recommendation(
        rule_id=rule_id,
        status=ClassificationStatus.RECOMMENDED,
        category=CATEGORY,
        message="Mensaje.",
        warnings=(),
        names=(rule_id,),
    )


def ids(recommendations) -> list[str]:
    return [r.rule_id for r in recommendations]


def test_confirms_only_after_required_frames():
    stabilizer = RecommendationStabilizer(confirm_frames=3, release_frames=5)

    assert stabilizer.update([rec("a")]) == []
    assert stabilizer.update([rec("a")]) == []
    assert ids(stabilizer.update([rec("a")])) == ["a"]


def test_does_not_repeat_while_object_stays_in_view():
    stabilizer = RecommendationStabilizer(confirm_frames=2, release_frames=5)
    stabilizer.update([rec("a")])
    assert ids(stabilizer.update([rec("a")])) == ["a"]

    for _ in range(10):
        assert stabilizer.update([rec("a")]) == []


def test_short_dropout_keeps_count_and_does_not_reannounce():
    stabilizer = RecommendationStabilizer(confirm_frames=2, release_frames=3)
    stabilizer.update([rec("a")])
    assert ids(stabilizer.update([rec("a")])) == ["a"]

    stabilizer.update([])  # parpadeo de un fotograma
    assert stabilizer.update([rec("a")]) == []


def test_announces_again_after_object_left_long_enough():
    stabilizer = RecommendationStabilizer(confirm_frames=2, release_frames=3)
    stabilizer.update([rec("a")])
    stabilizer.update([rec("a")])

    for _ in range(3):
        stabilizer.update([])  # el objeto se fue

    stabilizer.update([rec("a")])
    assert ids(stabilizer.update([rec("a")])) == ["a"]


def test_isolated_false_positive_is_never_announced():
    stabilizer = RecommendationStabilizer(confirm_frames=3, release_frames=2)

    assert stabilizer.update([rec("a")]) == []
    assert stabilizer.update([]) == []
    assert stabilizer.update([]) == []  # se olvida
    assert stabilizer.update([rec("a")]) == []  # el conteo empieza de cero
    assert stabilizer.update([rec("a")]) == []


def test_rules_are_tracked_independently():
    stabilizer = RecommendationStabilizer(confirm_frames=2, release_frames=5)
    stabilizer.update([rec("a")])

    assert ids(stabilizer.update([rec("a"), rec("b")])) == ["a"]
    assert ids(stabilizer.update([rec("a"), rec("b")])) == ["b"]


def test_reset_clears_state():
    stabilizer = RecommendationStabilizer(confirm_frames=1, release_frames=5)
    assert ids(stabilizer.update([rec("a")])) == ["a"]

    stabilizer.reset()

    assert ids(stabilizer.update([rec("a")])) == ["a"]


@pytest.mark.parametrize("confirm,release", [(0, 5), (3, 0), (-1, 1)])
def test_invalid_parameters_are_rejected(confirm, release):
    with pytest.raises(ValueError):
        RecommendationStabilizer(confirm_frames=confirm, release_frames=release)


def test_active_rule_ids_lists_only_confirmed_rules_still_in_scene():
    stabilizer = RecommendationStabilizer(confirm_frames=2, release_frames=2)
    stabilizer.update([rec("a"), rec("b")])
    stabilizer.update([rec("a")])  # a se confirma; b se ve solo una vez

    assert stabilizer.active_rule_ids == frozenset({"a"})

    stabilizer.update([])
    stabilizer.update([])  # a lleva dos fotogramas ausente: sale de escena

    assert stabilizer.active_rule_ids == frozenset()
