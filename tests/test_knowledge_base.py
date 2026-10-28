import copy
import json
from pathlib import Path

import pytest

from src.coco_labels import COCO_CLASSES
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError
from src.vision_yolo import YoloDetector

ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = ROOT / "data" / "waste_rules.json"
COCO_RULES_FILE = ROOT / "data" / "waste_rules_coco.json"
WASTE_MODEL = ROOT / "models" / "waste_yolo11n.onnx"

VALID_DATA = {
    "schema_version": 1,
    "categories": {
        "green_bin": {"name": "Recipiente verde", "description": "Orgánicos."},
        "white_bin": {"name": "Recipiente blanco", "description": "Aprovechables."},
    },
    "rules": [
        {
            "id": "organic_001",
            "classes": ["banana"],
            "names": {"banana": "banana"},
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


def valid_data() -> dict:
    return copy.deepcopy(VALID_DATA)


def error_text(data, known_labels=None) -> str:
    with pytest.raises(KnowledgeBaseError) as exc_info:
        KnowledgeBase.from_dict(data, known_labels)
    return str(exc_info.value)


def test_valid_data_builds_knowledge_base_with_defaults():
    kb = KnowledgeBase.from_dict(valid_data())

    banana = kb.find_rule("banana")
    assert banana is not None
    assert banana.category_id == "green_bin"
    assert banana.warnings == ()  # optional field defaults to empty
    assert banana.requires_verification is False
    assert kb.find_rule("bottle").requires_verification is True
    assert kb.get_category("green_bin").name == "Recipiente verde"


def test_rule_keeps_spanish_display_names():
    rule = KnowledgeBase.from_dict(valid_data()).find_rule("bottle")

    assert rule.names == {"bottle": "botella"}


def test_missing_names_is_rejected():
    data = valid_data()
    del data["rules"][0]["names"]

    message = error_text(data)

    assert "organic_001" in message
    assert "names" in message


def test_names_must_match_classes_exactly():
    data = valid_data()
    data["rules"][0]["classes"] = ["banana", "apple"]  # falta el nombre de apple

    assert "match 'classes' exactly" in error_text(data)


def test_unknown_label_has_no_rule():
    assert KnowledgeBase.from_dict(valid_data()).find_rule("dog") is None


def test_load_reads_valid_file(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(VALID_DATA), encoding="utf-8")

    assert KnowledgeBase.load(path).find_rule("banana") is not None


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="Cannot read"):
        KnowledgeBase.load(tmp_path / "missing.json")


def test_load_malformed_json_raises(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text('{"schema_version": 1, "rules": [', encoding="utf-8")

    with pytest.raises(KnowledgeBaseError, match="not valid JSON"):
        KnowledgeBase.load(path)


def test_top_level_must_be_an_object():
    assert "JSON object" in error_text([])


def test_unsupported_schema_version_is_rejected():
    data = valid_data()
    data["schema_version"] = 99

    assert "schema_version" in error_text(data)


def test_incomplete_rule_reports_missing_field():
    data = valid_data()
    del data["rules"][0]["recommendation"]

    message = error_text(data)

    assert "organic_001" in message
    assert "recommendation" in message


def test_category_without_match_is_rejected():
    data = valid_data()
    data["rules"][0]["category"] = "red_bin"

    assert "red_bin" in error_text(data)


def test_empty_classes_list_is_rejected():
    data = valid_data()
    data["rules"][0]["classes"] = []

    assert "classes" in error_text(data)


def test_invalid_optional_fields_are_rejected():
    data = valid_data()
    data["rules"][1]["warnings"] = "not a list"
    data["rules"][1]["requires_verification"] = "yes"

    message = error_text(data)

    assert "warnings" in message
    assert "requires_verification" in message


def test_duplicate_rule_id_is_rejected():
    data = valid_data()
    data["rules"][1]["id"] = "organic_001"

    assert "duplicate rule id" in error_text(data)


def test_class_covered_by_two_rules_is_rejected():
    data = valid_data()
    data["rules"][1]["classes"] = ["bottle", "banana"]
    data["rules"][1]["names"] = {"bottle": "botella", "banana": "banana"}

    message = error_text(data)

    assert "banana" in message
    assert "already covered" in message


def test_empty_rules_list_is_rejected():
    data = valid_data()
    data["rules"] = []

    assert "'rules'" in error_text(data)


def test_class_not_in_model_labels_is_rejected():
    data = valid_data()
    data["rules"][0]["classes"] = ["bananna"]  # typo

    assert "bananna" in error_text(data, known_labels=COCO_CLASSES)


def test_all_problems_are_reported_together():
    data = valid_data()
    data["rules"][0]["category"] = "red_bin"
    del data["rules"][1]["recommendation"]

    message = error_text(data)

    assert "red_bin" in message
    assert "bottle_001" in message


def test_shipped_coco_rules_use_real_coco_labels():
    kb = KnowledgeBase.load(COCO_RULES_FILE, known_labels=COCO_CLASSES)

    assert len(kb.rules) > 0


def test_shipped_waste_rules_are_structurally_valid():
    kb = KnowledgeBase.load(RULES_FILE)

    assert len(kb.rules) > 0


@pytest.mark.skipif(
    not WASTE_MODEL.is_file(), reason="Needs models/waste_yolo11n.onnx (see models/README.md)"
)
def test_shipped_waste_rules_use_the_real_waste_model_labels():
    from src.person_filter import DEFAULT_COMPLEMENT_LABELS

    waste_labels = YoloDetector(WASTE_MODEL).labels

    # Las reglas cubren las etiquetas del modelo de residuos y las clases COCO
    # complementarias; cualquier otra etiqueta sería un error de escritura.
    kb = KnowledgeBase.load(
        RULES_FILE, known_labels=(*waste_labels, *DEFAULT_COMPLEMENT_LABELS)
    )

    # Cada etiqueta del modelo debe tener regla: ninguna clase queda sin cubrir.
    assert all(kb.find_rule(label) is not None for label in waste_labels)


def with_follow_up(**overrides) -> dict:
    data = valid_data()
    follow_up = {
        "question": "¿Está sucio?",
        "yes_category": "green_bin",
        "yes_recommendation": "Va en el recipiente verde.",
        "yes_warnings": ["Consulta a tu servicio de aseo."],
    }
    follow_up.update(overrides)
    data["rules"][1]["follow_up"] = follow_up
    return data


def test_follow_up_is_parsed():
    follow_up = KnowledgeBase.from_dict(with_follow_up()).find_rule("bottle").follow_up

    assert follow_up.question == "¿Está sucio?"
    assert follow_up.yes_category_id == "green_bin"
    assert follow_up.yes_warnings == ("Consulta a tu servicio de aseo.",)


def test_rule_without_follow_up_has_none():
    assert KnowledgeBase.from_dict(valid_data()).find_rule("banana").follow_up is None


def test_follow_up_with_unknown_category_is_rejected():
    assert "black_bin" in error_text(with_follow_up(yes_category="black_bin"))


def test_follow_up_missing_question_is_rejected():
    assert "follow_up.question" in error_text(with_follow_up(question=""))


def test_follow_up_missing_recommendation_is_rejected():
    assert "yes_recommendation" in error_text(with_follow_up(yes_recommendation=" "))


def test_follow_up_must_be_an_object():
    data = valid_data()
    data["rules"][1]["follow_up"] = "¿Está sucio?"

    assert "'follow_up' must be an object" in error_text(data)


def test_shipped_rules_can_reach_the_black_bin_through_a_question():
    kb = KnowledgeBase.load(RULES_FILE)

    black_routes = [
        rule.id for rule in kb.rules
        if rule.follow_up is not None and rule.follow_up.yes_category_id == "black_bin"
    ]

    assert {"plastic_001", "glass_metal_001", "paper_001"} <= set(black_routes)


def test_shipped_rules_cover_every_complement_label():
    from src.person_filter import DEFAULT_COMPLEMENT_LABELS

    kb = KnowledgeBase.load(RULES_FILE)

    assert all(kb.find_rule(label) is not None for label in DEFAULT_COMPLEMENT_LABELS)
