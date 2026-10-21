import copy
import json
from pathlib import Path

import pytest

from src.coco_labels import COCO_CLASSES
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError

RULES_FILE = Path(__file__).resolve().parents[1] / "data" / "waste_rules.json"

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


def test_shipped_rules_file_is_valid_and_uses_real_model_labels():
    kb = KnowledgeBase.load(RULES_FILE, known_labels=COCO_CLASSES)

    assert len(kb.rules) > 0
