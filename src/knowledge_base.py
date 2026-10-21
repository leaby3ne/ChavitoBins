"""Carga y validación de la base de conocimiento de reglas de residuos (JSON).

La base de conocimiento relaciona las etiquetas del detector (en inglés, tal
como las emite el modelo) con categorías de disposición y mensajes en español
para el usuario. Todo se valida al cargar, para que los errores de
configuración fallen pronto y con un mensaje claro.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSION = 1


class KnowledgeBaseError(ValueError):
    """Se lanza si el archivo de reglas falta, está mal formado o es inconsistente."""


@dataclass(frozen=True)
class Category:
    """Una categoría de disposición (por ejemplo, un recipiente de color)."""

    id: str
    name: str
    description: str


@dataclass(frozen=True)
class Rule:
    """Relaciona una o más etiquetas del modelo con una categoría y una recomendación."""

    id: str
    classes: tuple[str, ...]
    category_id: str
    recommendation: str
    warnings: tuple[str, ...] = ()
    # True cuando la imagen no basta para decidir (material, contenido, estado):
    # el sistema muestra la recomendación como provisional.
    requires_verification: bool = False


def _is_text(value: Any) -> bool:
    """True si el valor es un texto con contenido (no vacío ni solo espacios)."""
    return isinstance(value, str) and bool(value.strip())


def _parse_categories(raw: Any, errors: list[str]) -> dict[str, Category]:
    if not isinstance(raw, dict) or not raw:
        errors.append("'categories' must be a non-empty object.")
        return {}
    categories: dict[str, Category] = {}
    for category_id, entry in raw.items():
        if (
            not isinstance(entry, dict)
            or not _is_text(entry.get("name"))
            or not _is_text(entry.get("description"))
        ):
            errors.append(
                f"category '{category_id}' needs non-empty 'name' and 'description'."
            )
            continue
        categories[category_id] = Category(
            id=category_id, name=entry["name"], description=entry["description"]
        )
    return categories


def _parse_rule(
    index: int,
    entry: Any,
    categories: dict[str, Category],
    known_labels: frozenset[str] | None,
    errors: list[str],
) -> Rule | None:
    """Valida una regla; agrega los problemas encontrados a ``errors``.

    Se acumulan todos los problemas (en lugar de detenerse en el primero)
    para que quien edita el JSON pueda corregirlos de una sola vez.
    """
    if not isinstance(entry, dict):
        errors.append(f"rules[{index}] must be an object.")
        return None

    name = f"rules[{index}] (id '{entry.get('id')}')"
    problems: list[str] = []

    if not _is_text(entry.get("id")):
        problems.append("missing or empty 'id'")
    classes = entry.get("classes")
    if not (
        isinstance(classes, list)
        and classes
        and all(_is_text(label) for label in classes)
    ):
        problems.append("'classes' must be a non-empty list of strings")
        classes = []
    if not _is_text(entry.get("recommendation")):
        problems.append("missing or empty 'recommendation'")

    category_id = entry.get("category")
    if not _is_text(category_id):
        problems.append("missing or empty 'category'")
    elif category_id not in categories:
        problems.append(f"category '{category_id}' is not defined in 'categories'")

    warnings = entry.get("warnings", [])
    if not (isinstance(warnings, list) and all(_is_text(w) for w in warnings)):
        problems.append("'warnings' must be a list of non-empty strings")
        warnings = []

    requires_verification = entry.get("requires_verification", False)
    if not isinstance(requires_verification, bool):
        problems.append("'requires_verification' must be true or false")
        requires_verification = False

    # Detecta etiquetas mal escritas: si no coinciden con las del modelo, la
    # regla nunca se aplicaría y el error pasaría desapercibido.
    if known_labels is not None:
        for label in classes:
            if label not in known_labels:
                problems.append(f"class '{label}' is not a label of the model")

    if problems:
        errors.extend(f"{name}: {problem}." for problem in problems)
        return None
    return Rule(
        id=entry["id"],
        classes=tuple(classes),
        category_id=category_id,
        recommendation=entry["recommendation"],
        warnings=tuple(warnings),
        requires_verification=requires_verification,
    )


def _parse_rules(
    raw: Any,
    categories: dict[str, Category],
    known_labels: frozenset[str] | None,
    errors: list[str],
) -> list[Rule]:
    if not isinstance(raw, list) or not raw:
        errors.append("'rules' must be a non-empty list.")
        return []

    rules: list[Rule] = []
    rule_ids: set[str] = set()
    class_owner: dict[str, str] = {}
    for index, entry in enumerate(raw):
        rule = _parse_rule(index, entry, categories, known_labels, errors)
        if rule is None:
            continue
        if rule.id in rule_ids:
            errors.append(f"rules[{index}]: duplicate rule id '{rule.id}'.")
            continue
        # Una clase en dos reglas sería ambigua: no sabríamos cuál aplicar.
        duplicated = [label for label in rule.classes if label in class_owner]
        if duplicated:
            owners = {label: class_owner[label] for label in duplicated}
            errors.append(
                f"rules[{index}] (id '{rule.id}'): classes already covered by "
                f"another rule: {owners}."
            )
            continue
        rule_ids.add(rule.id)
        class_owner.update({label: rule.id for label in rule.classes})
        rules.append(rule)
    return rules


class KnowledgeBase:
    """Colección de categorías y reglas ya validadas, de solo lectura."""

    def __init__(self, categories: dict[str, Category], rules: list[Rule]) -> None:
        self._categories = dict(categories)
        self._rules = tuple(rules)
        # Índice etiqueta -> regla para buscar en tiempo constante.
        self._rule_by_label = {
            label: rule for rule in self._rules for label in rule.classes
        }

    @classmethod
    def from_dict(
        cls, data: Any, known_labels: Iterable[str] | None = None
    ) -> KnowledgeBase:
        """Valida datos JSON ya leídos y construye la base de conocimiento.

        Todos los problemas encontrados se reportan juntos en una sola excepción.
        Si se indica ``known_labels``, cada clase de las reglas debe estar en ella.
        """
        if not isinstance(data, dict):
            raise KnowledgeBaseError("Rules data must be a JSON object.")

        errors: list[str] = []
        if data.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
            errors.append(
                f"'schema_version' must be {SUPPORTED_SCHEMA_VERSION}, "
                f"got {data.get('schema_version')!r}."
            )
        labels = frozenset(known_labels) if known_labels is not None else None
        categories = _parse_categories(data.get("categories"), errors)
        rules = _parse_rules(data.get("rules"), categories, labels, errors)

        if errors:
            raise KnowledgeBaseError(
                "Invalid rules data:\n- " + "\n- ".join(errors)
            )
        return cls(categories, rules)

    @classmethod
    def load(
        cls, path: Path, known_labels: Iterable[str] | None = None
    ) -> KnowledgeBase:
        """Lee y valida un archivo JSON de reglas."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise KnowledgeBaseError(f"Cannot read rules file '{path}': {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"Rules file '{path}' is not valid JSON: {exc}"
            ) from exc
        return cls.from_dict(data, known_labels)

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def find_rule(self, label: str) -> Rule | None:
        """Devuelve la regla de una etiqueta del modelo, o None si no tiene."""
        return self._rule_by_label.get(label)

    def get_category(self, category_id: str) -> Category:
        """Devuelve una categoría. Las reglas ya están validadas, así que su id siempre existe."""
        return self._categories[category_id]
