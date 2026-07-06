"""Static regression tests for Neo4j relationship governance.

The ingestion engine and the FastAPI retrieval layer must expose the same
semantic relationship whitelist.  The ingestion file is parsed with ``ast``
so importing it cannot start models, database clients or other heavy resources.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS,
    DEFAULT_NEO4J_RELATIONSHIP_ALIASES,
)


def _ingestion_path() -> Path:
    configured = os.getenv("INGESTION_FILE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    # Expected project layout:
    # assessmentAI/
    #   ingestion.py
    #   rag_api/tests/test_neo4j_relationship_policy.py
    return Path(__file__).resolve().parents[2] / "ingestion.py"


def _literal_assignment(path: Path, variable_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        value = None
        targets: list[ast.expr] = []

        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]

        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in targets
        ):
            continue

        if value is None:
            break

        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and value.args
        ):
            value = value.args[0]

        return ast.literal_eval(value)

    raise AssertionError(f"Variabile {variable_name!r} non trovata in {path}")


def test_ingestion_and_api_relationship_whitelists_are_identical() -> None:
    ingestion = _ingestion_path()
    assert ingestion.is_file(), f"File ingestion non trovato: {ingestion}"

    ingestion_allowed = set(
        _literal_assignment(ingestion, "NEO4J_ALLOWED_RELATIONSHIPS")
    )
    api_allowed = set(DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS)

    assert ingestion_allowed == api_allowed, (
        "Whitelist Neo4j non allineate. "
        f"Solo ingestion={sorted(ingestion_allowed - api_allowed)}; "
        f"solo API={sorted(api_allowed - ingestion_allowed)}"
    )


def test_known_unsafe_or_ambiguous_relationships_are_not_whitelisted() -> None:
    disallowed = {
        "UPDATES",
        "TRANSFORMS_INTO",
        "SELECTS",
        "INVALIDATES",
        "AVOIDS",
        "CONTROLS",
        "DOES_NOT_APPLY_TO",
        "HAS_NO_COMPONENT",
        "DOES_NOT",
        "TRANSFER_TO",
    }

    assert disallowed.isdisjoint(DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS)


def test_ingestion_and_api_relationship_aliases_are_identical() -> None:
    ingestion = _ingestion_path()
    ingestion_aliases = dict(
        _literal_assignment(ingestion, "NEO4J_RELATIONSHIP_ALIASES")
    )

    assert ingestion_aliases == DEFAULT_NEO4J_RELATIONSHIP_ALIASES
    assert ingestion_aliases["IMPLEMENTES"] == "IMPLEMENTS"
    assert set(ingestion_aliases.values()).issubset(
        DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS
    )
