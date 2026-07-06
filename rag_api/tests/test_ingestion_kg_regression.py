from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


TEST_FILE = Path(__file__).resolve()
RAG_API_ROOT = TEST_FILE.parents[1]
PROJECT_ROOT = RAG_API_ROOT.parent


def _resolve_source_path(
    env_name: str,
    *candidates: Path,
) -> Path:
    """
    Risolve un file sorgente senza dipendere dalla current working directory.

    Ordine:
    1. percorso esplicito da variabile ambiente;
    2. percorsi noti dell'alberatura assessmentAI/rag_api;
    3. errore descrittivo con tutti i path controllati.
    """
    checked: list[Path] = []

    env_value = str(os.getenv(env_name, "") or "").strip()
    if env_value:
        env_path = Path(env_value).expanduser().resolve()
        checked.append(env_path)
        if env_path.is_file():
            return env_path

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return resolved

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"Impossibile trovare il file richiesto per {env_name}. "
        f"Percorsi controllati:\n{checked_text}"
    )


# Alberatura prevista:
# E:\\Dev\\assessmentAI\\ingestion.py
# E:\\Dev\\assessmentAI\\rag_api\\core\\config.py
# E:\\Dev\\assessmentAI\\rag_api\\tests\\test_ingestion_kg_regression.py
INGESTION_PATH = _resolve_source_path(
    "INGESTION_SOURCE_PATH",
    PROJECT_ROOT / "ingestion.py",
    RAG_API_ROOT / "ingestion.py",  # fallback per layout alternativi
    Path.cwd() / "ingestion.py",
    Path.cwd().parent / "ingestion.py",
)

CONFIG_PATH = _resolve_source_path(
    "RAG_API_CONFIG_PATH",
    RAG_API_ROOT / "core" / "config.py",
    PROJECT_ROOT / "rag_api" / "core" / "config.py",
    Path.cwd() / "core" / "config.py",
    Path.cwd() / "rag_api" / "core" / "config.py",
)


def _module_tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Funzione top-level assente: {name}")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if name in targets:
                if isinstance(node.value, ast.Call):
                    # Supporta frozenset({...}) e set({...}) se usati come costanti.
                    if (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id in {"frozenset", "set"}
                        and len(node.value.args) == 1
                    ):
                        values = ast.literal_eval(node.value.args[0])
                        return (
                            frozenset(values)
                            if node.value.func.id == "frozenset"
                            else set(values)
                        )
                return ast.literal_eval(node.value)

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                if node.value is None:
                    raise AssertionError(f"Costante senza valore: {name}")
                if isinstance(node.value, ast.Call):
                    if (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id in {"frozenset", "set"}
                        and len(node.value.args) == 1
                    ):
                        values = ast.literal_eval(node.value.args[0])
                        return (
                            frozenset(values)
                            if node.value.func.id == "frozenset"
                            else set(values)
                        )
                return ast.literal_eval(node.value)

    raise AssertionError(f"Costante assente o non letterale: {name}")


def test_source_paths_are_resolved() -> None:
    assert INGESTION_PATH.is_file(), INGESTION_PATH
    assert CONFIG_PATH.is_file(), CONFIG_PATH
    assert INGESTION_PATH.parent == PROJECT_ROOT
    assert CONFIG_PATH.parent == RAG_API_ROOT / "core"


def test_kg_generation_functions_are_preserved() -> None:
    _, tree = _module_tree(INGESTION_PATH)

    required = {
        "llm_extract_kg",
        "_normalize_graph_schema",
        "_sanitize_graph",
        "enrich_formula_nodes_and_edges",
        "canonicalize_edges_to_verb_object",
        "canonicalize_edges_by_base_presence",
        "flush_neo4j_rows_batch",
        "validate_neo4j_graph_model",
    }

    present = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    missing = sorted(required - present)
    assert not missing, f"Funzioni KG mancanti: {missing}"


def test_entity_write_occurs_before_semantic_edge_guard() -> None:
    source, tree = _module_tree(INGESTION_PATH)
    flush = _top_level_function(tree, "flush_neo4j_rows_batch")
    body = ast.get_source_segment(source, flush) or ""

    entity_write = body.find("session.run(NEO4J_ENTITY_QUERY")
    edge_guard = body.find("resolve_allowed_neo4j_relationship_type")

    assert entity_write >= 0, "Scrittura Entity/MENTIONS non trovata"
    assert edge_guard >= 0, "Guardia whitelist relazioni non trovata"
    assert entity_write < edge_guard, (
        "La guardia delle relazioni deve essere applicata dopo la scrittura "
        "di Entity e MENTIONS, non prima."
    )


def test_semantic_whitelist_and_aliases() -> None:
    _, tree = _module_tree(INGESTION_PATH)

    allowed = _literal_assignment(tree, "NEO4J_ALLOWED_RELATIONSHIPS")
    aliases = _literal_assignment(tree, "NEO4J_RELATIONSHIP_ALIASES")

    assert "IMPLEMENTS" in allowed
    assert "COMPLIES_WITH" in allowed
    assert "HAS_FORMULA" in allowed
    assert aliases["IMPLEMENTES"] == "IMPLEMENTS"

    # Relazioni ambigue emerse dal test reale non devono essere approvate.
    for relation in {
        "UPDATES",
        "TRANSFORMS_INTO",
        "SELECTS",
        "INVALIDATES",
        "DOES_NOT",
        "TRANSFER_TO",
    }:
        assert relation not in allowed
        assert relation not in aliases


def test_ingestion_and_api_whitelists_are_aligned() -> None:
    _, ingestion_tree = _module_tree(INGESTION_PATH)
    _, config_tree = _module_tree(CONFIG_PATH)

    ingestion_allowed = set(
        _literal_assignment(
            ingestion_tree,
            "NEO4J_ALLOWED_RELATIONSHIPS",
        )
    )
    api_allowed = set(
        _literal_assignment(
            config_tree,
            "DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS",
        )
    )

    only_ingestion = sorted(ingestion_allowed - api_allowed)
    only_api = sorted(api_allowed - ingestion_allowed)
    assert ingestion_allowed == api_allowed, (
        "Whitelist ingestion/API non allineate. "
        f"Solo ingestion={only_ingestion}; solo API={only_api}"
    )


def test_resolver_behavior_without_importing_heavy_dependencies() -> None:
    _, tree = _module_tree(INGESTION_PATH)

    selected_nodes = []
    for name in (
        "RELTYPE_OK",
        "NEO4J_ALLOWED_RELATIONSHIPS",
        "NEO4J_RELATIONSHIP_ALIASES",
    ):
        for node in tree.body:
            target_names = []
            if isinstance(node, ast.Assign):
                target_names = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_names = [node.target.id]

            if name in target_names:
                selected_nodes.append(node)
                break

    selected_nodes.append(
        _top_level_function(tree, "normalize_neo4j_relationship_type")
    )
    selected_nodes.append(
        _top_level_function(tree, "resolve_allowed_neo4j_relationship_type")
    )

    isolated = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(isolated)

    namespace = {
        "re": __import__("re"),
        "Any": Any,
        "Optional": __import__("typing").Optional,
    }
    exec(compile(isolated, str(INGESTION_PATH), "exec"), namespace)

    resolve = namespace["resolve_allowed_neo4j_relationship_type"]

    assert resolve("implements") == "IMPLEMENTS"
    assert resolve("IMPLEMENTES") == "IMPLEMENTS"
    assert resolve("COMPLIES") == "COMPLIES_WITH"
    assert resolve("UPDATES") is None
    assert resolve("DOES_NOT_APPLY_TO") is None
    assert resolve("") is None
