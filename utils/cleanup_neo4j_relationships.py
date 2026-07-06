"""Audit and optionally remove non-governed Neo4j semantic relationships.

Safe default: dry-run. No data is modified unless ``--apply`` is supplied.
Only relationships between ``:Entity`` nodes are inspected; structural graph
links such as HAS_PAGE, HAS_CHUNK and MENTIONS are outside this operation.

Run from the ``rag_api`` directory:

    python utils/cleanup_neo4j_relationships.py
    python utils/cleanup_neo4j_relationships.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Make ``core`` importable when the script is run directly from utils/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (  # noqa: E402
    DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS,
    DEFAULT_NEO4J_RELATIONSHIP_ALIASES,
)


@dataclass(frozen=True)
class Neo4jConnection:
    uri: str
    user: str
    password: str


def _connection_from_env() -> Neo4jConnection:
    return Neo4jConnection(
        uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7688").strip(),
        user=os.getenv("NEO4J_USER", "neo4j").strip(),
        password=os.getenv("NEO4J_PASS", "admin_password"),
    )


def _print_table(rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        print("Nessuna relazione fuori whitelist.")
        return

    width = max(len(str(row["relationship_type"])) for row in rows)
    width = max(width, len("RELATIONSHIP_TYPE"))
    print(f"{'RELATIONSHIP_TYPE':<{width}}  COUNT")
    print(f"{'-' * width}  -----")
    for row in rows:
        print(
            f"{str(row['relationship_type']):<{width}}  "
            f"{int(row['occurrences']):>5}"
        )


def _invalid_summary(session: Any, allowed: list[str], active_only: bool) -> list[dict[str, Any]]:
    status_clause = (
        "AND toLower(coalesce(r.status, 'active')) = 'active'"
        if active_only
        else ""
    )
    query = f"""
    MATCH (:Entity)-[r]->(:Entity)
    WHERE NOT type(r) IN $allowed
      {status_clause}
    RETURN type(r) AS relationship_type,
           count(r) AS occurrences
    ORDER BY occurrences DESC, relationship_type
    """
    return [dict(record) for record in session.run(query, allowed=allowed)]


def _count_invalid(session: Any, allowed: list[str], active_only: bool) -> int:
    status_clause = (
        "AND toLower(coalesce(r.status, 'active')) = 'active'"
        if active_only
        else ""
    )
    query = f"""
    MATCH (:Entity)-[r]->(:Entity)
    WHERE NOT type(r) IN $allowed
      {status_clause}
    RETURN count(r) AS invalid_count
    """
    record = session.run(query, allowed=allowed).single()
    return int(record["invalid_count"] or 0)




def _migrate_aliases(
    session: Any,
    aliases: dict[str, str],
    active_only: bool,
) -> int:
    migrated_total = 0
    status_clause = (
        "AND toLower(coalesce(r.status, 'active')) = 'active'"
        if active_only
        else ""
    )

    for raw_type, canonical_type in sorted(aliases.items()):
        if raw_type == canonical_type:
            continue
        if raw_type not in {row["relationship_type"] for row in _invalid_summary(session, list(DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS), active_only)}:
            continue

        # Both identifiers come from static, validated source code constants.
        query = f"""
        MATCH (s:Entity)-[r:{raw_type}]->(t:Entity)
        WHERE 1 = 1
          {status_clause}
        MERGE (s)-[nr:{canonical_type}]->(t)
        SET nr += properties(r)
        DELETE r
        RETURN count(*) AS migrated_count
        """
        record = session.run(query).single()
        migrated = int(record["migrated_count"] or 0)
        if migrated:
            print(f"Migrati {migrated} archi {raw_type} -> {canonical_type}")
            migrated_total += migrated

    return migrated_total

def _delete_batch(
    session: Any,
    allowed: list[str],
    active_only: bool,
    batch_size: int,
) -> int:
    status_clause = (
        "AND toLower(coalesce(r.status, 'active')) = 'active'"
        if active_only
        else ""
    )
    query = f"""
    MATCH (:Entity)-[r]->(:Entity)
    WHERE NOT type(r) IN $allowed
      {status_clause}
    WITH r
    LIMIT $batch_size
    DELETE r
    RETURN count(*) AS deleted_count
    """
    record = session.run(
        query,
        allowed=allowed,
        batch_size=batch_size,
    ).single()
    return int(record["deleted_count"] or 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Controlla e, solo con --apply, elimina le relazioni semantiche "
            "Entity->Entity fuori whitelist."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="applica realmente la cancellazione; senza questa opzione è dry-run",
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="include anche relazioni non active; il default controlla solo active",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="numero massimo di relazioni eliminate per transazione (default: 500)",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 10_000:
        parser.error("--batch-size deve essere compreso tra 1 e 10000")

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        print("ERRORE: pacchetto neo4j non installato.", file=sys.stderr)
        print("Eseguire: python -m pip install neo4j", file=sys.stderr)
        return 2

    connection = _connection_from_env()
    allowed = sorted(set(DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS))
    active_only = not args.all_statuses

    print("Neo4j relationship governance cleanup")
    print("=" * 72)
    print(f"URI: {connection.uri}")
    print(f"Utente: {connection.user}")
    print(f"Perimetro: {'solo active' if active_only else 'tutti gli status'}")
    print(f"Whitelist: {len(allowed)} relationship types")
    print(f"Modalità: {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    driver = GraphDatabase.driver(
        connection.uri,
        auth=(connection.user, connection.password),
    )

    try:
        driver.verify_connectivity()
        with driver.session() as session:
            before = _invalid_summary(session, allowed, active_only)
            invalid_before = sum(int(row["occurrences"]) for row in before)

            print("Relazioni fuori whitelist rilevate:")
            _print_table(before)
            print(f"\nTotale: {invalid_before}")

            if not args.apply:
                print("\nDry-run concluso: nessuna relazione è stata modificata.")
                print("Per applicare: python utils/cleanup_neo4j_relationships.py --apply")
                return 0

            if invalid_before == 0:
                print("\nNessuna cancellazione necessaria.")
                return 0

            migrated_total = _migrate_aliases(
                session,
                dict(DEFAULT_NEO4J_RELATIONSHIP_ALIASES),
                active_only,
            )

            deleted_total = 0
            while True:
                deleted = _delete_batch(
                    session,
                    allowed,
                    active_only,
                    args.batch_size,
                )
                if deleted == 0:
                    break
                deleted_total += deleted
                print(f"Eliminate {deleted_total}/{invalid_before} relazioni...")

            invalid_after = _count_invalid(session, allowed, active_only)
            print(f"\nRelazioni migrate tramite alias: {migrated_total}")
            print(f"Relazioni eliminate: {deleted_total}")
            print(f"Relazioni fuori whitelist residue: {invalid_after}")

            if invalid_after != 0:
                print("ERRORE: la bonifica non è completa.", file=sys.stderr)
                return 3

            print("Bonifica completata correttamente.")
            return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
