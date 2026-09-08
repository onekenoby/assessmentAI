import os
import re
import sys
import uuid

import psycopg2
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase


# ============================================================
# CONFIGURAZIONE STACK ASSESSMENT
# ============================================================

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5433"))
PG_DB = os.getenv("PG_DB", "assessment_ingestion")
PG_USER = os.getenv("PG_USER", "admin")
PG_PASS = os.getenv("PG_PASS", "admin_password")

PG_DSN = (
    f"dbname={PG_DB} "
    f"user={PG_USER} "
    f"password={PG_PASS} "
    f"host={PG_HOST} "
    f"port={PG_PORT}"
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6334"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "assessment_docs")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "admin_password")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASS)


# ============================================================
# INPUT
# ============================================================

def parse_ids(raw_value):
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        return []

    tokens = [
        token.strip()
        for token in re.split(r"[\s,;]+", raw_value)
        if token.strip()
    ]

    result = []
    seen = set()
    invalid = []

    for token in tokens:
        try:
            value = str(uuid.UUID(token))
        except (ValueError, TypeError, AttributeError):
            invalid.append(token)
            continue

        if value not in seen:
            seen.add(value)
            result.append(value)

    if invalid:
        raise ValueError("UUID non validi: " + ", ".join(invalid))

    return result


# ============================================================
# PREFLIGHT
# ============================================================

def preflight_connections():
    print("\n🔎 PREFLIGHT CONNESSIONI")

    conn = psycopg2.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    inet_server_addr()::text,
                    inet_server_port()
                """
            )
            db_name, db_user, server_addr, server_port = cur.fetchone()

        print(
            "   ✅ PostgreSQL | "
            f"db={db_name} | user={db_user} | "
            f"server={server_addr}:{server_port} | "
            f"configured_host={PG_HOST}:{PG_PORT}"
        )
    finally:
        conn.close()

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    qdrant.get_collection(QDRANT_COLLECTION)
    print(
        "   ✅ Qdrant | "
        f"host={QDRANT_HOST}:{QDRANT_PORT} | "
        f"collection={QDRANT_COLLECTION}"
    )

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        with driver.session() as session:
            session.run("RETURN 1 AS ok").single()
        print(f"   ✅ Neo4j | uri={NEO4J_URI}")
    finally:
        driver.close()


# ============================================================
# QDRANT
# ============================================================

def qdrant_filter(target_ids):
    """
    L'UUID può essere:
      - payload.doc_id
      - payload.document_id
      - payload.ingestion_run_id

    La ricerca è deterministica: OR sui tre campi.
    """
    conditions = []

    for target_id in target_ids:
        for key in ("doc_id", "document_id", "ingestion_run_id"):
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=target_id),
                )
            )

    return models.Filter(should=conditions)


def clean_qdrant(target_ids):
    print("\n💠 QDRANT")

    try:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )

        flt = qdrant_filter(target_ids)

        before = client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=flt,
            exact=True,
        ).count

        print(
            f"   BEFORE | match doc_id/document_id/ingestion_run_id={before}"
        )

        if before:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=flt,
                wait=True,
            )

        after = client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=flt,
            exact=True,
        ).count

        print(f"   AFTER  | residui={after}")

        if int(after or 0) != 0:
            print("   ❌ Qdrant POST-CHECK fallito.")
            return False

        print(
            f"   ✅ Qdrant pulito | "
            f"points_deleted={int(before or 0)} | residui=0"
        )
        return True

    except Exception as exc:
        print(f"   ❌ Qdrant errore: {exc}")
        return False


# ============================================================
# POSTGRESQL
# ============================================================

def pg_discover_targets(cur, target_ids):
    """
    Distingue:
      - UUID usati come document_id/doc_id;
      - UUID usati come ingestion_run_id.

    Per un document_id vengono poi ricavate tutte le ingestion_run_id
    presenti nei chunk di quel documento.
    """

    # ID che sono realmente document_id/doc_id nella KB.
    cur.execute(
        """
        SELECT DISTINCT
            COALESCE(
                NULLIF(metadata_json ->> 'document_id', ''),
                NULLIF(metadata_json ->> 'doc_id', '')
            ) AS document_id
        FROM public.document_chunks
        WHERE metadata_json ->> 'document_id' = ANY(%s)
           OR metadata_json ->> 'doc_id' = ANY(%s)
        """,
        (target_ids, target_ids),
    )
    document_ids = {
        str(row[0])
        for row in cur.fetchall()
        if row[0]
    }

    # ID che sono realmente ingestion_run_id in una delle tabelle KB.
    cur.execute(
        """
        SELECT DISTINCT ingestion_run_id::text
        FROM (
            SELECT ingestion_run_id
            FROM public.document_chunks
            WHERE ingestion_run_id::text = ANY(%s)

            UNION

            SELECT ingestion_run_id
            FROM public.ingestion_images
            WHERE ingestion_run_id::text = ANY(%s)

            UNION

            SELECT ingestion_run_id
            FROM public.ingestion_logs
            WHERE ingestion_run_id::text = ANY(%s)
        ) x
        WHERE ingestion_run_id IS NOT NULL
        """,
        (target_ids, target_ids, target_ids),
    )
    direct_run_ids = {
        str(row[0])
        for row in cur.fetchall()
        if row[0]
    }

    # Se l'input è un document_id, ricava tutte le run del documento.
    derived_run_ids = set()

    if document_ids:
        doc_list = sorted(document_ids)
        cur.execute(
            """
            SELECT DISTINCT ingestion_run_id::text
            FROM public.document_chunks
            WHERE (
                    metadata_json ->> 'document_id' = ANY(%s)
                 OR metadata_json ->> 'doc_id' = ANY(%s)
            )
              AND ingestion_run_id IS NOT NULL
            """,
            (doc_list, doc_list),
        )
        derived_run_ids = {
            str(row[0])
            for row in cur.fetchall()
            if row[0]
        }

    run_ids = direct_run_ids | derived_run_ids

    # Log direttamente collegati ai documenti/run da rimuovere.
    log_ids = set()

    clauses = []
    params = []

    if document_ids:
        doc_list = sorted(document_ids)
        clauses.append(
            "(metadata_json ->> 'document_id' = ANY(%s) "
            "OR metadata_json ->> 'doc_id' = ANY(%s))"
        )
        params.extend([doc_list, doc_list])

    if run_ids:
        clauses.append("ingestion_run_id::text = ANY(%s)")
        params.append(sorted(run_ids))

    if clauses:
        cur.execute(
            f"""
            SELECT DISTINCT log_id
            FROM public.document_chunks
            WHERE {" OR ".join(clauses)}
            """,
            tuple(params),
        )
        log_ids.update(
            int(row[0])
            for row in cur.fetchall()
            if row[0] is not None
        )

    if run_ids:
        cur.execute(
            """
            SELECT DISTINCT log_id
            FROM public.ingestion_logs
            WHERE ingestion_run_id::text = ANY(%s)
            """,
            (sorted(run_ids),),
        )
        log_ids.update(
            int(row[0])
            for row in cur.fetchall()
            if row[0] is not None
        )

    return document_ids, run_ids, log_ids


def pg_counts(cur, document_ids, run_ids, log_ids):
    clauses = []
    params = []

    if document_ids:
        doc_list = sorted(document_ids)
        clauses.append(
            "(metadata_json ->> 'document_id' = ANY(%s) "
            "OR metadata_json ->> 'doc_id' = ANY(%s))"
        )
        params.extend([doc_list, doc_list])

    if run_ids:
        clauses.append("ingestion_run_id::text = ANY(%s)")
        params.append(sorted(run_ids))

    chunk_count = 0

    if clauses:
        cur.execute(
            f"""
            SELECT count(*)
            FROM public.document_chunks
            WHERE {" OR ".join(clauses)}
            """,
            tuple(params),
        )
        chunk_count = int(cur.fetchone()[0])

    image_count = 0
    image_clauses = []
    image_params = []

    if run_ids:
        image_clauses.append("ingestion_run_id::text = ANY(%s)")
        image_params.append(sorted(run_ids))

    if log_ids:
        image_clauses.append("log_id = ANY(%s)")
        image_params.append(sorted(log_ids))

    if image_clauses:
        cur.execute(
            f"""
            SELECT count(*)
            FROM public.ingestion_images
            WHERE {" OR ".join(image_clauses)}
            """,
            tuple(image_params),
        )
        image_count = int(cur.fetchone()[0])

    log_count = 0
    log_clauses = []
    log_params = []

    if run_ids:
        log_clauses.append("ingestion_run_id::text = ANY(%s)")
        log_params.append(sorted(run_ids))

    if log_ids:
        log_clauses.append("log_id = ANY(%s)")
        log_params.append(sorted(log_ids))

    if log_clauses:
        cur.execute(
            f"""
            SELECT count(*)
            FROM public.ingestion_logs
            WHERE {" OR ".join(log_clauses)}
            """,
            tuple(log_params),
        )
        log_count = int(cur.fetchone()[0])

    return chunk_count, image_count, log_count


def clean_postgres(target_ids):
    print("\n🐘 POSTGRESQL")

    conn = psycopg2.connect(PG_DSN)

    try:
        with conn.cursor() as cur:
            document_ids, run_ids, log_ids = pg_discover_targets(
                cur,
                target_ids,
            )

            print(
                "   MATCH | "
                f"document_ids={sorted(document_ids)} | "
                f"ingestion_run_ids={sorted(run_ids)} | "
                f"log_ids={sorted(log_ids)}"
            )

            before = pg_counts(
                cur,
                document_ids,
                run_ids,
                log_ids,
            )

            print(
                "   BEFORE | "
                f"chunks={before[0]} | "
                f"images={before[1]} | "
                f"logs={before[2]}"
            )

            if not document_ids and not run_ids and not log_ids:
                print(
                    "   ℹ️ Nessuna traccia PostgreSQL KB trovata "
                    "per gli UUID richiesti."
                )
                return True

            # 1. Immagini
            image_clauses = []
            image_params = []

            if run_ids:
                image_clauses.append("ingestion_run_id::text = ANY(%s)")
                image_params.append(sorted(run_ids))

            if log_ids:
                image_clauses.append("log_id = ANY(%s)")
                image_params.append(sorted(log_ids))

            images_deleted = 0
            if image_clauses:
                cur.execute(
                    f"""
                    DELETE FROM public.ingestion_images
                    WHERE {" OR ".join(image_clauses)}
                    """,
                    tuple(image_params),
                )
                images_deleted = cur.rowcount

            # 2. Chunks
            chunk_clauses = []
            chunk_params = []

            if document_ids:
                doc_list = sorted(document_ids)
                chunk_clauses.append(
                    "(metadata_json ->> 'document_id' = ANY(%s) "
                    "OR metadata_json ->> 'doc_id' = ANY(%s))"
                )
                chunk_params.extend([doc_list, doc_list])

            if run_ids:
                chunk_clauses.append("ingestion_run_id::text = ANY(%s)")
                chunk_params.append(sorted(run_ids))

            chunks_deleted = 0
            if chunk_clauses:
                cur.execute(
                    f"""
                    DELETE FROM public.document_chunks
                    WHERE {" OR ".join(chunk_clauses)}
                    """,
                    tuple(chunk_params),
                )
                chunks_deleted = cur.rowcount

            # 3. Logs
            log_clauses = []
            log_params = []

            if run_ids:
                log_clauses.append("ingestion_run_id::text = ANY(%s)")
                log_params.append(sorted(run_ids))

            if log_ids:
                log_clauses.append("log_id = ANY(%s)")
                log_params.append(sorted(log_ids))

            logs_deleted = 0
            if log_clauses:
                cur.execute(
                    f"""
                    DELETE FROM public.ingestion_logs
                    WHERE {" OR ".join(log_clauses)}
                    """,
                    tuple(log_params),
                )
                logs_deleted = cur.rowcount

            after = pg_counts(
                cur,
                document_ids,
                run_ids,
                log_ids,
            )

            if any(after):
                conn.rollback()
                print(
                    "   ❌ PostgreSQL POST-CHECK fallito | "
                    f"chunks={after[0]} | images={after[1]} | logs={after[2]}"
                )
                print("      ROLLBACK eseguito.")
                return False

        conn.commit()

        print(
            "   ✅ PostgreSQL pulito | "
            f"chunks_deleted={chunks_deleted} | "
            f"images_deleted={images_deleted} | "
            f"logs_deleted={logs_deleted} | "
            "residui=0"
        )
        return True

    except Exception as exc:
        conn.rollback()
        print(f"   ❌ PostgreSQL errore: {exc}")
        return False

    finally:
        conn.close()


# ============================================================
# NEO4J
# ============================================================

def clean_neo4j(target_ids):
    print("\n🕸️  NEO4J")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    try:
        with driver.session() as session:

            # --------------------------------------------------------
            # 1. Scopri se gli UUID sono doc_id e/o ingestion_run_id.
            # --------------------------------------------------------

            direct_doc_ids = {
                str(record["doc_id"])
                for record in session.run(
                    """
                    MATCH (n)
                    WHERE (n:Document OR n:Page OR n:Chunk)
                      AND n.doc_id IN $ids
                    RETURN DISTINCT n.doc_id AS doc_id
                    """,
                    ids=target_ids,
                )
                if record["doc_id"] is not None
            }

            direct_run_ids = set()

            # run_id sui nodi strutturali/formula
            for record in session.run(
                """
                MATCH (n)
                WHERE (n:Document OR n:Page OR n:Chunk OR n:Formula)
                  AND toString(n.ingestion_run_id) IN $ids
                RETURN DISTINCT toString(n.ingestion_run_id) AS run_id
                """,
                ids=target_ids,
            ):
                if record["run_id"]:
                    direct_run_ids.add(str(record["run_id"]))

            # run_id nelle Entity
            for record in session.run(
                """
                MATCH (e:Entity)
                UNWIND coalesce(e.ingestion_run_ids, []) AS rid
                WITH DISTINCT toString(rid) AS run_id
                WHERE run_id IN $ids
                RETURN run_id
                """,
                ids=target_ids,
            ):
                if record["run_id"]:
                    direct_run_ids.add(str(record["run_id"]))

            # run_id sulle relazioni
            for record in session.run(
                """
                MATCH ()-[r]->()
                WHERE toString(r.ingestion_run_id) IN $ids
                RETURN DISTINCT toString(r.ingestion_run_id) AS run_id
                """,
                ids=target_ids,
            ):
                if record["run_id"]:
                    direct_run_ids.add(str(record["run_id"]))

            # Se l'UUID è un doc_id, raccogli tutte le run note di quel doc.
            derived_run_ids = set()

            if direct_doc_ids:
                for record in session.run(
                    """
                    MATCH (n)
                    WHERE (n:Document OR n:Page OR n:Chunk OR n:Formula)
                      AND n.doc_id IN $doc_ids
                      AND n.ingestion_run_id IS NOT NULL
                    RETURN DISTINCT toString(n.ingestion_run_id) AS run_id
                    """,
                    doc_ids=sorted(direct_doc_ids),
                ):
                    if record["run_id"]:
                        derived_run_ids.add(str(record["run_id"]))

            run_ids = direct_run_ids | derived_run_ids

            print(
                "   MATCH | "
                f"doc_ids={sorted(direct_doc_ids)} | "
                f"ingestion_run_ids={sorted(run_ids)}"
            )

            # --------------------------------------------------------
            # 2. Identifica i Chunk che verranno cancellati.
            # --------------------------------------------------------

            selected_chunk_ids = set()

            for record in session.run(
                """
                MATCH (c:Chunk)
                WHERE
                    c.doc_id IN $doc_ids
                    OR toString(c.ingestion_run_id) IN $run_ids
                RETURN DISTINCT toString(c.id) AS chunk_id
                """,
                doc_ids=sorted(direct_doc_ids),
                run_ids=sorted(run_ids),
            ):
                if record["chunk_id"]:
                    selected_chunk_ids.add(str(record["chunk_id"]))

            # Entità toccate dai chunk da rimuovere.
            affected_entity_ids = set()

            if selected_chunk_ids:
                for record in session.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE toString(c.id) IN $chunk_ids
                    RETURN DISTINCT e.id AS entity_id
                    """,
                    chunk_ids=sorted(selected_chunk_ids),
                ):
                    if record["entity_id"]:
                        affected_entity_ids.add(str(record["entity_id"]))

            # --------------------------------------------------------
            # 3. BEFORE
            # --------------------------------------------------------

            structural_before = session.run(
                """
                MATCH (n)
                WHERE (n:Document OR n:Page OR n:Chunk)
                  AND (
                        n.doc_id IN $doc_ids
                        OR toString(n.ingestion_run_id) IN $run_ids
                  )
                RETURN count(n) AS count
                """,
                doc_ids=sorted(direct_doc_ids),
                run_ids=sorted(run_ids),
            ).single()["count"]

            formula_before = session.run(
                """
                MATCH (f:Formula)
                WHERE toString(f.ingestion_run_id) IN $run_ids
                RETURN count(f) AS count
                """,
                run_ids=sorted(run_ids),
            ).single()["count"]

            print(
                "   BEFORE | "
                f"structural_nodes={structural_before} | "
                f"formula_nodes={formula_before} | "
                f"chunk_ids={len(selected_chunk_ids)} | "
                f"affected_entities={len(affected_entity_ids)}"
            )

            if (
                not direct_doc_ids
                and not run_ids
                and not selected_chunk_ids
            ):
                print(
                    "   ℹ️ Nessuna traccia Neo4j trovata "
                    "per gli UUID richiesti."
                )
                return True

            # --------------------------------------------------------
            # 4. Pulisci provenance delle relazioni Entity -> Entity.
            # --------------------------------------------------------

            if selected_chunk_ids:
                session.run(
                    """
                    MATCH (:Entity)-[r]->(:Entity)
                    WHERE any(
                        cid IN coalesce(r.chunk_ids, [])
                        WHERE toString(cid) IN $chunk_ids
                    )
                    SET r.chunk_ids = [
                        cid IN coalesce(r.chunk_ids, [])
                        WHERE NOT toString(cid) IN $chunk_ids
                    ]
                    """,
                    chunk_ids=sorted(selected_chunk_ids),
                ).consume()

                session.run(
                    """
                    MATCH (:Entity)-[r]->(:Entity)
                    WHERE r.chunk_ids IS NOT NULL
                      AND size(r.chunk_ids) = 0
                    DELETE r
                    """
                ).consume()

            # --------------------------------------------------------
            # 5. Pulisci ingestion_run_ids dalle Entity.
            # --------------------------------------------------------

            if run_ids:
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE any(
                        rid IN coalesce(e.ingestion_run_ids, [])
                        WHERE toString(rid) IN $run_ids
                    )
                    SET e.ingestion_run_ids = [
                        rid IN coalesce(e.ingestion_run_ids, [])
                        WHERE NOT toString(rid) IN $run_ids
                    ]
                    """,
                    run_ids=sorted(run_ids),
                ).consume()

                session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE toString(r.ingestion_run_id) IN $run_ids
                    REMOVE r.ingestion_run_id
                    """,
                    run_ids=sorted(run_ids),
                ).consume()

            # --------------------------------------------------------
            # 6. Cancella Formula e nodi strutturali.
            # --------------------------------------------------------

            if run_ids:
                session.run(
                    """
                    MATCH (f:Formula)
                    WHERE toString(f.ingestion_run_id) IN $run_ids
                    DETACH DELETE f
                    """,
                    run_ids=sorted(run_ids),
                ).consume()

            summary = session.run(
                """
                MATCH (n)
                WHERE (n:Document OR n:Page OR n:Chunk)
                  AND (
                        n.doc_id IN $doc_ids
                        OR toString(n.ingestion_run_id) IN $run_ids
                  )
                DETACH DELETE n
                """,
                doc_ids=sorted(direct_doc_ids),
                run_ids=sorted(run_ids),
            ).consume()

            # Formula eventualmente rimaste orfane.
            orphan_formula_deleted = session.run(
                """
                MATCH (f:Formula)
                WHERE NOT (f)<-[:HAS_FORMULA]-(:Chunk)
                DELETE f
                RETURN count(f) AS deleted
                """
            ).single()["deleted"]

            # Entità appartenenti esclusivamente ai chunk eliminati:
            # elimina solo quelle rimaste totalmente isolate.
            isolated_entities_deleted = 0

            if affected_entity_ids:
                isolated_entities_deleted = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.id IN $entity_ids
                      AND NOT (e)<-[:MENTIONS]-(:Chunk)
                      AND NOT (e)--(:Entity)
                    DELETE e
                    RETURN count(e) AS deleted
                    """,
                    entity_ids=sorted(affected_entity_ids),
                ).single()["deleted"]

            # --------------------------------------------------------
            # 7. POST-CHECK MIRATO.
            #    Nessuna scansione generica delle proprietà array:
            #    evita l'errore toString(StringArray) visto nel log.
            # --------------------------------------------------------

            structural_after = session.run(
                """
                MATCH (n)
                WHERE (n:Document OR n:Page OR n:Chunk)
                  AND (
                        n.doc_id IN $doc_ids
                        OR toString(n.ingestion_run_id) IN $run_ids
                  )
                RETURN count(n) AS count
                """,
                doc_ids=sorted(direct_doc_ids),
                run_ids=sorted(run_ids),
            ).single()["count"]

            formula_after = session.run(
                """
                MATCH (f:Formula)
                WHERE toString(f.ingestion_run_id) IN $run_ids
                RETURN count(f) AS count
                """,
                run_ids=sorted(run_ids),
            ).single()["count"]

            entity_run_after = session.run(
                """
                MATCH (e:Entity)
                WHERE any(
                    rid IN coalesce(e.ingestion_run_ids, [])
                    WHERE toString(rid) IN $run_ids
                )
                RETURN count(e) AS count
                """,
                run_ids=sorted(run_ids),
            ).single()["count"]

            relationship_run_after = session.run(
                """
                MATCH ()-[r]->()
                WHERE toString(r.ingestion_run_id) IN $run_ids
                RETURN count(r) AS count
                """,
                run_ids=sorted(run_ids),
            ).single()["count"]

            relationship_chunk_after = 0

            if selected_chunk_ids:
                relationship_chunk_after = session.run(
                    """
                    MATCH (:Entity)-[r]->(:Entity)
                    WHERE any(
                        cid IN coalesce(r.chunk_ids, [])
                        WHERE toString(cid) IN $chunk_ids
                    )
                    RETURN count(r) AS count
                    """,
                    chunk_ids=sorted(selected_chunk_ids),
                ).single()["count"]

            if any(
                [
                    structural_after,
                    formula_after,
                    entity_run_after,
                    relationship_run_after,
                    relationship_chunk_after,
                ]
            ):
                print("   ❌ Neo4j POST-CHECK fallito:")
                print(f"      structural={structural_after}")
                print(f"      formula={formula_after}")
                print(f"      entity_run={entity_run_after}")
                print(f"      relationship_run={relationship_run_after}")
                print(f"      relationship_chunk={relationship_chunk_after}")
                return False

            print(
                "   ✅ Neo4j pulito | "
                f"nodes_deleted={summary.counters.nodes_deleted} | "
                f"relationships_deleted="
                f"{summary.counters.relationships_deleted} | "
                f"orphan_formulas_deleted={orphan_formula_deleted} | "
                f"isolated_entities_deleted={isolated_entities_deleted} | "
                "residui=0"
            )
            return True

    except Exception as exc:
        print(f"   ❌ Neo4j errore: {exc}")
        return False

    finally:
        driver.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("   RESET MIRATO KB PER UUID")
    print("   UUID accettati: document_id/doc_id oppure ingestion_run_id")
    print("   Qdrant + PostgreSQL KB + Neo4j")
    print("=" * 76)

    raw_ids = input(
        "\nInserisci uno o più UUID da eliminare "
        "(separati da virgola, ';' o spazio)\n"
        "oppure 'ALL' per interrompere:\n> "
    ).strip()

    if raw_ids.upper() == "ALL":
        print(
            "\n⚠️ Operazione interrotta. "
            "Per il reset TOTALE usa reset_dbs.py."
        )
        return 0

    try:
        target_ids = parse_ids(raw_ids)
    except ValueError as exc:
        print(f"\n❌ {exc}")
        return 2

    if not target_ids:
        print("\n⚠️ Nessun UUID specificato.")
        return 0

    print("\n🎯 UUID selezionati:")
    for target_id in target_ids:
        print(f"   - {target_id}")

    try:
        preflight_connections()
    except Exception as exc:
        print(
            "\n❌ Preflight fallito. "
            "Nessuna cancellazione è stata avviata."
        )
        print(f"   Dettaglio: {exc}")
        return 1

    print("\n🧹 Avvio pulizia mirata della Knowledge Base...")

    qdrant_ok = clean_qdrant(target_ids)
    neo4j_ok = clean_neo4j(target_ids)
    postgres_ok = clean_postgres(target_ids)

    print("\n" + "=" * 76)

    if qdrant_ok and neo4j_ok and postgres_ok:
        print("✨ PULIZIA KB VERIFICATA: residui=0 sui tre backend.")
        print("   - Qdrant")
        print("   - PostgreSQL KB (assessment_ingestion/public)")
        print("   - Neo4j")
        print(
            "   Nota: assessment_gestio_tier.rag_ingestion.rag_document "
            "NON viene modificato."
        )
        print("=" * 76)
        return 0

    print("❌ PULIZIA NON VERIFICATA.")
    print(f"   Qdrant:     {'OK' if qdrant_ok else 'ERRORE'}")
    print(f"   Neo4j:      {'OK' if neo4j_ok else 'ERRORE'}")
    print(f"   PostgreSQL: {'OK' if postgres_ok else 'ERRORE'}")
    print("=" * 76)
    return 1


if __name__ == "__main__":
    sys.exit(main())
