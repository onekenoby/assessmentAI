import os
import time
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
from typing import Optional, Tuple, List, Dict, Any
from config import *

# =========================
# CLIENTS INIT (LAZY)
# =========================
qdrant_client = None

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return qdrant_client

neo4j_driver = None
if NEO4J_ENABLED:
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    except Exception as e:
        print(f"⚠️ Neo4j disabled (driver init failed): {e}")
        NEO4J_ENABLED = False

pg_pool = ThreadedConnectionPool(
    PG_MIN_CONN, PG_MAX_CONN,
    host=PG_HOST, port=PG_PORT, dbname=PG_DB,
    user=PG_USER, password=PG_PASS
)

def pg_get_conn():
    return pg_pool.getconn()

def pg_put_conn(conn):
    pg_pool.putconn(conn)

def pg_start_log(source_name: str, source_type: str) -> int:
    conn = pg_get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingestion_logs (source_name, source_type, ingestion_ts, status) "
                "VALUES (%s, %s, NOW(), %s) RETURNING log_id",
                (source_name, source_type, "RUNNING")
            )
            log_id = cur.fetchone()[0]
        conn.commit()
        return log_id
    finally:
        pg_put_conn(conn)

def pg_close_log(log_id: int, status: str, total_chunks: int, processing_ms: int, error_msg: str = None):
    conn = pg_get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ingestion_logs SET status = %s, total_chunks = %s, processing_time_ms = %s, error_message = %s "
                "WHERE log_id = %s",
                (status, total_chunks, processing_ms, error_msg, log_id)
            )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        pg_put_conn(conn)

def pg_get_image_by_hash(image_hash: str, cur) -> Optional[Tuple[int, str]]:
    cur.execute(
        "SELECT image_id, description_ai FROM ingestion_images WHERE image_hash = %s LIMIT 1",
        (image_hash,)
    )
    return cur.fetchone()

def pg_save_image(log_id: int, image_bytes: bytes, mime_type: str, description: str, cur) -> int:
    import hashlib
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    cached = pg_get_image_by_hash(img_hash, cur)
    if cached:
        return cached[0]
    cur.execute(
        "INSERT INTO ingestion_images (log_id, image_data, image_hash, mime_type, description_ai, ingestion_ts) "
        "VALUES (%s, %s, %s, %s, %s, NOW()) RETURNING image_id",
        (log_id, psycopg2.Binary(image_bytes), img_hash, mime_type, description)
    )
    return cur.fetchone()[0]

def flush_postgres_chunks_batch(batch_data: List[Tuple]):
    if not batch_data: return
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    conn = pg_get_conn()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO document_chunks (
                    log_id, chunk_index, toon_type, content_raw, 
                    content_semantic, metadata_json, chunk_uuid, ingestion_ts
                )
                VALUES %s
                ON CONFLICT (chunk_uuid, ingestion_ts) DO UPDATE SET
                    content_semantic = EXCLUDED.content_semantic,
                    metadata_json = EXCLUDED.metadata_json
                """,
                [row + (now,) for row in batch_data]
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"   ⚠️ Postgres Batch Error: {e}")
    finally:
        pg_put_conn(conn)

def ensure_qdrant_collection():
    from ai.llm_engine import get_embedder
    embedder = get_embedder()
    dim = embedder.get_embedding_dimension()
    try:
        info = qdrant_client.get_collection(QDRANT_COLLECTION)
        if info.config.params.vectors.size != dim:
            print(f"⚠️ Vector dim mismatch! {info.config.params.vectors.size} vs {dim}")
    except Exception:
        print(f"🆕 Creating Qdrant collection '{QDRANT_COLLECTION}' (dim={dim})")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

NEO4J_BATCH_QUERY = """
UNWIND $rows AS r
MERGE (d:Document {doc_id: r.doc_id})
SET d.filename = r.filename, d.doc_type = r.doc_type, d.log_id = r.log_id, d.ingested_at = datetime()
WITH d, r
MERGE (p:Page {pid: r.doc_id + "::" + toString(r.page_no)})
SET p.doc_id = r.doc_id, p.page_no = r.page_no
MERGE (d)-[:HAS_PAGE]->(p)
WITH p, r
MERGE (c:Chunk {id: r.chunk_id})
SET c.chunk_index = r.chunk_index, c.toon_type = r.toon_type, c.page = r.page_no, c.filename = r.filename, c.text = left(r.text_sem, 1000), c.section_hint = coalesce(r.section_hint, ""), c.ontology = r.ontology
MERGE (p)-[:HAS_CHUNK]->(c)
WITH r, c
UNWIND coalesce(r.nodes, []) AS n
WITH n, c, r
WHERE n.id IS NOT NULL AND n.id <> ""
MERGE (e:Entity {id: n.id})
ON CREATE SET e.name = n.id, e.category = CASE WHEN n.category IS NOT NULL AND n.category <> 'UNCLASSIFIED' THEN n.category ELSE 'UNCLASSIFIED' END, e.sources = [r.filename_norm]
ON MATCH SET e.category = CASE WHEN n.category IS NOT NULL AND n.category <> 'UNCLASSIFIED' THEN n.category ELSE e.category END, e.sources = CASE WHEN r.filename_norm IN coalesce(e.sources, []) THEN coalesce(e.sources, []) ELSE coalesce(e.sources, []) + r.filename_norm END
SET e += coalesce(n.props, {})
MERGE (e)-[:PRESENT_IN]->(c)
"""

NEO4J_FORMULA_QUERY = """
UNWIND $rows AS r
MATCH (c:Chunk {id: r.chunk_id})
MERGE (f:Formula {fid: r.fid})
SET f.latex = r.latex, f.latex_raw = r.latex_raw, f.plain = r.plain, f.meaning_it = r.meaning_it, f.keywords = r.keywords, f.page = r.page_no, f.source = r.filename
MERGE (f)-[:MENTIONED_IN]->(c)
"""

def flush_neo4j_rows_batch(rows: List[Dict[str, Any]]):
    if not NEO4J_ENABLED or not rows: return
    try:
        with neo4j_driver.session() as session:
            session.run(NEO4J_BATCH_QUERY, rows=rows)
            edges_by_type = {}
            for r in rows:
                for edge in r.get("edges", []):
                    rel_type = edge.get("relation", "RELATES_TO").upper().replace(" ", "_")
                    rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_")
                    if not rel_type: continue
                    if rel_type not in edges_by_type: edges_by_type[rel_type] = []
                    edges_by_type[rel_type].append({"source": edge.get("source"), "target": edge.get("target"), "props": edge.get("props", {})})
            for rel_type, edges in edges_by_type.items():
                edge_query = f"UNWIND $batch AS e MATCH (s:Entity {{id: e.source}}) MATCH (t:Entity {{id: e.target}}) MERGE (s)-[r:{rel_type}]->(t) SET r += coalesce(e.props, {{}}), r.last_seen = datetime(), r.count = coalesce(r.count, 0) + 1"
                session.run(edge_query, batch=edges)
    except Exception as e:
        print(f"   ⚠️ Neo4j Batch Error: {e}")

def flush_neo4j_formulas_batch(rows: List[Dict[str, Any]]):
    if not NEO4J_ENABLED or not rows: return
    try:
        with neo4j_driver.session() as session:
            session.run(NEO4J_FORMULA_QUERY, rows=rows)
    except Exception as e:
        print(f"   ⚠️ Neo4j Formula Batch Error: {e}")