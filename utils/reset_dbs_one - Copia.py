import psycopg2
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
import os

# ==========================================
# --- CONFIGURAZIONE STACK ASSESSMENT ---
# ==========================================

# Postgres (Porta 5433, DB assessment_ingestion)
PG_DSN = "dbname=assessment_ingestion user=admin password=admin_password host=localhost port=5433"

# Qdrant (Porta 6334, Collection assessment_docs)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6334
QDRANT_COLLECTION = "assessment_docs"

# Neo4j (Porta 7688, Auth admin_password)
NEO4J_URI = "bolt://localhost:7688"
NEO4J_AUTH = ("neo4j", "admin_password")

def clean_postgres_targeted(filename):
    print(f"🐘 Postgres: Rimozione record per {filename}...", end=" ")
    try:
        conn = psycopg2.connect(PG_DSN)
        cur = conn.cursor()
        
        # 1. Recuperiamo i log_id associati al file
        cur.execute("SELECT log_id FROM ingestion_logs WHERE source_name = %s", (filename,))
        log_ids = [row[0] for row in cur.fetchall()]
        
        if log_ids:
            # 2. Eliminiamo i dati collegati
            # Grazie al 'ON DELETE CASCADE' nel nuovo file PG_DDL min.sql 
            # basterebbe cancellare il log, ma forzare la cancellazione è più sicuro.
            cur.execute("DELETE FROM ingestion_images WHERE log_id = ANY(%s)", (log_ids,))
            cur.execute("DELETE FROM document_chunks WHERE log_id = ANY(%s)", (log_ids,))
            cur.execute("DELETE FROM ingestion_logs WHERE log_id = ANY(%s)", (log_ids,))
            conn.commit()
            print(f"✅ Rimossi {len(log_ids)} log e relativi chunk/immagini.")
        else:
            print("⚠️ Nessun dato trovato.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Errore: {e}")
        
def clean_neo4j_targeted(filename):
    print(f"🕸️  Neo4j: Rimozione sotto-grafo per {filename}...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            # Elimina Documento -> Pagine -> Chunk e le loro relazioni
            # Le Entity (nodi di concetti/controlli/rischi) rimangono nell'ontologia 
            # globale ma perdono il legame esclusivo con questo file.
            query = """
            MATCH (d:Document {filename: $fname})
            OPTIONAL MATCH (d)-[:HAS_PAGE]->(p)-[:HAS_CHUNK]->(c)
            DETACH DELETE d, p, c
            """
            result = session.run(query, fname=filename)
            summary = result.consume()
            print(f"✅ Eliminati {summary.counters.nodes_deleted} nodi.")
        driver.close()
    except Exception as e:
        print(f"❌ Errore: {e}")
        

def clean_qdrant_targeted(filename):
    print(f"💠 Qdrant: Rimozione vettori per {filename}...", end=" ")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Eliminiamo solo i punti dove il payload "filename" coincide
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    ),
                ]
            ),
        )
        print("✅ Punti eliminati.")
    except Exception as e:
        print(f"❌ Errore: {e}")
        

if __name__ == "__main__":
    target_file = input("Inserisci il nome del file da pulire (es. ISO_27001_Audit.pdf) o 'ALL' per interrompere: ")
    
    if target_file.upper() == 'ALL':
        print("\n⚠️ Per il reset TOTALE usa lo script 'reset_dbs.py' invece di questo.")
    else:
        clean_postgres_targeted(target_file)
        clean_neo4j_targeted(target_file)
        clean_qdrant_targeted(target_file)
        print(f"\n✨ Database puliti. Ora puoi ricaricare il file: {target_file}")
    