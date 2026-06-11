import psycopg2
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

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

def clean_postgres():
    print("🐘 Cleaning Postgres...", end=" ")
    try:
        conn = psycopg2.connect(PG_DSN)
        cur = conn.cursor()
        
        # Aggiornato in base al nuovo PG_DDL min.sql
        cur.execute("""
            TRUNCATE TABLE 
                ingestion_images,
                document_chunks,
                ingestion_logs
            RESTART IDENTITY CASCADE;
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Fatto.")
    except Exception as e:
        print(f"❌ Errore: {e}")

def clean_neo4j():
    print("🕸️  Cleaning Neo4j...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        print("✅ Fatto.")
    except Exception as e:
        print(f"❌ Errore: {e}")

def clean_qdrant():
    print("💠 Cleaning Qdrant...", end=" ")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if client.collection_exists(QDRANT_COLLECTION):
            client.delete_collection(QDRANT_COLLECTION)
            print("✅ Collection eliminata.", end=" ")
        else:
            print("⚠️ Collection non trovata (già vuota).", end=" ")
        print("")
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    print("--- 🗑️  RESET TOTALE DATABASE (ASSESSMENT) 🗑️  ---")
    confirm = input("Sei sicuro di voler cancellare TUTTI i dati? (s/N): ")
    if confirm.lower() == 's':
        clean_postgres()
        clean_neo4j()
        clean_qdrant()
        print("\n✨ Ambiente Assessment pulito e pronto per una nuova ingestion.")
    else:
        print("Operazione annullata.")