-- --- PULIZIA TOTALE ---
DROP TABLE IF EXISTS ingestion_images CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS ingestion_logs CASCADE;

-- 1. ABILITAZIONE ESTENSIONE
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 2. TABELLA LOGS (Standard Postgres Table)
CREATE TABLE ingestion_logs (
    log_id              BIGSERIAL PRIMARY KEY, 
    source_name         TEXT NOT NULL,
    source_type         TEXT,
    ingestion_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status              TEXT,
    total_chunks        INTEGER DEFAULT 0,
    processing_time_ms  INTEGER,
    error_message       TEXT
);

-- 3. TABELLA CHUNKS (Hypertable)
CREATE TABLE document_chunks (
    chunk_id            BIGSERIAL,
    log_id              BIGINT NOT NULL,
    chunk_index         INTEGER NOT NULL,
    ingestion_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    toon_type           TEXT,
    content_raw         TEXT,
    content_semantic    TEXT,
    metadata_json       JSONB,
    chunk_uuid          VARCHAR(32), -- AGGIUNTA: Richiesta dallo script Python
    
    -- MODIFICATA: Affinché l'ON CONFLICT di Python funzioni, 
    -- chunk_uuid e ingestion_ts devono formare la Primary Key
    PRIMARY KEY (chunk_uuid, ingestion_ts),
    
    CONSTRAINT fk_logs 
        FOREIGN KEY (log_id) 
        REFERENCES ingestion_logs(log_id)
        ON DELETE CASCADE
);

-- CORRETTA: Rimosso migrate_data => true
SELECT create_hypertable('document_chunks', 'ingestion_ts');


-- 4. TABELLA IMMAGINI (Hypertable)
CREATE TABLE ingestion_images (
    image_id            BIGSERIAL,
    log_id              BIGINT NOT NULL,
    ingestion_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    image_data          BYTEA,
    image_hash          CHAR(64),
    mime_type           TEXT DEFAULT 'image/png',
    description_ai      TEXT,
    
    PRIMARY KEY (image_id, ingestion_ts),
    
    CONSTRAINT fk_logs_img 
        FOREIGN KEY (log_id) 
        REFERENCES ingestion_logs(log_id)
        ON DELETE CASCADE
);

-- CORRETTA: Rimosso migrate_data => true
SELECT create_hypertable('ingestion_images', 'ingestion_ts');

-- Indici extra
CREATE INDEX idx_chunks_log ON document_chunks(log_id, chunk_index);
CREATE INDEX idx_images_hash ON ingestion_images(image_hash);
