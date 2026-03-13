"""
db.py — Database connection and schema setup.

Shared by both ingest.py and query.py.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.environ["DATABASE_URL"]

EMBEDDING_DIM = 1536  # text-embedding-3-small


def get_conn():
    """Return a new connection with pgvector registered."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    """Create tables + pgvector extension if they don't exist."""
    # Use a plain connection first (without register_vector)
    # because the extension might not exist yet
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Create the extension FIRST
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    # NOW register the vector type on this connection
    register_vector(conn)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id          SERIAL PRIMARY KEY,
            filename    TEXT UNIQUE NOT NULL,
            ingested_at TIMESTAMPTZ DEFAULT now()
        );
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          SERIAL PRIMARY KEY,
            file_id     INTEGER REFERENCES files(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            text        TEXT NOT NULL,
            embedding   vector({EMBEDDING_DIM}),
            metadata    JSONB DEFAULT '{{}}'::jsonb
        );
    """)

    # HNSW index for fast approximate nearest-neighbour search
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id                SERIAL PRIMARY KEY,
            query_text        TEXT NOT NULL,
            retrieved_chunk_ids INTEGER[],
            answer            TEXT,
            created_at        TIMESTAMPTZ DEFAULT now()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✓ Database initialised.")


if __name__ == "__main__":
    init_db()