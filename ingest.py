"""
ingest.py — PDF → chunks → embeddings → database.

Usage:
    # Ingest all PDFs in /app/data:
    docker compose exec app python ingest.py

    # Ingest a specific file:
    docker compose exec app python ingest.py myfile.pdf
"""

import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI

from db import get_conn, init_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("/app/data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))       # characters
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))  # characters
BATCH_SIZE = 100  # embeddings per API call (OpenAI limit is 2048)

openai_client = OpenAI()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF. Returns list of {page, text}."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split page texts into overlapping chunks.
    Each chunk carries its source page number.
    """
    chunks = []
    idx = 0

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_index": idx,
                    "page_number": page_num,
                    "text": chunk_text,
                })
                idx += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI. Handles batching internally."""
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([d.embedding for d in resp.data])

        if i + BATCH_SIZE < len(texts):
            time.sleep(0.2)  # gentle rate-limit pause

    return all_embeddings


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------
def store_chunks(file_id: int, chunks: list[dict], embeddings: list[list[float]]):
    """Bulk-insert chunks + embeddings into the database."""
    conn = get_conn()
    cur = conn.cursor()

    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append((
            file_id,
            chunk["chunk_index"],
            chunk["page_number"],
            chunk["text"],
            emb,
        ))

    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """
        INSERT INTO chunks (file_id, chunk_index, page_number, text, embedding)
        VALUES %s
        """,
        rows,
        template="(%s, %s, %s, %s, %s::vector)",
    )

    conn.commit()
    cur.close()
    conn.close()


def register_file(filename: str) -> int | None:
    """Register a file. Returns file_id, or None if already ingested."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM files WHERE filename = %s", (filename,))
    row = cur.fetchone()
    if row:
        cur.close()
        conn.close()
        return None  # already ingested

    cur.execute(
        "INSERT INTO files (filename) VALUES (%s) RETURNING id",
        (filename,),
    )
    file_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return file_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def ingest_file(pdf_path: Path):
    """Full pipeline for one PDF."""
    filename = pdf_path.name
    print(f"\n📄 Processing: {filename}")

    # 1. Register
    file_id = register_file(filename)
    if file_id is None:
        print(f"   ⏭  Already ingested, skipping.")
        return

    # 2. Extract
    pages = extract_pages(pdf_path)
    print(f"   Extracted {len(pages)} pages.")

    # 3. Chunk
    chunks = chunk_pages(pages)
    print(f"   Created {len(chunks)} chunks.")

    if not chunks:
        print("   ⚠  No text found, skipping.")
        return

    # 4. Embed
    texts = [c["text"] for c in chunks]
    print(f"   Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # 5. Store
    store_chunks(file_id, chunks, embeddings)
    print(f"   ✓ Stored {len(chunks)} chunks for file_id={file_id}.")


def main():
    init_db()

    # If a specific file is passed as argument, ingest just that
    if len(sys.argv) > 1:
        for fname in sys.argv[1:]:
            path = DATA_DIR / fname
            if path.exists():
                ingest_file(path)
            else:
                print(f"❌ File not found: {path}")
        return

    # Otherwise, ingest all PDFs in data/
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}. Drop some in there and re-run.")
        return

    print(f"Found {len(pdfs)} PDF(s) to process.")
    for pdf_path in pdfs:
        ingest_file(pdf_path)

    print("\n✓ Ingest complete.")


if __name__ == "__main__":
    main()
