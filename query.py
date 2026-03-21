"""
query.py — FastAPI server.

Endpoints:
    POST /query          — Ask a question, get a RAG-powered answer
    GET  /files          — List ingested files
    GET  /stats          — Chunk counts, file counts
    GET  /               — Serves the frontend
"""

import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from db import get_conn, init_db, EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
TOP_K = int(os.environ.get("TOP_K", 5))

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG System", version="0.1.0")


@app.on_event("startup")
def startup():
    init_db()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class FileInfo(BaseModel):
    id: int
    filename: str
    ingested_at: str


class HistoryEntry(BaseModel):
    id: int
    query_text: str
    answer: str
    created_at: str


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
    total: int


class ChunkInfo(BaseModel):
    id: int
    chunk_index: int
    page_number: int | None
    text: str


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
def search_chunks(query_embedding: list[float], top_k: int) -> list[dict]:
    """Find the top_k most similar chunks by cosine distance."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT c.id, c.text, c.page_number, c.chunk_index, f.filename,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    )

    results = []
    for row in cur.fetchall():
        results.append({
            "chunk_id": row[0],
            "text": row[1],
            "page_number": row[2],
            "chunk_index": row[3],
            "filename": row[4],
            "similarity": round(float(row[5]), 4),
        })

    cur.close()
    conn.close()
    return results


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Answer ONLY based on the provided context chunks.
- If the context doesn't contain enough information to answer, say so clearly.
- Cite your sources by referencing the filename and page number.
- Be concise and direct."""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Send query + retrieved chunks to Claude and return the answer."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['filename']}, page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    user_message = f"""Context:
{context_block}

Question: {question}"""

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    # 1. Embed the question
    query_embedding = embed_query(req.question)

    # 2. Retrieve relevant chunks
    chunks = search_chunks(query_embedding, req.top_k)

    if not chunks:
        return QueryResponse(
            answer="No documents have been ingested yet. Run the ingest pipeline first.",
            sources=[],
            query=req.question,
        )

    # 3. Generate answer
    answer = generate_answer(req.question, chunks)

    # 4. Log the query
    try:
        conn = get_conn()
        cur = conn.cursor()
        chunk_ids = [c["chunk_id"] for c in chunks]
        cur.execute(
            "INSERT INTO queries (query_text, retrieved_chunk_ids, answer) VALUES (%s, %s, %s)",
            (req.question, chunk_ids, answer),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass  # logging failure shouldn't break the response

    # 5. Return
    sources = [
        {
            "filename": c["filename"],
            "page": c["page_number"],
            "similarity": c["similarity"],
            "excerpt": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        }
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources, query=req.question)


@app.get("/files", response_model=list[FileInfo])
def list_files():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, ingested_at FROM files ORDER BY ingested_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        FileInfo(id=r[0], filename=r[1], ingested_at=str(r[2]))
        for r in rows
    ]


@app.get("/stats")
def stats():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM files")
    file_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM queries")
    query_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return {
        "files": file_count,
        "chunks": chunk_count,
        "queries_logged": query_count,
    }


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = 20, offset: int = 0):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM queries")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT id, query_text, answer, created_at FROM queries ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limit, offset),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return HistoryResponse(
        entries=[
            HistoryEntry(id=r[0], query_text=r[1], answer=r[2], created_at=str(r[3]))
            for r in rows
        ],
        total=total,
    )


@app.get("/files/{file_id}/chunks", response_model=list[ChunkInfo])
def get_file_chunks(file_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, chunk_index, page_number, text FROM chunks WHERE file_id = %s ORDER BY chunk_index",
        (file_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        raise HTTPException(404, "File not found or has no chunks")
    return [
        ChunkInfo(id=r[0], chunk_index=r[1], page_number=r[2], text=r[3])
        for r in rows
    ]


DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    dest = DATA_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)
    from ingest import ingest_file
    background_tasks.add_task(ingest_file, dest)
    return {"filename": file.filename, "status": "processing"}


@app.delete("/chunks/{chunk_id}")
def delete_chunk(chunk_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE id = %s RETURNING id", (chunk_id,))
    deleted = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    if not deleted:
        raise HTTPException(404, "Chunk not found")
    return {"deleted": chunk_id}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


app.mount("/pdfs", StaticFiles(directory="data"), name="pdfs")
