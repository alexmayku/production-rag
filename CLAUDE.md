# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG (Retrieval-Augmented Generation) system for PDF question-answering. PDFs are ingested into PostgreSQL with pgvector embeddings, then queried via FastAPI with Claude generating answers from retrieved chunks.

## Architecture

The pipeline has two phases:
1. **Ingest** (`ingest.py`): PDF → page extraction (PyMuPDF) → fixed-size character chunking → OpenAI embedding → PostgreSQL/pgvector storage
2. **Query** (`query.py`): User question → OpenAI embedding → cosine similarity vector search → top-K chunks → Claude generates answer with citations

`db.py` is the shared database layer (connection helper, schema init with pgvector extension, HNSW index).

The frontend is a single-page app (`frontend/index.html`) served by FastAPI at `/`. PDFs are served statically at `/pdfs/`.

## Common Commands

### Local development (requires running PostgreSQL with pgvector)
```bash
# Run the FastAPI server
uvicorn query:app --host 0.0.0.0 --port 8000 --reload

# Ingest all PDFs in data/
python ingest.py

# Ingest a specific file
python ingest.py myfile.pdf
```

### Docker (primary deployment method)
```bash
docker compose up -d --build          # Start app + postgres
docker compose logs -f                # Tail logs
docker compose exec app python ingest.py   # Ingest PDFs
docker compose down                   # Stop
```

## Key Configuration

All config is via environment variables (see `.env.example`):
- `DATABASE_URL` — set automatically by docker-compose
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — required API keys
- `CHUNK_SIZE` (default 500), `CHUNK_OVERLAP` (default 50) — chunking params in characters
- `TOP_K` (default 5) — number of chunks retrieved per query
- `CLAUDE_MODEL`, `EMBEDDING_MODEL` — model selection

## API Endpoints

- `POST /query` — ask a question (body: `{"question": "...", "top_k": 5}`)
- `GET /files` — list ingested files
- `GET /stats` — file/chunk/query counts
- `GET /history` — paginated query history (`?limit=20&offset=0`)
- `GET /files/{file_id}/chunks` — get chunks for a file
- `POST /upload` — upload and ingest a PDF (background task)
- `DELETE /chunks/{chunk_id}` — delete a chunk

## Important Details

- Embeddings are 1536-dimensional (OpenAI `text-embedding-3-small`), stored as pgvector `vector(1536)` with an HNSW index using cosine similarity
- `ingest.py` skips already-ingested files (checks by filename in `files` table)
- `ingest.py` hardcodes `DATA_DIR = Path("/app/data")` — this is the Docker container path; locally, PDFs go in `./data/`
- The `/upload` endpoint triggers ingestion as a FastAPI background task by importing `ingest_file` from `ingest.py`
- Database connections are created per-request (no connection pool) via `get_conn()`
