# RAG System

PDF question-answering system using pgvector, FastAPI, and Claude.

## Architecture

```
PDFs → ingest.py → [extract → chunk → embed] → PostgreSQL (pgvector)
                                                       ↓
Frontend → FastAPI (query.py) → [embed query → vector search → Claude] → Answer
```

## Quick Start (Local with Docker)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 2. Launch

```bash
docker compose up -d --build
```

Check logs:
```bash
docker compose logs -f
```

### 3. Ingest PDFs

Drop PDFs into the `data/` folder, then:

```bash
# Ingest all PDFs
docker compose exec app python ingest.py

# Or a specific file
docker compose exec app python ingest.py myfile.pdf
```

### 4. Use it

Open `http://localhost:8000` in your browser.

Or test with curl:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## API Endpoints

| Method | Path     | Description                    |
|--------|----------|--------------------------------|
| POST   | /query   | Ask a question                 |
| GET    | /files   | List ingested files            |
| GET    | /stats   | Chunk/file/query counts        |
| GET    | /        | Frontend UI                    |

## File Structure

```
rag-system/
├── docker-compose.yml    # Postgres + App services
├── Dockerfile            # Python 3.12 container
├── .env                  # API keys (not committed)
├── .env.example          # Template
├── requirements.txt      # Python deps
├── db.py                 # DB connection + schema
├── ingest.py             # PDF → chunks → embeddings → DB
├── query.py              # FastAPI server + Claude generation
├── frontend/
│   └── index.html        # UI
├── data/                 # Drop PDFs here
│   └── ...
└── README.md
```

## Notes

- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Generation**: Claude Sonnet via Anthropic API
- **Vector search**: pgvector HNSW index with cosine similarity
- **Chunking**: Fixed 500-char windows with 50-char overlap (configurable via .env)
