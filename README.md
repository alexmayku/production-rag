# RAG System

PDF question-answering system using pgvector, FastAPI, and Claude.

## Architecture

```
PDFs → ingest.py → [extract → chunk → embed] → PostgreSQL (pgvector)
                                                       ↓
Frontend → FastAPI (query.py) → [embed query → vector search → Claude] → Answer
```

## Quick Start (Hetzner Deployment)

### 1. SSH into your server

```bash
ssh root@159.69.186.14
```

(Use the private key from the learning portal — save it to `~/.ssh/fac_key`
and run `chmod 600 ~/.ssh/fac_key`, then `ssh -i ~/.ssh/fac_key root@159.69.186.14`)

### 2. Install Docker (if not already installed)

```bash
curl -fsSL https://get.docker.com | sh
```

### 3. Clone / upload the project

```bash
# Option A: git clone (if you push to a repo)
git clone <your-repo-url> rag-system
cd rag-system

# Option B: rsync from your local machine
# (run this locally, not on the server)
rsync -avz ./rag-system/ root@159.69.186.14:/root/rag-system/
```

### 4. Configure environment

```bash
cd /root/rag-system
cp .env.example .env
nano .env   # Add your ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 5. Launch

```bash
docker compose up -d --build
```

Check logs:
```bash
docker compose logs -f
```

### 6. Ingest PDFs

Drop PDFs into the `data/` folder, then:

```bash
# Ingest all PDFs
docker compose exec app python ingest.py

# Or a specific file
docker compose exec app python ingest.py myfile.pdf
```

### 7. Use it

Open `http://159.69.186.14:8000` in your browser.

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
