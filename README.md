# rag-qa-bot

Ask questions about your own documents from the command line. Documents are chunked and embedded
locally (`all-MiniLM-L6-v2`), stored in a persistent ChromaDB collection, and answered by Claude
with numbered source citations.

## Quick start

```bash
make install                                   # 1. install the package + dev tools
cp .env.example .env && $EDITOR .env           # 2. set ANTHROPIC_API_KEY
cp ~/my-docs/* data/ && make ingest && make query Q="What does the contract say about renewal?"
```

Requires Python 3.10+. The first `ingest` downloads the embedding model (~90 MB).

## Commands

| Command | What it does |
|---|---|
| `rag-bot ingest [PATH]` | Load, chunk, embed and store `.txt .md .pdf .docx`. Defaults to `DATA_DIR`. Safe to re-run: a changed file replaces its old chunks. |
| `rag-bot query "question" [-k N] [--min-score X]` | One-shot answer with sources. |
| `rag-bot chat` | Interactive REPL. `exit` or Ctrl-D to quit. |
| `rag-bot reset [-y]` | Delete the collection. |

Add `-v` before the command (`rag-bot -v query ...`) for debug logs. Logs go to stderr; answers go to stdout.

Make targets: `install`, `ingest`, `query Q="..."`, `chat`, `test`, `lint`, `format`.

## Configuration

Set via environment variables or `.env`.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Required for `query` and `chat`. |
| `CHROMA_DIR` | `./chroma_db` | Where the vector store persists. |
| `DATA_DIR` | `./data` | Default folder for `ingest`. Also the base for source names in citations. |
| `COLLECTION_NAME` | `documents` | Chroma collection name. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it requires `reset` + re-ingest. |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Anthropic model ID. |
| `CHUNK_SIZE` | `1000` | Max characters per chunk. |
| `CHUNK_OVERLAP` | `200` | Characters shared between adjacent chunks. Must be < `CHUNK_SIZE`. |
| `TOP_K` | `4` | Max chunks sent to the model. |
| `MIN_SCORE` | `0.30` | Minimum cosine similarity (−1 to 1) for a chunk to be used. |
| `MAX_TOKENS` | `1024` | Max tokens in the answer. |
| `TEMPERATURE` | `0.0` | Sampling temperature. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

## Development

```bash
make test     # pytest (LLM and vector store are mocked; no network or API key needed)
make lint     # ruff check + format check
make format
```

## Known limitations

- **No chat memory.** `chat` answers each question independently, so follow-ups like "and the second one?" won't resolve.
- **No OCR.** Scanned PDFs with no text layer are skipped (with a warning).
- **Basic DOCX support.** Paragraphs and table cells are extracted; headers, footers, footnotes and images are ignored. Legacy `.doc` is unsupported.
- **Chunks don't span PDF pages.** Text that continues across a page break is split there.
- **Re-ingest re-embeds every file** in the target. There is no change detection, and files deleted from disk are not removed from the store (use `reset`).
- **`MIN_SCORE` needs tuning.** 0.30 is a starting point for MiniLM; a strict value can yield "nothing found", a loose one admits noise.
- **Dense retrieval only.** No keyword/hybrid search or reranking, so exact IDs and rare terms may be missed. MiniLM truncates input at 256 tokens, which a 1000-character chunk can approach (more for code or non-English text); lower `CHUNK_SIZE` if recall suffers.
- **Citations depend on the model.** Sources shown are the `[n]` markers Claude emitted. A claim left uncited shows no source.
- **Single collection, single process.** Chroma's persistent client isn't designed for concurrent writers.
- **Plain-text output only.** No streaming and no web UI, by design.
