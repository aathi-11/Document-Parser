# Document Parser (Ollama RAG)

Local document Q&A with Ollama. Upload one or more files, build a vector index, and ask questions over the content. Includes lightweight, client-side charting for CSV or TXT uploads.

## Features
- Local-first RAG using Ollama chat + embedding models
- Multi-file upload; re-uploads append to the current session
- Optional OCR via Tesseract for images
- Client-side charts from CSV/TXT when requested
- Configurable chunking and retrieval parameters

## Requirements
- Python 3.10+
- Ollama running locally
- Models: `qwen2.5-coder:3b` (or `gemma3:4b`) and `nomic-embed-text:latest`
- Optional: Tesseract OCR

## Quickstart
1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure Ollama models are available locally
4. Start the server: `uvicorn app.main:app --reload`
5. Open `http://localhost:8000`

## Usage
- Upload one or more files from the UI.
- Ask questions in the chat box.
- For charts, ask explicitly (e.g., "pie chart of expenses"). The chart is generated from retrieved context in the browser.

## Configuration
Environment variables (defaults in parentheses):
- OLLAMA_BASE_URL (http://localhost:11434)
- OLLAMA_CHAT_MODEL (qwen2.5-coder:3b)
- OLLAMA_EMBED_MODEL (nomic-embed-text:latest)
- STORAGE_DIR (app/storage)
- CHUNK_SIZE (800 words)
- CHUNK_OVERLAP (120 words)
- TOP_K (4)
- FETCH_K (15)
- MAX_CONTEXT_CHARS (6000)
- MAX_UPLOAD_MB (50)

## Notes
- If OCR is enabled but Tesseract is not installed, image parsing will fail.
- Large files may take time to embed; start with small documents to validate setup.
- Files larger than MAX_UPLOAD_MB are skipped with a warning.
