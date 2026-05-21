from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
from threading import Lock
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.models import AskRequest
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.extraction import extract_text_from_file
from app.services.ollama_client import ollama_chat, ollama_embed, ollama_health_check
from app.services.session_store import create_session_dir, save_upload_files
from app.services.tabular_query import (
    is_aggregation_question,
    run_tabular_query,
    TABULAR_EXTS,
)
from app.services.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent

SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CHART_TYPES = {
    "pie", "bar", "line", "histogram",
    "scatter", "area", "donut", "bubble",
    "radar", "funnel", "waterfall",
    "stacked_bar", "grouped_bar",
}
progress_lock = Lock()
progress_store: dict[str, dict] = {}


def _validate_session_id(value: str) -> str:
    if not SESSION_ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid session ID.")
    return value


def _wants_chart(question: str) -> bool:
    lowered = question.strip().strip("\"'").lower()
    keywords = [
        "chart",
        "graph",
        "plot",
        "pie",
        "bar",
        "line",
        "histogram",
        "visualize",
        "visualization",
        "scatter",
        "trend",
        "compare",
        "distribution",
        "breakdown",
        "show me",
        "display",
        "area",
    ]
    return any(keyword in lowered for keyword in keywords)


def _parse_chart_spec(raw: str) -> dict | None:
    # Strip markdown code fences if the LLM wrapped the JSON
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    chart_type = str(data.get("type", "")).lower().strip()
    if chart_type not in CHART_TYPES:
        return None

    labels = data.get("labels")
    values = data.get("values")
    if not isinstance(labels, list) or not isinstance(values, list):
        return None
    if len(labels) == 0 or len(labels) != len(values):
        return None

    # values may be plain floats OR arrays of floats (scatter/bubble datasets)
    cleaned_values: List[float] | List[list] = []
    first = values[0]
    if isinstance(first, (list, dict)):
        # Pass through nested structures (e.g. scatter {x,y} or boxplot arrays)
        cleaned_values = values  # type: ignore[assignment]
    else:
        for value in values:
            try:
                cleaned_values.append(float(value))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

    title = str(data.get("title", "")).strip() or f"{chart_type.title()} chart"

    return {
        "type": chart_type,
        "labels": labels,
        "values": cleaned_values,
        "title": title,
    }


def _chart_summary(chart_spec: dict) -> str:
    title = chart_spec.get("title", "Chart")
    labels = chart_spec.get("labels")
    values = chart_spec.get("values")

    if isinstance(labels, list) and labels:
        if isinstance(values, list) and values and not isinstance(values[0], (list, dict)):
            return f"Chart generated: {title} ({len(labels)} points, {labels[0]} to {labels[-1]})."

    return f"Chart generated: {title}."


def _chat(messages: List[dict]) -> str:
    """Dispatch a chat request to Ollama."""
    return ollama_chat(settings.ollama_base_url, settings.ollama_chat_model, messages)


def _set_progress(
    session_id: str,
    stage: str,
    percent: int,
    processed_files: int = 0,
    total_files: int = 0,
    message: str = "",
) -> None:
    with progress_lock:
        progress_store[session_id] = {
            "session_id": session_id,
            "stage": stage,
            "percent": max(0, min(100, int(percent))),
            "processed_files": processed_files,
            "total_files": total_files,
            "message": message,
            "updated_at": time.time(),
        }


def _get_progress(session_id: str) -> dict | None:
    with progress_lock:
        return progress_store.get(session_id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    (settings.storage_dir / "sessions").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Document Parser", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/api/progress/{session_id}")
def get_progress(session_id: str) -> dict:
    session_id = _validate_session_id(session_id)
    data = _get_progress(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Progress not found.")
    return data


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: str | None = Form(None),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Fail fast if Ollama is unreachable before we do any file I/O
    try:
        await run_in_threadpool(ollama_health_check, settings.ollama_base_url)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    session_id = session_id.strip() if session_id else None
    append_to_session = False

    if session_id:
        session_id = _validate_session_id(session_id)
        session_dir = settings.storage_dir / "sessions" / session_id
        if session_dir.exists():
            append_to_session = True
        else:
            session_id, session_dir = create_session_dir(settings.storage_dir, session_id=session_id)
    else:
        session_id, session_dir = create_session_dir(settings.storage_dir)

    _set_progress(
        session_id,
        "saving",
        0,
        processed_files=0,
        total_files=len(files),
        message="Saving uploads",
    )

    warnings: List[str] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024
    saved_files, save_warnings = await save_upload_files(
        session_dir, files, max_bytes=max_bytes
    )
    warnings.extend(save_warnings)

    total_files = len(saved_files)
    if total_files == 0:
        _set_progress(
            session_id,
            "error",
            0,
            processed_files=0,
            total_files=0,
            message="No valid files to process",
        )
        raise HTTPException(status_code=400, detail="No text extracted from the uploaded files.")

    _set_progress(
        session_id,
        "indexing",
        0,
        processed_files=0,
        total_files=total_files,
        message="Extracting text",
    )

    all_chunks: List[str] = []
    all_meta: List[dict] = []
    processed_files = 0
    for saved in saved_files:
        text = ""
        extracted = False
        try:
            text = await run_in_threadpool(
                extract_text_from_file,
                saved["path"],
                saved["filename"],
            )
            extracted = True
        except Exception as exc:
            warnings.append(f"Failed to parse {saved['filename']}: {exc}")

        if extracted and not text.strip():
            warnings.append(f"No text extracted from {saved['filename']}.")

        if extracted and text.strip():
            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap, filename=saved["filename"])
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({"file_name": saved["filename"], "chunk_index": idx})

        processed_files += 1
        percent = int((processed_files / total_files) * 70)
        _set_progress(
            session_id,
            "indexing",
            percent,
            processed_files=processed_files,
            total_files=total_files,
            message="Extracting text",
        )

    if not all_chunks:
        _set_progress(
            session_id,
            "error",
            0,
            processed_files=processed_files,
            total_files=total_files,
            message="No text extracted",
        )
        raise HTTPException(status_code=400, detail="No text extracted from the uploaded files.")

    try:
        _set_progress(
            session_id,
            "embedding",
            80,
            processed_files=processed_files,
            total_files=total_files,
            message="Embedding...",
        )

        def _embedding_progress(done: int, total: int) -> None:
            percent = 80 + int((done / max(total, 1)) * 20)
            _set_progress(
                session_id,
                "embedding",
                percent,
                processed_files=processed_files,
                total_files=total_files,
                message="Embedding...",
            )

        embeddings = await run_in_threadpool(embed_texts, all_chunks, 8, _embedding_progress)
    except Exception as exc:
        _set_progress(
            session_id,
            "error",
            0,
            processed_files=processed_files,
            total_files=total_files,
            message=f"Embedding failed: {exc}",
        )
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    data_path = session_dir / "data.json"
    embeddings_path = session_dir / "embeddings.npy"
    if data_path.exists() and embeddings_path.exists():
        store = await run_in_threadpool(VectorStore.get_session_store, session_dir, session_id)
    else:
        store = VectorStore([], [], [])

    store.extend(embeddings, all_chunks, all_meta)
    await run_in_threadpool(store.save, session_dir)

    _set_progress(
        session_id,
        "completed",
        100,
        processed_files=processed_files,
        total_files=total_files,
        message="Completed",
    )

    return {
        "session_id": session_id,
        "files": [saved["filename"] for saved in saved_files],
        "chunks_added": len(all_chunks),
        "chunks_total": len(store.chunks),
        "warnings": warnings,
        "appended": append_to_session,
    }


@app.post("/api/ask")
async def ask_question(payload: AskRequest) -> dict:
    session_id = _validate_session_id(payload.session_id)
    session_dir = settings.storage_dir / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    # ------------------------------------------------------------------
    # Fast path: deterministic tabular answering for count/aggregation
    # questions on uploaded spreadsheet / CSV files.
    # ------------------------------------------------------------------
    upload_dir = session_dir / "uploads"
    has_tabular_uploads = upload_dir.exists() and any(
        p.suffix.lower() in TABULAR_EXTS for p in upload_dir.iterdir()
    )

    if has_tabular_uploads and is_aggregation_question(payload.question, session_dir):
        cleaned_dir = session_dir / "cleaned_csvs"
        csv_files = [p for p in cleaned_dir.iterdir() if p.suffix.lower() == ".csv"] if cleaned_dir.exists() else []
        
        if len(csv_files) > 1:
            try:
                tabular_result = await run_in_threadpool(
                    run_tabular_query,
                    session_dir,
                    payload.question,
                    settings.ollama_base_url,
                    settings.ollama_chat_model,
                )
            except Exception:
                tabular_result = {"handled": False}

            if tabular_result.get("handled"):
                details = tabular_result.get("details", [])
                sources = [
                    {
                        "file_name": d["file"],
                        "chunk_index": 0,
                        "score": 1.0,
                        "snippet": (
                            f"{d['count']} matching record(s). "
                            + (
                                f"IDs: {', '.join(d.get('matched_ids', [])[:10])}"
                                + ("…" if len(d.get('matched_ids', [])) > 10 else "")
                                if d.get('matched_ids') else ""
                            )
                        ),
                    }
                    for d in details
                ]
                return {
                    "answer": tabular_result["answer"],
                    "sources": sources,
                    "chart_spec": None,
                }

    # ------------------------------------------------------------------
    # Standard RAG path
    # ------------------------------------------------------------------
    try:
        store = await run_in_threadpool(VectorStore.get_session_store, session_dir, session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load session index: {exc}")

    try:
        query_embedding = await run_in_threadpool(
            ollama_embed,
            settings.ollama_base_url,
            settings.ollama_embed_model,
            payload.question,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    top_k = payload.top_k or settings.top_k
    results = await run_in_threadpool(
        store.search,
        query_embedding,
        payload.question,
        top_k,
        settings.fetch_k,
    )
    if not results:
        return {
            "answer": "I could not find relevant context in the uploaded documents.",
            "sources": [],
        }

    # Sort results sequentially by file and chunk order for tabular context
    results_sorted = sorted(
        results,
        key=lambda r: (
            r.get("metadata", {}).get("file_name", ""),
            r.get("metadata", {}).get("chunk_index", 0),
        ),
    )

    context_blocks = []
    current_chars = 0
    for idx, result in enumerate(results_sorted, start=1):
        chunk_content = result["chunk"]
        if current_chars + len(chunk_content) > settings.max_context_chars and current_chars > 0:
            break
        context_blocks.append(f"[{idx}] {chunk_content}")
        current_chars += len(chunk_content)

    context = "\n\n".join(context_blocks)

    is_counting_query = any(
        w in payload.question.lower()
        for w in ["how many", "count", "sum", "average", "total", "breakdown", "list"]
    )
    is_tabular_context = "|" in context

    if is_counting_query and is_tabular_context:
        system_prompt = (
            "Answer using only the provided context. If the answer is not in the context, "
            "say you do not know. "
            "To count accurately, count the records table by table. "
            "Show each table's matching Policy IDs and count them, "
            "then sum them up for the final answer. Keep explanations to a minimum."
        )
    else:
        system_prompt = (
            "Answer using only the provided context. If the answer is not in the context, "
            "say you do not know. Be extremely direct and concise. Return only the correct value, "
            "number, or exact answer with no introductory filler, preamble, or conversational explanation."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\nQuestion: {payload.question}\n\n"
                "Answer the question based on the context."
            ),
        },
    ]

    try:
        answer = await run_in_threadpool(_chat, messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")

    chart_spec = None

    sources = [
        {
            "file_name": result["metadata"]["file_name"],
            "chunk_index": result["metadata"]["chunk_index"],
            "score": result["score"],
            "snippet": result["chunk"][:240],
        }
        for result in results_sorted
    ]

    return {"answer": answer, "sources": sources, "chart_spec": chart_spec}
