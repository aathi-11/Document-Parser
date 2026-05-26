import json
from pathlib import Path
import re
import shutil
import time
from typing import Dict, List
from uuid import uuid4

from fastapi import UploadFile


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_filename(name: str) -> str:
    base = Path(name).name
    sanitized = _SAFE_FILENAME_RE.sub("_", base)
    return sanitized or "file"


def create_session_dir(storage_dir: Path, session_id: str | None = None) -> tuple[str, Path]:
    session_id = session_id or uuid4().hex
    session_dir = storage_dir / "sessions" / session_id
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


async def save_upload_files(
    session_dir: Path,
    files: List[UploadFile],
    max_bytes: int | None = None,
) -> tuple[List[Dict[str, str]], List[str]]:
    upload_dir = session_dir / "uploads"
    saved: List[Dict[str, str]] = []
    warnings: List[str] = []

    for file in files:
        safe_name = safe_filename(file.filename or "file")
        max_mb = max_bytes // (1024 * 1024) if max_bytes else None
        file_path = upload_dir / safe_name
        if file_path.exists():
            base = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while True:
                candidate = upload_dir / f"{base}_{counter}{suffix}"
                if not candidate.exists():
                    file_path = candidate
                    safe_name = candidate.name
                    break
                counter += 1
        content = await file.read()
        if max_bytes is not None and len(content) > max_bytes:
            warnings.append(f"{safe_name}: file too large (max {max_mb} MB).")
            continue
        file_path.write_bytes(content)
        saved.append({"path": str(file_path), "filename": safe_name})

    return saved, warnings


def cleanup_old_sessions(storage_dir: Path, ttl_days: int) -> int:
    if ttl_days <= 0:
        return 0

    sessions_dir = storage_dir / "sessions"
    if not sessions_dir.exists():
        return 0

    cutoff = time.time() - (ttl_days * 24 * 60 * 60)
    removed = 0
    for session_path in sessions_dir.iterdir():
        if not session_path.is_dir():
            continue
        try:
            if session_path.stat().st_mtime < cutoff:
                shutil.rmtree(session_path, ignore_errors=True)
                removed += 1
        except FileNotFoundError:
            continue

    return removed


def save_summary(session_dir: Path, filename: str, summary_data: dict) -> None:
    summaries_dir = session_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(filename)
    summary_path = summaries_dir / f"{safe_name}.summary.json"
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def load_summaries(session_dir: Path) -> dict[str, dict]:
    summaries_dir = session_dir / "summaries"
    if not summaries_dir.exists():
        return {}

    summaries: dict[str, dict] = {}
    for summary_path in summaries_dir.glob("*.summary.json"):
        try:
            raw = summary_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue

        name = summary_path.name
        if name.endswith(".summary.json"):
            key = name[: -len(".summary.json")]
        else:
            key = summary_path.stem
        summaries[key] = data

    return summaries
