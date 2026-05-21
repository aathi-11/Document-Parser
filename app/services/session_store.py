from pathlib import Path
import re
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
        size = _get_upload_size(file)
        if max_bytes is not None and size is not None and size > max_bytes:
            warnings.append(f"{safe_name}: file too large (max {max_mb} MB).")
            continue
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


def _get_upload_size(file: UploadFile) -> int | None:
    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        return size
    except Exception:
        return None
