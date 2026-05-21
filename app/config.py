from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: app/config.py → project root)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent
    storage_dir: Path | None = None
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5-coder:3b") # or gemma3:4b
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    fetch_k: int = int(os.getenv("FETCH_K", "15"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "620000"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))

    # ── Groq (set GROQ_API_KEY or set GROK_API_KEY to a gsk_ key to enable) ──
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # ── Grok / xAI (set GROK_API_KEY to enable; leave blank to use Ollama for chat) ──
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    grok_chat_model: str = os.getenv("GROK_CHAT_MODEL", "grok-3-mini")
    grok_base_url: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

    def __post_init__(self) -> None:
        if self.storage_dir is None:
            resolved = Path(os.getenv("STORAGE_DIR", str(self.base_dir / "storage")))
            object.__setattr__(self, "storage_dir", resolved)

        # Automatically detect if GROK_API_KEY is actually a Groq API key (starts with gsk_)
        grok_key = os.getenv("GROK_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        
        if not groq_key and grok_key.startswith("gsk_"):
            object.__setattr__(self, "groq_api_key", grok_key)
            object.__setattr__(self, "grok_api_key", "")


settings = Settings()
