from math import ceil
from typing import Callable, List

from app.config import settings
from app.services.ollama_client import ollama_embed

# Must stay in sync with chunking._MAX_EMBED_CHARS; 7 000 gives a small
# extra margin above the chunking limit for any metadata added post-chunking.
_MAX_EMBED_CHARS = 7_000


def embed_texts(
    texts: List[str],
    batch_size: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
) -> List[list[float]]:
    # Drop chunks that are entirely whitespace — Ollama returns 400 for empty input.
    # Also truncate anything that would exceed the model's token limit.
    texts = [t[:_MAX_EMBED_CHARS] for t in texts if t.strip()]
    if not texts:
        return []

    # Smaller batches reduce long pauses during embeddings.
    all_embeddings: List[list[float]] = []

    batch_size = max(1, batch_size)
    total_batches = ceil(len(texts) / batch_size)
    current_batch = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = ollama_embed(settings.ollama_base_url, settings.ollama_embed_model, batch)
        all_embeddings.extend(batch_embeddings)
        current_batch += 1
        if progress_callback:
            progress_callback(current_batch, total_batches)

    return all_embeddings
