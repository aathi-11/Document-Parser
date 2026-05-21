from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from threading import Lock
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

    batch_size = max(1, batch_size)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)

    results: List[tuple[int, List[list[float]]]] = []
    completed_batches = 0
    lock = Lock()

    max_workers = min(8, total_batches)
    print(f"[EMBED] Parallelizing embedding generation across {total_batches} batches with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                ollama_embed,
                settings.ollama_base_url,
                settings.ollama_embed_model,
                batch
            ): idx
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                batch_embeddings = future.result()
            except Exception as exc:
                print(f"[EMBED ERROR] Batch {idx} failed: {exc}")
                raise exc

            with lock:
                results.append((idx, batch_embeddings))
                completed_batches += 1
                if progress_callback:
                    progress_callback(completed_batches, total_batches)

    # Restore the original order
    results.sort(key=lambda x: x[0])

    all_embeddings: List[list[float]] = []
    for _, batch_embeddings in results:
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
