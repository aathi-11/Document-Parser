import logging
from typing import List
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

from app.config import settings

logger = logging.getLogger(__name__)

def ollama_health_check(base_url: str) -> None:
    """Raise a clear RuntimeError if the Ollama server is not reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except RequestsConnectionError:
        raise RuntimeError(
            f"Ollama is not running at {base_url}. "
            "Please start Ollama (open the Ollama app or run 'ollama serve') and try again."
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama health check failed: {exc}")


def ollama_embed(base_url: str, model: str, texts: str | List[str]) -> List[float] | List[List[float]]:
    url = f"{base_url}/api/embed"

    if isinstance(texts, str):
        # Single string path — return a zero-vector for empty input
        clean = texts.strip()
        if not clean:
            raise RuntimeError("Cannot embed an empty string.")
        payload = {"model": model, "input": clean}
        try:
            response = requests.post(url, json=payload, timeout=300)
        except RequestsConnectionError:
            raise RuntimeError(
                f"Ollama is not running at {base_url}. "
                "Please start Ollama (open the Ollama app or run 'ollama serve') and try again."
            )
        response.raise_for_status()
        data = response.json()
        if "embeddings" not in data:
            raise RuntimeError("Ollama embeddings response missing embeddings field.")
        return data["embeddings"][0]

    # Batch path — filter empties and map results back to original positions
    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    non_empty_texts = [texts[i].strip() for i in non_empty_indices]

    if not non_empty_texts:
        raise RuntimeError("All texts in the embedding batch are empty.")

    payload = {"model": model, "input": non_empty_texts}
    try:
        response = requests.post(url, json=payload, timeout=300)
    except RequestsConnectionError:
        raise RuntimeError(
            f"Ollama is not running at {base_url}. "
            "Please start Ollama (open the Ollama app or run 'ollama serve') and try again."
        )
    response.raise_for_status()
    data = response.json()
    if "embeddings" not in data:
        raise RuntimeError("Ollama embeddings response missing embeddings field.")

    fetched = data["embeddings"]
    dim = len(fetched[0]) if fetched else 0

    # Re-insert zero-vectors for any positions that were empty
    result: List[List[float]] = [[0.0] * dim for _ in range(len(texts))]
    for out_pos, orig_pos in enumerate(non_empty_indices):
        result[orig_pos] = fetched[out_pos]
    return result


def ollama_chat(base_url: str, model: str, messages: List[dict]) -> str:
    if settings.groq_api_key:
        logger.info(f"Using Groq API for chat with model: {settings.groq_chat_model}")
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json"
        }
        url = f"{settings.groq_base_url}/chat/completions"
        payload = {
            "model": settings.groq_chat_model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API call failed: {e}. Falling back to local Ollama.")
            
    elif settings.grok_api_key:
        logger.info(f"Using Grok API for chat with model: {settings.grok_chat_model}")
        headers = {
            "Authorization": f"Bearer {settings.grok_api_key}",
            "Content-Type": "application/json"
        }
        url = f"{settings.grok_base_url}/chat/completions"
        payload = {
            "model": settings.grok_chat_model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Grok API call failed: {e}. Falling back to local Ollama.")
            
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
    except RequestsConnectionError:
        raise RuntimeError(
            f"Ollama is not running at {base_url}. "
            "Please start Ollama (open the Ollama app or run 'ollama serve') and try again."
        )
    response.raise_for_status()
    data = response.json()

    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    if "response" in data:
        return data["response"]

    raise RuntimeError("Ollama chat response missing content field.")


