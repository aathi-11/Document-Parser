import json
import re

from app.services.ollama_client import ollama_chat


_MAX_SUMMARY_CHARS = 6000


def summarise_document(text: str, filename: str, base_url: str, model: str) -> dict:
    snippet = (text or "")[:_MAX_SUMMARY_CHARS]
    prompt = (
        "Return ONLY a JSON object in this exact format: "
        '{"summary": "one paragraph summary here", "topics": ["topic1", "topic2", "topic3"]}.\n\n'
        "Rules:\n"
        "- Summary is 2-3 sentences.\n"
        "- Topics are 5-8 key themes, entities, or data categories.\n"
        "- No markdown, no code fences, no extra text.\n\n"
        f"Document filename: {filename}\n\n"
        f"Document content:\n{snippet}"
    )

    response = ollama_chat(base_url, model, [{"role": "user", "content": prompt}])
    cleaned = re.sub(r"```[a-z]*\n?", "", response.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    fallback = {"summary": "Summary unavailable.", "topics": []}
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(data, dict):
        return fallback
    summary = data.get("summary")
    topics = data.get("topics")
    if not isinstance(summary, str) or not isinstance(topics, list):
        return fallback

    return {"summary": summary, "topics": topics}
