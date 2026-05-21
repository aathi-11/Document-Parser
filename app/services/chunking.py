import re
from pathlib import Path
from typing import List

# Conservative character limit per chunk — stays well within nomic-embed-text's
# 8 192-token context window even for wide multi-column spreadsheets.
_MAX_EMBED_CHARS = 6_000


def is_tabular_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in {".csv", ".xlsx", ".xls", ".xlsm"}


def to_markdown_table(header_line: str, rows: List[str]) -> str:
    def format_row(line: str) -> str:
        cells = [c.strip().replace("|", "\\|") for c in line.split("\t")]
        return "| " + " | ".join(cells) + " |"
        
    md_header = format_row(header_line)
    num_cols = len(header_line.split("\t"))
    md_separator = "|" + "|".join("---" for _ in range(num_cols)) + "|"
    md_rows = [format_row(row) for row in rows]
    
    return "\n".join([md_header, md_separator] + md_rows)


def chunk_tabular_text(
    text: str,
    filename: str,
    rows_per_chunk: int = 15,
) -> List[str]:
    sections = text.split("\n\n")
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n")

        start_idx = 0
        metadata_line = ""
        if lines[0].startswith("[") and lines[0].endswith("]"):
            metadata_line = lines[0]
            start_idx = 1

        if len(lines) <= start_idx:
            if lines:
                chunks.append(section[:_MAX_EMBED_CHARS])
            continue

        header_line = lines[start_idx]
        data_rows = lines[start_idx + 1:]

        if not data_rows:
            chunks.append(section[:_MAX_EMBED_CHARS])
            continue

        # ── Adaptive rows-per-chunk based on actual row width ──────────────────
        # Sample up to 10 rows to estimate average formatted row length.
        sample = data_rows[:min(10, len(data_rows))]
        avg_raw_row = sum(len(r) for r in sample) / len(sample) if sample else 80
        num_cols = max(1, len(header_line.split("\t")))
        # Markdown pipe formatting adds ~3 chars per cell boundary
        fmt_row_chars = avg_raw_row + num_cols * 3
        # Budget: total limit minus the fixed overhead (label + header + separator)
        overhead = len(metadata_line) + len(header_line) + num_cols * 5 + 60
        available = max(_MAX_EMBED_CHARS - overhead, 1)
        adaptive_rows = max(1, min(rows_per_chunk, int(available // max(fmt_row_chars, 1))))
        # ────────────────────────────────────────────────────────────────

        for i in range(0, len(data_rows), adaptive_rows):
            sub_rows = data_rows[i : i + adaptive_rows]

            # Format rows as a Markdown table
            md_table = to_markdown_table(header_line, sub_rows)

            # Prepend metadata
            if metadata_line:
                label = f"{metadata_line[:-1]} (rows {i + start_idx + 1} to {i + start_idx + len(sub_rows)})]"
            else:
                label = f"[{filename} | rows {i + start_idx + 1} to {i + start_idx + len(sub_rows)}]"

            chunk = f"{label}\n{md_table}"
            # Hard truncation as a final safety net (preserves the label line)
            chunks.append(chunk[:_MAX_EMBED_CHARS])

    return chunks


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, filename: str | None = None) -> List[str]:
    if not text:
        return []

    if filename and is_tabular_file(filename):
        return chunk_tabular_text(text, filename)

    spans = []
    for m in re.finditer(r'\S+', text):
        spans.append((m.start(), m.end()))
        
    if not spans:
        return []
        
    chunk_size = max(chunk_size, 1)
    chunk_overlap = max(min(chunk_overlap, chunk_size - 1), 0)
    
    chunks: List[str] = []
    start = 0
    while start < len(spans):
        end = min(start + chunk_size, len(spans))
        chunk_start_char = spans[start][0]
        chunk_end_char = spans[end - 1][1]
        chunks.append(text[chunk_start_char:chunk_end_char])
        if end == len(spans):
            break
        start = end - chunk_overlap
        
    return chunks

