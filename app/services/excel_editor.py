import contextlib
import io
import logging
import re
import concurrent.futures
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.services.tabular_preprocessor import preprocess_tabular_files
from app.services.tabular_query import (
    _ensure_safe_code,
    _run_with_timeout,
    sanitize_df_name,
)

logger = logging.getLogger(__name__)

_EDIT_TRIGGERS = [
    "pivot",
    "pivot table",
    "add column",
    "add a column",
    "new column",
    "create column",
    "calculate column",
    "append column",
    "insert column",
    "rename column",
    "drop column",
    "remove column",
    "delete column",
    "fill column",
    "update column",
    "modify",
    "edit the",
    "transform",
    "reshape",
    "melt",
    "unpivot",
    "transpose",
    "group by and pivot",
    "crosstab",
]


def is_edit_question(question: str) -> bool:
    lowered = question.lower()
    return any(trigger in lowered for trigger in _EDIT_TRIGGERS)


def run_excel_edit(
    session_dir: Path,
    question: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    cleaned_dir = preprocess_tabular_files(session_dir)
    if not cleaned_dir.exists():
        return {"handled": False}

    csv_files = [p for p in cleaned_dir.iterdir() if p.suffix.lower() == ".csv"]
    if not csv_files:
        return {"handled": False}

    preloaded_dfs: Dict[str, pd.DataFrame] = {}
    schema_parts: list[str] = []

    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            logger.error(f"Error preloading/parsing {csv_file.name}: {exc}")
            continue

        var_name = sanitize_df_name(csv_file.name)
        preloaded_dfs[var_name] = df
        cols = df.columns.tolist()
        sample_rows = df.head(2).to_string(index=False)

        schema_parts.append(
            f"- DataFrame variable: `{var_name}` (sourced from '{csv_file.name}')\n"
            f"  Columns: {cols}\n"
            f"  Sample Data:\n"
            f"  {sample_rows}"
        )

    if not preloaded_dfs:
        return {"handled": False}

    schema_str = "\n\n".join(schema_parts)

    prompt = f"""You are a professional Python data analyst. The user wants to modify or transform data.

The following pandas DataFrames are pre-loaded in the environment:
{schema_str}

Instructions:
- Write Python Pandas code to perform the requested transformation.
- The final result must be stored in a variable called exactly result_df (mandatory).
- If the operation is a pivot, result_df should be the pivot table (use pd.pivot_table or df.pivot_table).
- If the operation adds or modifies a column, result_df should be the modified DataFrame.
- Do NOT use pd.read_csv or read any files from disk.
- Do NOT import pandas or numpy (already available as pd and np).
- Do NOT call plt.show() or any plotting functions.
- Print result_df to stdout at the end using print(result_df.to_string()).
- Return only the Python code block starting with ```python and ending with ```.

User instruction: {question}
Python Code:"""

    from app.services.ollama_client import ollama_chat

    messages = [{"role": "user", "content": prompt}]
    success = False
    result_df: pd.DataFrame | None = None

    for attempt in range(1, 4):
        code = ""
        try:
            text = ollama_chat(base_url, model, messages)

            code = text
            if "```python" in text:
                code = text.split("```python")[1].split("```")[0]

            _ensure_safe_code(code)

            exec_globals = {"pd": pd, "np": np, **preloaded_dfs}
            exec_locals: Dict[str, Any] = {}

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                _run_with_timeout(lambda: exec(code, exec_globals, exec_locals), 20)

            result_df = exec_locals.get("result_df")
            if result_df is None:
                result_df = exec_globals.get("result_df")

            if isinstance(result_df, pd.DataFrame):
                success = True
                break

            raise RuntimeError("result_df was not set to a pandas DataFrame.")
        except Exception:
            import traceback

            error_str = traceback.format_exc()
            logger.warning(
                f"Excel edit execution failed on attempt {attempt}. Error:\n{error_str}"
            )
            if code:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            error_msg = (
                "The code execution failed with the following traceback/error:\n"
                f"{error_str.strip()}\n\n"
                "Please correct the code. Ensure you use the correct preloaded DataFrame variables "
                "and write correct Pandas code. Return ONLY the executable python block."
            )
            messages.append({"role": "user", "content": error_msg})

    if not success or result_df is None:
        return {"handled": False}

    output_dir = session_dir / "edited_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    words = re.findall(r"[A-Za-z0-9]+", question.lower())[:6]
    base_name = "_".join(words) if words else "edited_output"
    base_name = re.sub(r"[^A-Za-z0-9_]", "", base_name).strip("_") or "edited_output"
    filename = f"{base_name}_edited.xlsx"

    output_path = output_dir / filename
    result_df.to_excel(output_path, index=True, engine="openpyxl")

    return {
        "handled": True,
        "filename": output_path.name,
        "row_count": len(result_df),
        "col_count": len(result_df.columns),
        "preview": result_df.head(10).to_dict(orient="records"),
        "columns": list(result_df.columns.astype(str)),
    }


def get_edited_file_path(session_dir: Path, filename: str) -> Path | None:
    safe_name = re.sub(r"[^A-Za-z0-9_\-.]", "", Path(filename).name)
    if not safe_name:
        return None
    file_path = session_dir / "edited_outputs" / safe_name
    return file_path if file_path.exists() else None
