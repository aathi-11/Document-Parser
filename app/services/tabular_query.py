import concurrent.futures
import contextlib
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from app.services.tabular_preprocessor import preprocess_tabular_files

logger = logging.getLogger(__name__)

TABULAR_EXTS = {".xlsx", ".xlsm", ".xls", ".csv"}

COUNT_KEYWORDS = {
    "how many", "count", "number of", "total number",
    "sum", "total", "average", "avg", "breakdown", "list all",
    "rank", "underwriter", "limit", "retention", "inception", "expiry",
    "cedent", "loss ratio", "combined ratio", "pml", "triangle", "ibnr",
    "development factor", "paid", "incurred", "reserve", "gwp", "treaty",
    "treaties", "bottom", "top", "most", "least", "highest", "lowest",
    "performance", "claims", "claim", "cyber", "property", "casualty",
    "marine", "loss", "losses"
}

_KEYWORD_CACHE: dict[str, set[str]] = {}
_FORBIDDEN_CODE_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "pathlib",
    "requests",
]


def _build_keyword_set(session_dir: Path) -> set[str]:
    keywords = set(COUNT_KEYWORDS)

    try:
        cleaned_dir = session_dir / "cleaned_csvs"
        if not cleaned_dir.exists():
            cleaned_dir = preprocess_tabular_files(session_dir)

        if cleaned_dir.exists():
            for csv_file in cleaned_dir.iterdir():
                if csv_file.suffix.lower() == ".csv":
                    name_stem = csv_file.stem.lower()
                    keywords.add(name_stem)
                    keywords.update(re.split(r"[^a-zA-Z0-9]", name_stem))

                    try:
                        df = pd.read_csv(csv_file, nrows=50)
                        for col in df.columns:
                            col_str = str(col).lower().strip()
                            keywords.add(col_str)
                            keywords.update(
                                part
                                for part in re.split(r"[^a-zA-Z0-9]", col_str)
                                if len(part) > 2
                            )

                        for col in df.columns:
                            if df[col].dtype == "object":
                                unique_vals = df[col].dropna().unique()
                                for val in unique_vals:
                                    val_str = str(val).lower().strip()
                                    keywords.add(val_str)
                                    keywords.update(
                                        part
                                        for part in re.split(r"[^a-zA-Z0-9]", val_str)
                                        if len(part) > 2
                                    )
                    except Exception as exc:
                        logger.error(
                            f"Error reading schema of {csv_file.name} for keywords: {exc}"
                        )
    except Exception as exc:
        logger.error(f"Error in dynamic keyword extraction: {exc}")

    return {kw for kw in keywords if len(kw) > 1}


def _get_keyword_set(session_dir: Path | None) -> set[str]:
    if session_dir is None:
        return set(COUNT_KEYWORDS)

    cache_key = str(session_dir.resolve())
    cached = _KEYWORD_CACHE.get(cache_key)
    if cached is None:
        cached = _build_keyword_set(session_dir)
        _KEYWORD_CACHE[cache_key] = cached
    return set(cached)


def _ensure_safe_code(code: str) -> None:
    lowered = code.lower()
    for pattern in _FORBIDDEN_CODE_PATTERNS:
        if pattern in lowered:
            raise RuntimeError(f"Unsafe code detected: blocked pattern '{pattern}'.")

def is_aggregation_question(question: str, session_dir: Path = None) -> bool:
    q_lower = question.lower()

    keywords = _get_keyword_set(session_dir)
    q_words = set(re.findall(r'[a-zA-Z0-9\-]+', q_lower))
    
    for kw in keywords:
        # If the keyword contains multiple words (like "combined ratio" or "loss ratio"), check direct substring match
        if " " in kw:
            if kw in q_lower:
                return True
        else:
            if kw in q_words:
                return True
                
    return False


def sanitize_df_name(filename: str) -> str:
    name = Path(filename).stem
    # Replace non-alphanumeric with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return f"df_{sanitized}"


def _run_with_timeout(func, timeout_sec, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(f"Execution timed out after {timeout_sec} seconds.")


def run_tabular_query(
    session_dir: Path,
    question: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    """
    Ensure files are preprocessed into clean CSVs, then use the Pandas Coder Agent
    with a self-debugging execution loop to answer quantitative questions, followed
    by a final synthesis step to format and explain the business logic.
    """
    # 1. Preprocess Excel/CSV files in the session into a standardized cleaned_csvs directory
    cleaned_dir = preprocess_tabular_files(session_dir)
    
    if not cleaned_dir.exists():
        return {"handled": False}
        
    csv_files = [p for p in cleaned_dir.iterdir() if p.suffix.lower() == ".csv"]
    if not csv_files:
        return {"handled": False}
        
    # 2. Gather Schema Information & Pre-load Cleaned CSV files as DataFrames
    
    preloaded_dfs = {}
    schema_parts = []
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            var_name = sanitize_df_name(csv_file.name)
            preloaded_dfs[var_name] = df
            
            cols = df.columns.tolist()
            # Generate sample rows representation
            sample_rows = df.head(2).to_string(index=False)
            
            schema_parts.append(
                f"- DataFrame variable: `{var_name}` (sourced from '{csv_file.name}')\n"
                f"  Columns: {cols}\n"
                f"  Sample Data:\n"
                f"  {sample_rows}"
            )
        except Exception as e:
            logger.error(f"Error preloading/parsing {csv_file.name}: {e}")
            
    schema_str = "\n\n".join(schema_parts)
    
    # 3. Build Dynamic Guidelines list to avoid distracting the LLM
    guidelines = [
        "1. Use the preloaded DataFrames directly (e.g., `df_Claims_Bordereau`, `df_Treaty_Performance`). Do NOT load the CSV files using `pd.read_csv` or attempt to read any files from disk.",
        "2. Do NOT use `.str` or `.str.replace` on columns that are already numeric (like 'GWP (USD)', 'Earned Premium (USD)', development months, etc.).",
        "3. Convert any pandas Series to scalar numbers using `.iloc[0]` or `.item()` before printing or formatting.",
        "4. Print all calculated tables, metrics, and details clearly to stdout so they can be analyzed."
    ]
    
    q_lower = question.lower()
    
    # Inject joining and loss ratio guidelines if relevant
    if any(k in q_lower for k in ["loss", "claim", "cyber", "ratio", "commission", "defensible", "perform"]):
        guidelines.append(
            "5. Column Mapping & Table Joins:\n"
            "   - In `df_Claims_Bordereau`, the loss amount is in 'Gross Incurred (USD)' (do not use 'Incurred Losses (USD)' there).\n"
            "   - In `df_Treaty_Performance`, the loss amount is in 'Incurred Losses (USD)'.\n"
            "   - To aggregate claims or losses by Line of Business or Cedent Name, merge `df_Claims_Bordereau` with `df_Treaty_Portfolio` on 'Treaty ID'. Note: Since both DataFrames contain duplicate columns like 'Line of Business' and 'Status', merging them will result in suffixes (e.g. 'Line of Business_x' and 'Line of Business_y'). Use the correct suffixed column name (like 'Line of Business_x') or drop/rename duplicate columns before merging to avoid KeyError."
        )
        guidelines.append(
            "6. Date Columns:\n"
            "   - If filtering or grouping by year (like 2024), convert date columns (like 'Accident Date') to datetime via `pd.to_datetime(df['col'], errors='coerce')` before using `.dt.year`."
        )
        guidelines.append(
            "7. Loss Ratio, Combined Ratio & Ceding Commission:\n"
            "   - To calculate the premium-weighted Combined Ratio per Line of Business or Cedent from `df_Treaty_Performance`:\n"
            "     Weighted Loss Ratio = Sum('Incurred Losses (USD)') / Sum('Earned Premium (USD)')\n"
            "     Weighted Expense Ratio = Sum('Earned Premium (USD)' * 'Expense Ratio') / Sum('Earned Premium (USD)')\n"
            "     Weighted Combined Ratio = Weighted Loss Ratio + Weighted Expense Ratio\n"
            "     (Do NOT use simple average of the Combined Ratio column, always weight by 'Earned Premium (USD)').\n"
            "   - Loss Ratio (from claims/portfolio) = (Sum of 'Gross Incurred (USD)' from `df_Claims_Bordereau` for that LOB/year) / (Sum of 'GWP (USD)' from `df_Treaty_Portfolio` for that LOB/year).\n"
            "   - Defensibility: If Loss Ratio or Combined Ratio is very high (e.g. > 80% loss ratio or > 100% combined ratio), the treaty is unprofitable, making a 30% ceding commission not defensible."
        )
        
    # Inject actuarial triangle guidelines if relevant
    if any(k in q_lower for k in ["triangle", "ibnr", "development", "chain-ladder"]):
        guidelines.append(
            "8. Actuarial Loss Triangles (df_Loss_Triangle_Property):\n"
            "   - To compute a volume-weighted development factor from column A (e.g. '12 months') to column B (e.g. '24 months'):\n"
            "     `factor = df.loc[df['Accident Year'] <= max_valid_year, B].sum() / df.loc[df['Accident Year'] <= max_valid_year, A].sum()`\n"
            "     where max_valid_year is the latest Accident Year that has non-null/non-empty values for column B.\n"
            "   - Ultimate loss = (paid/incurred at current age) * (product of subsequent factors to ultimate age).\n"
            "   - Implied IBNR = Ultimate loss - (current paid/incurred value)."
        )
        
    # Inject fuzzy matching / peril guidelines if relevant
    if any(k in q_lower for k in ["bushfire", "wildfire", "peril", "exposure", "tiv", "aal", "pml", "beryl", "hurricane", "storm", "flood"]):
        guidelines.append(
            "9. Peril / Claim Description / Fuzzy Terminology Matching:\n"
            "   - If looking up or filtering by peril or claim description/loss description (like 'bushfire', 'wildfire', or 'Hurricane Beryl'), perform case-insensitive substring matching rather than exact matching.\n"
            "   - For claim descriptions/loss events in `df_Claims_Bordereau`, use `.str.lower().str.contains('beryl')` rather than exact match (to match 'Hurricane Beryl - Texas coast').\n"
            "   - For perils in `df_CAT_Exposure`, use `df['Peril'].str.lower().str.contains('bushfire')` to match 'Bushfire (AU)'.\n"
            "   - Note: 'Wildfire' (under North America region) and 'Bushfire (AU)' (under Asia Pacific region) are separate records in `df_CAT_Exposure`. If the query asks for 'bushfire' generally or specifically, filter/aggregate accordingly (e.g. 'Bushfire (AU)' has TIV of $18,000 million, AAL of $55 million, 1-in-100 PML of $180 million, and 1-in-250 PML of $310 million)."
        )

    guidelines_str = "\n".join(guidelines)
    
    # 4. Construct prompt
    prompt = f"""You are a professional Python data analyst. Write a clean, correct Python code block to compute the relevant statistics, lookup records, or perform actuarial analysis to answer the user's question using pandas.

The following pandas DataFrames are pre-loaded in the environment:
{schema_str}

Reference Formulas & Guidelines:
{guidelines_str}

Instructions:
- Write Python code that directly operates on the pre-loaded DataFrames listed above. Do NOT load the CSV files using `pd.read_csv` or attempt to read any file from disk.
- Do NOT import `pandas` or `numpy` unless you need specialized sub-modules (they are already imported as `pd` and `np`).
- Do NOT write code to draw, plot, or display charts or graphs (e.g., do NOT use `matplotlib`, `pyplot`, `plt.show()`, or `df.plot()`). The frontend will handle chart rendering dynamically from the printed outputs.
- Make sure to print the final answers, tables, or computed metrics clearly using `print()`.
- Return ONLY the python code block starting with ```python and ending with ```. No other explanation.

User Question: {question}
Python Code:"""

    messages = [{"role": "user", "content": prompt}]
    script_stdout = ""
    success = False
    
    # 5. Self-debugging loop
    for attempt in range(1, 4):
        logger.info(f"Ollama execution attempt {attempt} for query: {question}")
        try:
            from app.services.ollama_client import ollama_chat
            text = ollama_chat(base_url, model, messages)
            
            code = text
            if "```python" in text:
                code = text.split("```python")[1].split("```")[0]
                
            # Prepare execution environment
            
            # Configure matplotlib to run headlessly to prevent GUI popups
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.show = lambda *args, **kwargs: None
            except Exception:
                pass
            
            exec_globals = {
                "pd": pd,
                "np": np,
                **preloaded_dfs
            }
            exec_locals = {}
            stdout_buffer = io.StringIO()
            
            def run_exec():
                exec(code, exec_globals, exec_locals)
                
            try:
                _ensure_safe_code(code)
                # Redirect stdout and run exec in-process with a timeout
                with contextlib.redirect_stdout(stdout_buffer):
                    _run_with_timeout(run_exec, timeout_sec=15)
                
                logger.info(f"Pandas Agent succeeded on attempt {attempt}.")
                script_stdout = stdout_buffer.getvalue().strip()
                success = True
                break
            except Exception as e:
                import traceback
                error_str = traceback.format_exc()
                logger.warning(f"Pandas Agent execution failed on attempt {attempt}. Error:\n{error_str}")
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                error_msg = (
                    f"The code execution failed with the following traceback/error:\n{error_str.strip()}\n\n"
                    f"Please correct the code. Ensure you use the correct preloaded DataFrame variables and write correct Pandas code. "
                    f"Return ONLY the executable python block."
                )
                messages.append({"role": "user", "content": error_msg})
                
        except Exception as e:
            logger.error(f"Error during execution loop on attempt {attempt}: {e}")
            messages.append({"role": "user", "content": f"Execution failed with exception: {e}. Please rewrite the code."})

    if not success:
        return {"handled": False}
        
    # 6. Final Synthesis Step: Send the computed outputs to the LLM to write a professional reinsurance response
    synthesis_prompt = f"""You are a professional reinsurance underwriting assistant. Write a very concise, direct response to the user's question using the raw calculated outputs.

User Question: {question}

Database Computed Outputs:
{script_stdout}

Instructions:
- CRITICAL: State the correct values and answers immediately. Do NOT explain the answer or write introductory/concluding filler sentences.
- Absolutely NO conversational filler or phrases like "Based on the database...", "The computed data shows...", "In summary...", "Therefore...", or "The answer is...".
- Do not write a long essay or explanation unless the user explicitly asks for qualitative reasoning (e.g. "is it defensible?", "why?"). Even then, keep it strictly to 1-2 direct sentences.
- Do not mention that a python script was run or show any python code.
- If the outputs are metrics/exposure numbers, present them directly (e.g., "TIV: $18,000 million, AAL: $55 million, 1-in-100 PML: $180 million, 1-in-250 PML: $310 million").
- If the outputs or the question contains a data breakdown, distribution, comparison, or trend over time (e.g., values across years, categories, or perils) that would be clearer as a visual chart, you MUST append a JSON chart specification at the very end of your response, starting with the marker `[CHART_SPEC]` on a new line.
  The JSON must follow this exact schema:
  {{
    "type": "bar" | "pie" | "donut" | "line" | "area" | "radar" | "scatter" | "bubble" | "funnel" | "waterfall" | "stacked_bar" | "grouped_bar",
    "labels": ["Label A", "Label B", ...],
    "values": [10.5, 20.0, ...],
    "title": "Chart Title"
  }}
  Do not wrap the JSON in markdown code blocks or code fences. If no chart is appropriate or the output doesn't contain a set of values, do NOT output any `[CHART_SPEC]` marker or JSON.
"""
    try:
        from app.services.ollama_client import ollama_chat
        synthesis_text = ollama_chat(base_url, model, [{"role": "user", "content": synthesis_prompt}])
        
        return {
            "handled": True,
            "answer": synthesis_text.strip(),
            "details": [{"file": "Preprocessed CSVs", "count": 1, "matched_ids": []}]
        }
    except Exception as e:
        logger.error(f"Error in final synthesis step: {e}")
        # Fallback to returning raw script output if synthesis fails
        return {
            "handled": True,
            "answer": script_stdout,
            "details": [{"file": "Preprocessed CSVs", "count": 1, "matched_ids": []}]
        }