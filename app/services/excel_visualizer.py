import contextlib
import io
import logging
import re
import concurrent.futures
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.chart.series import DataPoint
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, GradientFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from app.services.tabular_preprocessor import preprocess_tabular_files
from app.services.tabular_query import (
    _ensure_safe_code,
    _run_with_timeout,
    sanitize_df_name,
)

logger = logging.getLogger(__name__)


def is_visualization_question(question: str) -> bool:
    phrases = [
        "dashboard", "visuali", "chart", "graph", "plot",
        "bar chart", "line chart", "pie chart", "area chart",
        "column chart", "scatter chart", "chart in excel",
        "charts in excel", "excel chart", "excel dashboard",
        "add chart", "create chart", "generate chart",
        "insert chart", "draw chart", "make chart",
        "add graph", "create graph", "visualise", "visualize"
    ]
    lowered = question.lower()
    return any(phrase in lowered for phrase in phrases)


def _build_dashboard_prompt(schema_str: str, question: str) -> str:
    return f"""Tell the LLM it is a professional Python data analyst and Excel dashboard developer. The following pandas DataFrames are preloaded: {schema_str}. The openpyxl library is also available — BarChart, LineChart, PieChart, Reference, Font, PatternFill, Alignment, Border, Side, GradientFill, get_column_letter, dataframe_to_rows are all already imported.

Instructions:
- Write Python code that creates a new openpyxl.Workbook() stored in a variable called exactly result_wb.
- For each meaningful DataFrame, write the data to a sheet using dataframe_to_rows with index=False, header=True.
- Style the header row of each data sheet: bold font, a dark blue fill ("1F3864"), white font color, center-aligned, with all borders.
- Auto-size columns by iterating over them and setting column_dimensions[letter].width based on the max content length (cap at 40).
- Freeze the top row of each data sheet using ws.freeze_panes = "A2".
- Create charts using only openpyxl chart objects — BarChart, LineChart, PieChart — NOT matplotlib or any other library.
- For each chart: create a Reference for values and a Reference for categories, add series to the chart, set chart.title, chart.style = 10, chart.width = 20, chart.height = 12.
- If the user asks for a dashboard, create a separate sheet called "Dashboard" and place all charts on it using ws.add_chart(chart, anchor) with anchors like "A1", "K1", "A22", "K22" for a 2x2 grid layout.
- Store the final workbook in result_wb — this variable is mandatory and must exist after execution.
- Do NOT call result_wb.save() — saving is handled externally.
- Do NOT import openpyxl, pandas, or numpy — they are already available.
- Do NOT use matplotlib, pyplot, or any GUI library.
- Print a summary to stdout: how many sheets created, how many charts added.
- Return ONLY the python code block starting with ```python and ending with ```.

User instruction: {question}
Python Code:"""


def run_excel_visualization(session_dir: Path, question: str, base_url: str, model: str) -> Dict[str, Any]:
    # Part A — Load DataFrames
    cleaned_dir = preprocess_tabular_files(session_dir)
    if not cleaned_dir.exists():
        return {"handled": False}

    csv_files = [p for p in cleaned_dir.iterdir() if p.suffix.lower() == ".csv"]
    if not csv_files:
        return {"handled": False}

    preloaded_dfs = {}
    schema_parts = []

    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
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
        except Exception as e:
            logger.error(f"Error preloading/parsing {csv_file.name}: {e}")

    schema_str = "\n\n".join(schema_parts)

    # Part B — Build prompt and run self-debugging loop
    prompt = _build_dashboard_prompt(schema_str, question)
    messages = [{"role": "user", "content": prompt}]
    success = False
    result_wb = None

    for attempt in range(1, 4):
        logger.info(f"Ollama dashboard/visualization attempt {attempt} for query: {question}")
        try:
            from app.services.ollama_client import ollama_chat
            text = ollama_chat(base_url, model, messages)

            code = text
            if "```python" in text:
                code = text.split("```python")[1].split("```")[0]

            code = code.strip()

            try:
                _ensure_safe_code(code)

                exec_globals = {
                    "pd": pd,
                    "np": np,
                    "openpyxl": openpyxl,
                    "BarChart": BarChart,
                    "LineChart": LineChart,
                    "PieChart": PieChart,
                    "Reference": Reference,
                    "DataPoint": DataPoint,
                    "Font": Font,
                    "PatternFill": PatternFill,
                    "Alignment": Alignment,
                    "Border": Border,
                    "Side": Side,
                    "GradientFill": GradientFill,
                    "get_column_letter": get_column_letter,
                    "dataframe_to_rows": dataframe_to_rows,
                    **preloaded_dfs,
                }
                exec_locals = {}

                # Redirect stdout and execute code with timeout
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    _run_with_timeout(lambda: exec(code, exec_globals, exec_locals), 30)

                # Check both exec_locals and exec_globals for result_wb
                wb = exec_locals.get("result_wb") or exec_globals.get("result_wb")
                if wb is not None and isinstance(wb, openpyxl.Workbook):
                    result_wb = wb
                    success = True
                    break
                else:
                    raise RuntimeError("Variable 'result_wb' of type openpyxl.Workbook was not created or found in globals/locals.")
            except Exception:
                import traceback
                error_str = traceback.format_exc()
                logger.warning(f"Excel visualization execution failed on attempt {attempt}. Error:\n{error_str}")
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                error_msg = (
                    f"The code execution failed with the following traceback/error:\n{error_str.strip()}\n\n"
                    f"Please correct the code. Ensure you create a new openpyxl.Workbook() stored in 'result_wb'. "
                    f"Return ONLY the executable python block starting with ```python and ending with ```."
                )
                messages.append({"role": "user", "content": error_msg})

        except Exception as e:
            logger.error(f"Error during visualization execution loop on attempt {attempt}: {e}")
            messages.append({"role": "user", "content": f"Execution failed with exception: {e}. Please rewrite the code."})

    # Part C — Save the workbook
    if success and result_wb is not None:
        outputs_dir = session_dir / "edited_outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: take first 6 words of question, join with underscores, strip non-alphanumeric, append _dashboard.xlsx
        words = question.split()[:6]
        joined = "_".join(words)
        sanitized = re.sub(r"[^A-Za-z0-9_]", "", joined)
        filename = f"{sanitized.lower()}_dashboard.xlsx"
        output_path = outputs_dir / filename

        result_wb.save(str(output_path))
        sheet_names = result_wb.sheetnames
        total_charts = sum(len(ws._charts) for ws in result_wb.worksheets)

        return {
            "handled": True,
            "filename": output_path.name,
            "sheet_names": sheet_names,
            "total_charts": total_charts,
            "has_dashboard": "Dashboard" in sheet_names,
        }

    return {"handled": False}


def get_visualization_file_path(session_dir: Path, filename: str) -> Path | None:
    sanitised_filename = re.sub(r"[^A-Za-z0-9_\-.]", "", Path(filename).name)
    target_path = session_dir / "edited_outputs" / sanitised_filename
    if target_path.exists():
        return target_path
    return None
