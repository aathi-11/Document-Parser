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
    return f"""You are a professional Python developer. Write Python code using openpyxl to create Excel charts.

The following pandas DataFrames are already loaded in the environment:
{schema_str}

These variables are already available (do NOT import them):
- pd, np — pandas and numpy
- openpyxl, BarChart, LineChart, PieChart, Reference, DataPoint
- Font, PatternFill, Alignment, Border, Side, GradientFill
- get_column_letter, dataframe_to_rows

Your task:
1. Create a new workbook: result_wb = openpyxl.Workbook()
2. Remove the default sheet: result_wb.remove(result_wb.active)
3. Create a sheet called "Dashboard": dash_ws = result_wb.create_sheet("Dashboard")
4. Pick the most relevant DataFrame for charting based on the user instruction.
5. Write that DataFrame into a hidden data sheet called "ChartData" using dataframe_to_rows.
6. Build 2 or 3 charts using ONLY BarChart, LineChart, or PieChart from openpyxl.
- For each chart: set chart.title, chart.style = 10, chart.width = 20, chart.height = 12
- Create a Reference for data and a Reference for categories from the ChartData sheet
- Use add_data(ref, titles_from_data=True) and set_categories(cat_ref)
- Place charts on the Dashboard sheet using dash_ws.add_chart(chart, "A1"), dash_ws.add_chart(chart2, "K1"), etc.
7. Set dash_ws.sheet_view.showGridLines = False
8. Store the final workbook in result_wb — this is mandatory.
9. Do NOT call result_wb.save()
10. Do NOT use matplotlib, pyplot, plt, or seaborn.
11. Do NOT import anything.
12. Print "Charts created successfully" to stdout.

Return ONLY a Python code block starting with ```python and ending with ```.

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
        # Always inject Original Data as the first sheet
        original_inserted = False
        uploads_dir = session_dir / "uploads"
        if uploads_dir.exists():
            excel_files = sorted([
                f for f in uploads_dir.iterdir()
                if f.suffix.lower() in {".xlsx", ".xls", ".csv"}
            ])
            if excel_files:
                try:
                    src = excel_files[0]
                    if src.suffix.lower() == ".csv":
                        orig_df = pd.read_csv(src)
                    else:
                        orig_df = pd.read_excel(src, engine="openpyxl")

                    orig_ws = result_wb.create_sheet("Original Data", 0)
                    orig_ws.sheet_properties.tabColor = "1F3864"
                    for row in dataframe_to_rows(orig_df, index=False, header=True):
                        orig_ws.append(row)
                    for cell in orig_ws[1]:
                        cell.font = Font(bold=True, color="FFFFFF", size=11)
                        cell.fill = PatternFill("solid", fgColor="1F3864")
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    orig_ws.freeze_panes = "A2"
                    orig_ws.auto_filter.ref = orig_ws.dimensions
                    original_inserted = True
                except Exception as e:
                    logger.warning(f"Could not insert original data sheet: {e}")

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
            "has_original": original_inserted,
        }

    return {"handled": False}


def get_visualization_file_path(session_dir: Path, filename: str) -> Path | None:
    sanitised_filename = re.sub(r"[^A-Za-z0-9_\-.]", "", Path(filename).name)
    target_path = session_dir / "edited_outputs" / sanitised_filename
    if target_path.exists():
        return target_path
    return None
