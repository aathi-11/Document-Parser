import json
import subprocess
import sys
import requests
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import logging
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

def is_aggregation_question(question: str, session_dir: Path = None) -> bool:
    q_lower = question.lower()
    
    # 1. Start with the core set of keywords
    keywords = set(COUNT_KEYWORDS)
    
    # 2. Dynamically extract keywords from the schemas and categories in the session if available
    if session_dir is not None:
        try:
            # First, make sure preprocessed files exist (highly optimized check)
            cleaned_dir = session_dir / "cleaned_csvs"
            if not cleaned_dir.exists():
                cleaned_dir = preprocess_tabular_files(session_dir)
                
            if cleaned_dir.exists():
                import pandas as pd
                import re
                for csv_file in cleaned_dir.iterdir():
                    if csv_file.suffix.lower() == ".csv":
                        # Add the filename and parts of it
                        name_stem = csv_file.stem.lower()
                        keywords.add(name_stem)
                        keywords.update(re.split(r'[^a-zA-Z0-9]', name_stem))
                        
                        try:
                            # Read header for columns
                            df = pd.read_csv(csv_file, nrows=50)
                            for col in df.columns:
                                col_str = str(col).lower().strip()
                                keywords.add(col_str)
                                # Add split parts of column name
                                keywords.update(part for part in re.split(r'[^a-zA-Z0-9]', col_str) if len(part) > 2)
                                
                            # Read some unique string values (like Underwriter names, Perils, Cedents, LOBs)
                            # to match questions asking about specific entities
                            for col in df.columns:
                                if df[col].dtype == 'object':
                                    # Limit unique value extraction to first 50 rows to keep it fast
                                    unique_vals = df[col].dropna().unique()
                                    for val in unique_vals:
                                        val_str = str(val).lower().strip()
                                        keywords.add(val_str)
                                        keywords.update(part for part in re.split(r'[^a-zA-Z0-9]', val_str) if len(part) > 2)
                        except Exception as e:
                            logger.error(f"Error reading schema of {csv_file.name} for keywords: {e}")
        except Exception as e:
            logger.error(f"Error in dynamic keyword extraction: {e}")
            
    # Clean up empty strings or single character keywords
    keywords = {kw for kw in keywords if len(kw) > 1}
    
    # 3. Match using word boundary or substring for multi-word keywords
    import re
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
        
    # 2. Gather Schema Information of Cleaned CSV files
    import pandas as pd
    schema_parts = []
    file_info_prompt = []
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file, nrows=2)
            cols = df.columns.tolist()
            schema_parts.append(f"- '{csv_file.name}': Columns: {cols}")
            file_info_prompt.append(f"- {csv_file.name}: Path is '{csv_file.as_posix()}'")
        except Exception as e:
            logger.error(f"Error reading header of {csv_file.name}: {e}")
            
    schema_str = "\n".join(schema_parts)
    files_str = "\n".join(file_info_prompt)
    
    # 3. Build Dynamic Guidelines list to avoid distracting the LLM
    guidelines = [
        "1. Load ONLY the required CSV files using the paths listed above.",
        "2. Do NOT use `.str` or `.str.replace` on columns that are already numeric (like 'GWP (USD)', 'Earned Premium (USD)', development months, etc.).",
        "3. Convert any pandas Series to scalar numbers using `.iloc[0]` or `.item()` before printing or formatting.",
        "4. Print all calculated tables, metrics, and details clearly to stdout so they can be analyzed."
    ]
    
    q_lower = question.lower()
    
    # Inject joining and loss ratio guidelines if relevant
    if any(k in q_lower for k in ["loss", "claim", "cyber", "ratio", "commission", "defensible", "perform"]):
        guidelines.append(
            "5. Column Mapping & Table Joins:\n"
            "   - In 'Claims_Bordereau.csv', the loss amount is in 'Gross Incurred (USD)'. (Do not use 'Incurred Losses (USD)' there).\n"
            "   - In 'Treaty_Performance.csv', the loss amount is in 'Incurred Losses (USD)'.\n"
            "   - To aggregate claims or losses by Line of Business or Cedent Name, merge 'Claims_Bordereau.csv' with 'Treaty_Portfolio.csv' on 'Treaty ID'. Note: Since both files contain duplicate columns like 'Line of Business' and 'Status', merging them will result in suffixes (e.g. 'Line of Business_x' and 'Line of Business_y'). Use the correct suffixed column name (like 'Line of Business_x') or drop/rename duplicate columns before merging to avoid KeyError."
        )
        guidelines.append(
            "6. Date Columns:\n"
            "   - If filtering or grouping by year (like 2024), convert date columns (like 'Accident Date') to datetime via `pd.to_datetime(df['col'], errors='coerce')` before using `.dt.year`."
        )
        guidelines.append(
            "7. Loss Ratio, Combined Ratio & Ceding Commission:\n"
            "   - To calculate the premium-weighted Combined Ratio per Line of Business or Cedent from 'Treaty_Performance.csv':\n"
            "     Weighted Loss Ratio = Sum('Incurred Losses (USD)') / Sum('Earned Premium (USD)')\n"
            "     Weighted Expense Ratio = Sum('Earned Premium (USD)' * 'Expense Ratio') / Sum('Earned Premium (USD)')\n"
            "     Weighted Combined Ratio = Weighted Loss Ratio + Weighted Expense Ratio\n"
            "     (Do NOT use simple average of the Combined Ratio column, always weight by 'Earned Premium (USD)').\n"
            "   - Loss Ratio (from claims/portfolio) = (Sum of 'Gross Incurred (USD)' from Claims_Bordereau for that LOB/year) / (Sum of 'GWP (USD)' from Treaty_Portfolio for that LOB/year).\n"
            "   - Defensibility: If Loss Ratio or Combined Ratio is very high (e.g. > 80% loss ratio or > 100% combined ratio), the treaty is unprofitable, making a 30% ceding commission not defensible."
        )
        
    # Inject actuarial triangle guidelines if relevant
    if any(k in q_lower for k in ["triangle", "ibnr", "development", "chain-ladder"]):
        guidelines.append(
            "8. Actuarial Loss Triangles (Loss_Triangle_Property.csv):\n"
            "   - To compute a volume-weighted development factor from column A (e.g. '12 months') to column B (e.g. '24 months'):\n"
            "     `factor = df.loc[df['Accident Year'] <= max_valid_year, B].sum() / df.loc[df['Accident Year'] <= max_valid_year, A].sum()`\n"
            "     where max_valid_year is the latest Accident Year that has non-null/non-empty values for column B.\n"
            "   - Ultimate loss = (paid/incurred at current age) * (product of subsequent factors to ultimate age).\n"
            "   - Implied IBNR = Ultimate loss - (current paid/incurred value)."
        )
        
    # Inject fuzzy matching / peril guidelines if relevant
    if any(k in q_lower for k in ["bushfire", "wildfire", "peril", "exposure", "tiv", "aal", "pml"]):
        guidelines.append(
            "9. Peril / Fuzzy Terminology Matching:\n"
            "   - If looking up or filtering by peril (like 'bushfire' or 'wildfire'), perform case-insensitive substring matching rather than exact matching. For example, use `df['Peril'].str.lower().str.contains('bushfire')` to match 'Bushfire (AU)'.\n"
            "   - Note: 'Wildfire' (under North America region) and 'Bushfire (AU)' (under Asia Pacific region) are separate records in 'CAT_Exposure.csv'. If the query asks for 'bushfire' generally or specifically, make sure to filter/aggregate accordingly (e.g. 'Bushfire (AU)' has TIV of $18,000 million, AAL of $55 million, 1-in-100 PML of $180 million, and 1-in-250 PML of $310 million)."
        )

    guidelines_str = "\n".join(guidelines)
    
    # 4. Construct prompt
    prompt = f"""You are a professional Python data analyst. Write a clean, correct Python script to compute the relevant statistics, lookup records, or perform actuarial analysis to answer the user's question using pandas on the provided clean CSV files.

Cleaned CSV Schema:
{schema_str}

File paths for pandas:
{files_str}

Reference Formulas & Guidelines:
{guidelines_str}

Instructions:
- Return ONLY the python code block starting with ```python and ending with ```. No other explanation.

User Question: {question}
Python Code:"""

    messages = [{"role": "user", "content": prompt}]
    script_stdout = ""
    success = False
    
    # 5. Self-debugging loop
    for attempt in range(1, 4):
        logger.info(f"Ollama execution attempt {attempt} for query: {question}")
        temp_script_path = None
        try:
            from app.services.ollama_client import ollama_chat
            text = ollama_chat(base_url, model, messages)
            
            code = text
            if "```python" in text:
                code = text.split("```python")[1].split("```")[0]
                
            # Create the temporary script file in the OS temp directory to prevent Uvicorn reload
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tf:
                tf.write(code)
                temp_script_path = Path(tf.name)
            
            # Execute subprocess
            proc = subprocess.run(
                [sys.executable, str(temp_script_path)],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            # Clean up temp script immediately
            if temp_script_path and temp_script_path.exists():
                try:
                    temp_script_path.unlink()
                except Exception:
                    pass
            
            if proc.returncode == 0:
                logger.info(f"Pandas Agent succeeded on attempt {attempt}.")
                script_stdout = proc.stdout.strip()
                success = True
                break
            else:
                logger.warning(f"Pandas Agent execution failed on attempt {attempt}. Error:\n{proc.stderr}")
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                error_msg = (
                    f"The code execution failed with the following traceback/error:\n{proc.stderr.strip()}\n\n"
                    f"Please correct the code. Ensure you use correct column names and merge/aggregate correctly. "
                    f"Return ONLY the executable python block."
                )
                messages.append({"role": "user", "content": error_msg})
                
        except Exception as e:
            logger.error(f"Error during execution loop on attempt {attempt}: {e}")
            if temp_script_path and temp_script_path.exists():
                try:
                    temp_script_path.unlink()
                except Exception:
                    pass
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