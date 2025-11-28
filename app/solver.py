import os
import logging
import json
import requests
import subprocess
import re
import tempfile
import pandas as pd
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse, unquote

# Third-party
from playwright.sync_api import sync_playwright
from openai import OpenAI
from markdownify import markdownify as md

# Logging Setup
logger = logging.getLogger("solver")
logger.setLevel(logging.INFO)

class QuizAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://aipipe.org/openai/v1"
        )
        self.work_dir = tempfile.mkdtemp(prefix="quiz_task_")
        logger.info(f"Workspace created: {self.work_dir}")

    def scrape_page(self, url: str) -> str:
        """Renders JS and converts to Markdown to preserve tables/structure."""
        logger.info(f"Scraping: {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=15000)
                content_html = page.content()
                # Strip images/links to save tokens, keep tables
                content_md = md(content_html, strip=['img', 'a']) 
                return content_md
            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                # Fallback: return raw text if MD fails
                return "Error scraping page content."
            finally:
                browser.close()

    def parse_task(self, page_content: str) -> Dict[str, Any]:
        """Extracts structured instructions from the page."""
        system_prompt = (
            "You are a precise data analyst agent. Extract the following from the quiz page text:\n"
            "1. question: The specific question asked.\n"
            "2. data_url: The URL of any file mentioned (CSV/PDF/etc). If relative, return as is. If none, null.\n"
            "3. submit_url: The URL to POST the answer to.\n"
            "4. answer_key: The JSON key expected for the answer (usually 'answer')."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Page Content:\n{page_content}"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Parsing failed: {e}")
            raise

    def download_file(self, file_url: str, base_url: str) -> Optional[str]:
        if not file_url:
            return None
        
        # Normalize URL
        full_url = urljoin(base_url, file_url)
        
        # Clean Filename (Handle ?query=params)
        parsed_url = urlparse(full_url)
        filename = os.path.basename(parsed_url.path)
        if not filename or "." not in filename:
            filename = "data_file.dat" # Fallback
        
        # Decode URL encoded chars
        filename = unquote(filename)
        local_path = os.path.join(self.work_dir, filename)
        
        logger.info(f"Downloading {full_url} -> {local_path}")
        try:
            r = requests.get(full_url, verify=False, timeout=15)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def inspect_file_content(self, file_path: str) -> str:
        """
        [CRITICAL FIX] 
        Peeks into the file to give the LLM the ACTUAL column names/structure 
        before it writes code. Prevents 'Column not found' errors.
        """
        if not file_path or not os.path.exists(file_path):
            return "No file available."
        
        try:
            # Try CSV
            if file_path.endswith('.csv') or 'csv' in file_path:
                df = pd.read_csv(file_path, nrows=3)
                return f"CSV Columns: {list(df.columns)}\nFirst Row: {df.iloc[0].to_dict()}"
            
            # Try JSON
            if file_path.endswith('.json') or 'json' in file_path:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return f"JSON List. First Item Keys: {list(data[0].keys()) if data else 'Empty'}"
                    return f"JSON Keys: {list(data.keys())}"
            
            # Try Excel
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, nrows=3)
                return f"Excel Columns: {list(df.columns)}"

            # Text/Unknown
            with open(file_path, 'r', errors='ignore') as f:
                return f"File Start: {f.read(500)}"
                
        except Exception as e:
            return f"Error inspecting file: {str(e)}"

    def execute_python_solution(self, question: str, data_path: str) -> Any:
        max_retries = 3
        last_error = None
        
        # [NEW] Get file metadata to prevent guessing
        file_metadata = self.inspect_file_content(data_path)
        logger.info(f"File Metadata extracted: {file_metadata}")

        for attempt in range(max_retries):
            logger.info(f"Code Gen Attempt {attempt+1}")
            
            prompt = (
                f"Goal: {question}\n"
                f"Data File: '{data_path}'\n"
                f"File Structure Preview: {file_metadata}\n"
                "Write a Python script to calculate the answer.\n"
                "CRITICAL RULES:\n"
                "1. Use 'pypdf' for PDFs (NOT PyPDF2).\n"
                "2. Use 'pandas' for CSV/Excel.\n"
                "3. PRINT only the final answer to stdout.\n"
                "4. If column names in the Preview match roughly, use the exact names from Preview.\n"
            )
            
            if last_error:
                prompt += f"\nPREVIOUS ERROR: {last_error}\nFix it."

            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_code = completion.choices[0].message.content
            code_match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            code = code_match.group(1).strip() if code_match else raw_code
            
            script_path = os.path.join(self.work_dir, "solve.py")
            with open(script_path, "w") as f:
                f.write(code)
            
            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    cwd=self.work_dir,
                    timeout=30
                )
                
                if result.returncode == 0:
                    ans = result.stdout.strip()
                    if not ans:
                        last_error = "Script ran but printed nothing. You MUST print(result)."
                        continue
                    logger.info(f"Answer found: {ans}")
                    return ans
                else:
                    last_error = result.stderr
                    logger.warning(f"Script Error: {last_error}")
            except Exception as e:
                last_error = str(e)
                
        raise Exception("Failed to solve after retries.")

    def solve_recursive(self, start_url: str, email: str, secret: str):
        current_url = start_url
        visited = set()
        history = []

        while current_url:
            if current_url in visited:
                break
            visited.add(current_url)
            
            try:
                page_md = self.scrape_page(current_url)
                task = self.parse_task(page_md)
                
                # Normalize Submit URL
                if task.get("submit_url"):
                    task["submit_url"] = urljoin(current_url, task["submit_url"])
                
                data_path = self.download_file(task.get("data_url"), current_url)
                answer = self.execute_python_solution(task["question"], data_path)
                
                # Numeric sanitation
                try:
                    clean_ans = str(answer).replace(',', '')
                    if '.' in clean_ans:
                        json_answer = float(clean_ans)
                    else:
                        json_answer = int(clean_ans)
                except:
                    json_answer = answer 

                payload = {
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    task.get("answer_key", "answer"): json_answer
                }
                
                logger.info(f"Submitting to {task['submit_url']}")
                resp = requests.post(task['submit_url'], json=payload, verify=False, timeout=10)
                resp_data = resp.json()
                
                history.append(resp_data)
                
                if resp_data.get("correct", False):
                    next_url = resp_data.get("url")
                    if next_url:
                        current_url = next_url
                    else:
                        return {"status": "completed", "history": history}
                else:
                    return {"status": "failed", "reason": resp_data, "history": history}

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
        
        return {"status": "completed", "history": history}