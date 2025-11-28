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
        logger.info(f"Scraping: {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=15000)
                content_html = page.content()
                content_md = md(content_html, strip=['img', 'a']) 
                return content_md
            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                return "Error scraping page content."
            finally:
                browser.close()

    def parse_task(self, page_content: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a precise data analyst agent. Extract the following from the quiz page text:\n"
            "1. question: The specific question asked.\n"
            "2. data_url: The URL of any file mentioned. Return null if none.\n"
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

    def get_filename_from_cd(self, cd: str) -> Optional[str]:
        """Extract filename from Content-Disposition header"""
        if not cd:
            return None
        fname = re.findall('filename=(.+)', cd)
        if len(fname) == 0:
            return None
        return fname[0].strip().strip('"').strip("'")

    def download_file(self, file_url: str, base_url: str) -> Optional[str]:
        if not file_url:
            return None
        
        full_url = urljoin(base_url, file_url)
        logger.info(f"Downloading {full_url}")
        
        try:
            r = requests.get(full_url, verify=False, timeout=15)
            r.raise_for_status()
            
            # [FIX] Smart Filename Detection
            filename = self.get_filename_from_cd(r.headers.get("Content-Disposition"))
            if not filename:
                filename = os.path.basename(urlparse(full_url).path)
            
            # [FIX] Handle URL parameters in filename or empty filename
            if not filename or "." not in filename:
                # Check Content-Type for hint
                ctype = r.headers.get("Content-Type", "").lower()
                if "html" in ctype:
                    filename = "data.html"
                elif "json" in ctype:
                    filename = "data.json"
                elif "csv" in ctype:
                    filename = "data.csv"
                else:
                    filename = "data_file.dat"

            filename = unquote(filename)
            local_path = os.path.join(self.work_dir, filename)
            
            with open(local_path, "wb") as f:
                f.write(r.content)
            
            return local_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def inspect_file_content(self, file_path: str) -> str:
        if not file_path or not os.path.exists(file_path):
            return "No file available."
        
        try:
            # Check content for HTML even if extension is wrong
            with open(file_path, 'r', errors='ignore') as f:
                head = f.read(500)
                if "<html" in head.lower() or "<!doctype html" in head.lower() or "<div" in head.lower():
                    return f"File Type: HTML Source Code.\nPreview: {head}..."

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=3)
                return f"CSV Columns: {list(df.columns)}\nRow 1: {df.iloc[0].to_dict()}"
            
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return f"JSON Keys: {list(data.keys()) if isinstance(data, dict) else 'List of items'}"
            
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, nrows=3)
                return f"Excel Columns: {list(df.columns)}"

            return f"Raw File Start: {head}"
        except Exception as e:
            return f"Inspection Error: {str(e)}"

    def execute_python_solution(self, question: str, data_path: str) -> Any:
        max_retries = 3
        last_error = None
        file_metadata = self.inspect_file_content(data_path)
        
        logger.info(f"Metadata: {file_metadata}")

        for attempt in range(max_retries):
            logger.info(f"Gen Attempt {attempt+1}")
            
            prompt = (
                f"Goal: {question}\n"
                f"File Metadata: {file_metadata}\n"
                "Write a Python script to solve this.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. The variable 'file_path' is already defined for you. USE IT.\n"
                "2. If metadata says HTML, use BeautifulSoup.\n"
                "3. If CSV/Excel, use pandas.\n"
                "4. PRINT ONLY the final answer.\n"
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
            
            # [FIX] INJECT PATH directly into the script
            # This handles Windows/Linux path escaping automatically via python repr
            injected_code = f"file_path = {repr(data_path)}\n" + code

            script_path = os.path.join(self.work_dir, "solve.py")
            with open(script_path, "w") as f:
                f.write(injected_code)
            
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
                        last_error = "Script printed nothing. You must print(result)."
                        continue
                    logger.info(f"Answer: {ans}")
                    return ans
                else:
                    last_error = result.stderr
                    logger.warning(f"Error: {last_error}")
            except Exception as e:
                last_error = str(e)
                
        raise Exception("Solution failed after retries.")

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
                
                if task.get("submit_url"):
                    task["submit_url"] = urljoin(current_url, task["submit_url"])
                
                data_path = self.download_file(task.get("data_url"), current_url)
                
                # Execute Logic
                answer = self.execute_python_solution(task["question"], data_path)
                
                # Type sanitation
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