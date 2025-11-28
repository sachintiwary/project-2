import os
import logging
import json
import requests
import subprocess
import tempfile
import time
import pandas as pd
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse, unquote

# Third-party
from playwright.sync_api import sync_playwright
import google.generativeai as genai
from google.api_core import exceptions
from markdownify import markdownify as md

# Logging Setup
logger = logging.getLogger("solver")
logger.setLevel(logging.INFO)

class QuizAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Using 1.5 Pro (The smartest available public model)
        self.model_name = "gemini-1.5-pro" 
        self.model = genai.GenerativeModel(self.model_name)
        self.work_dir = tempfile.mkdtemp(prefix="quiz_task_")
        logger.info(f"Workspace: {self.work_dir} | Model: {self.model_name}")

    def generate_with_backoff(self, prompt: str, is_json: bool = True) -> str:
        config = {"response_mime_type": "application/json"} if is_json else None
        retries = 3
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt, generation_config=config)
                return response.text
            except exceptions.ResourceExhausted:
                wait_time = 20
                logger.warning(f"Rate Limit ({self.model_name}). Sleeping {wait_time}s...")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"GenAI Error: {e}")
                raise
        raise Exception("Failed to generate content after retries.")

    def scrape_page(self, url: str) -> str:
        logger.info(f"Scraping: {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=20000)
                content_html = page.content()
                content_md = md(content_html, strip=['img', 'a']) 
                return content_md
            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                return "Error scraping page."
            finally:
                browser.close()

    def parse_task(self, page_content: str, current_url: str) -> Dict[str, Any]:
        parsed = urlparse(current_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"
        
        prompt = f"""
        You are a data extraction agent. 
        CONTEXT:
        - Current Page: {current_url}
        - Base Domain: {base_domain}
        
        INSTRUCTIONS:
        1. Extract the question and URLs.
        2. If a URL is relative (e.g. starts with / or has <span class="origin">), PREPEND the Base Domain.
        3. Do NOT assume localhost.

        PAGE CONTENT:
        {page_content}

        OUTPUT JSON:
        {{
            "question": "The specific analysis question",
            "data_url": "Full absolute URL of data file (or null)",
            "submit_url": "Full absolute URL for submission",
            "answer_key": "JSON key for answer"
        }}
        """
        try:
            text_resp = self.generate_with_backoff(prompt, is_json=True)
            return json.loads(text_resp)
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            raise

    def get_filename_from_cd(self, cd: str) -> Optional[str]:
        if not cd: return None
        import re
        fname = re.findall('filename=(.+)', cd)
        if len(fname) == 0: return None
        return fname[0].strip().strip('"').strip("'")

    def download_file(self, file_url: str, base_url: str) -> Optional[str]:
        if not file_url: return None
        full_url = urljoin(base_url, file_url)
        logger.info(f"Downloading {full_url}")
        try:
            r = requests.get(full_url, verify=False, timeout=15)
            r.raise_for_status()
            
            filename = self.get_filename_from_cd(r.headers.get("Content-Disposition"))
            if not filename: filename = os.path.basename(urlparse(full_url).path)
            
            if not filename or "." not in filename:
                ctype = r.headers.get("Content-Type", "").lower()
                if "html" in ctype: filename = "data.html"
                elif "json" in ctype: filename = "data.json"
                elif "csv" in ctype: filename = "data.csv"
                else: filename = "data_file.dat"

            filename = unquote(filename)
            local_path = os.path.join(self.work_dir, filename)
            
            with open(local_path, "wb") as f: f.write(r.content)
            return local_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def inspect_file_content(self, file_path: str) -> str:
        if not file_path or not os.path.exists(file_path): return "No file available."
        try:
            with open(file_path, 'r', errors='ignore') as f:
                head = f.read(500)
                if "<html" in head.lower(): return f"File Type: HTML Source.\nPreview: {head}..."
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=3)
                return f"CSV Columns: {list(df.columns)}\nRow 1: {df.iloc[0].to_dict()}"
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    preview = list(data.keys()) if isinstance(data, dict) else "List"
                    return f"JSON Structure: {preview}"
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, nrows=3)
                return f"Excel Columns: {list(df.columns)}"
            
            return f"Raw File Start: {head}"
        except Exception as e: return f"Inspection Error: {str(e)}"

    def execute_python_solution(self, question: str, data_path: str) -> Any:
        max_retries = 3
        last_error = None
        file_metadata = self.inspect_file_content(data_path)
        
        context_payload = {
            "task": "Write Python script to solve question.",
            "question": question,
            "file_info": {
                "path": data_path if data_path else None,
                "metadata": file_metadata
            },
            "constraints": [
                "NO Selenium/Webdriver.",
                "NO requests to localhost/127.0.0.1.",
                "IF file_path exists, READ IT directly. Do not download it again.",
                "PRINT ONLY final answer to stdout."
            ]
        }

        for attempt in range(max_retries):
            logger.info(f"Gen Attempt {attempt+1}")
            
            prompt = f"""
            Act as a Python Developer.
            INPUT: {json.dumps(context_payload)}
            
            RULES:
            1. If `file_info.path` is NOT None, your code MUST open that file path. Do NOT try to fetch it from the internet.
            2. If `file_info.path` IS None, and the question mentions a URL, use `requests`.
            3. Return JSON with keys: "thought_process" and "python_code".
            """

            if last_error: prompt += f"\nPREVIOUS ERROR: {last_error}"

            try:
                response_text = self.generate_with_backoff(prompt, is_json=True)
                llm_response = json.loads(response_text)
                code = llm_response.get("python_code", "")
                
                if not code: raise ValueError("Empty code generated")

                path_val = repr(data_path) if data_path else "None"
                injected_code = f"import base64\nfile_path = {path_val}\n" + code

                script_path = os.path.join(self.work_dir, "solve.py")
                with open(script_path, "w") as f: f.write(injected_code)
                
                result = subprocess.run(["python", script_path], capture_output=True, text=True, cwd=self.work_dir, timeout=30)
                
                output = result.stdout.strip()
                error_out = result.stderr.strip()

                if result.returncode != 0:
                    last_error = error_out
                    logger.warning(f"Script Failed: {last_error}")
                    continue
                
                # [CRITICAL FIX] Check if output looks like an error message
                if "error" in output.lower() or "exception" in output.lower() or not output:
                    if not output: output = error_out # If stdout empty, check stderr
                    last_error = f"Script printed an error message instead of answer: {output}"
                    logger.warning(last_error)
                    continue

                logger.info(f"Answer: {output[:100]}...")
                return output

            except Exception as e:
                last_error = str(e)
                logger.error(f"Execution Exception: {e}")
                
        raise Exception("Solution failed after retries.")

    def solve_recursive(self, start_url: str, email: str, secret: str):
        current_url = start_url
        visited = set()
        history = []

        while current_url:
            if current_url in visited: break
            visited.add(current_url)
            try:
                page_md = self.scrape_page(current_url)
                task = self.parse_task(page_md, current_url)
                
                if task.get("submit_url"): 
                    task["submit_url"] = urljoin(current_url, task["submit_url"])
                
                data_path = self.download_file(task.get("data_url"), current_url)
                answer = self.execute_python_solution(task["question"], data_path)
                
                try:
                    if isinstance(answer, str) and len(answer) < 20:
                        clean = answer.replace(',', '')
                        json_answer = float(clean) if '.' in clean else int(clean)
                    else: json_answer = answer
                except: json_answer = answer 

                payload = {"email": email, "secret": secret, "url": current_url, task.get("answer_key", "answer"): json_answer}
                
                logger.info(f"Submitting to {task['submit_url']}")
                resp = requests.post(task['submit_url'], json=payload, verify=False, timeout=10)
                resp_data = resp.json()
                history.append(resp_data)
                
                if resp_data.get("correct", False):
                    next_url = resp_data.get("url")
                    if next_url: current_url = next_url
                    else: return {"status": "completed", "history": history}
                else: return {"status": "failed", "reason": resp_data, "history": history}
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
        return {"status": "completed", "history": history}