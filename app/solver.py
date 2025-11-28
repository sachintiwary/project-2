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
        
        # Using 1.5 Pro for maximum reasoning capability
        self.model_name = "gemini-2.5-pro" 
        self.model = genai.GenerativeModel(self.model_name)
        
        self.work_dir = tempfile.mkdtemp(prefix="quiz_task_")
        logger.info(f"Workspace: {self.work_dir} | Model: {self.model_name}")

    def generate_with_backoff(self, prompt: str, is_json: bool = True) -> str:
        """
        Forces JSON mode to ensure structured output.
        """
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

    def parse_task(self, page_content: str) -> Dict[str, Any]:
        """
        Input: Raw Text
        Output: JSON Structure
        """
        prompt = f"""
        You are a data extraction agent. 
        Analyze the following text and return a JSON object with these exact keys:
        {{
            "question": "The specific analysis question to answer",
            "data_url": "The full URL of the data file mentioned (or null if none)",
            "submit_url": "The full URL to POST the answer to",
            "answer_key": "The JSON key required for the answer (usually 'answer')"
        }}

        PAGE CONTENT:
        {page_content}
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
        
        # 1. Construct the Input Context Object
        # This is the "JSON Prompt" strategy
        context_payload = {
            "task": "Write a Python script to solve the user's question.",
            "user_question": question,
            "environment": {
                "file_path": data_path if data_path else None,
                "file_metadata": file_metadata,
                "working_directory": self.work_dir
            },
            "constraints": [
                "Do NOT use Selenium, Webdriver, or Browsers.",
                "Use 'requests' for HTTP calls if needed.",
                "Use 'pandas', 'json', 'BeautifulSoup', 'pypdf' as needed.",
                "The script must PRINT only the final answer to stdout."
            ]
        }

        for attempt in range(max_retries):
            logger.info(f"Gen Attempt {attempt+1}")
            
            if last_error:
                context_payload["previous_error"] = last_error

            # 2. Send the Prompt
            prompt = f"""
            Act as a Senior Python Developer.
            
            INPUT CONTEXT:
            {json.dumps(context_payload, indent=2)}

            RESPONSE FORMAT (Must be valid JSON):
            {{
                "thought_process": "Analyze the file path status and metadata. Explain your plan.",
                "python_code": "The executable python code string. Must handle imports."
            }}
            """

            # 3. Get JSON Response
            response_text = self.generate_with_backoff(prompt, is_json=True)
            
            try:
                llm_response = json.loads(response_text)
                
                # Extract logic
                reasoning = llm_response.get("thought_process", "No reasoning provided.")
                code = llm_response.get("python_code", "")
                
                logger.info(f"LLM Thoughts: {reasoning}")
                
                if not code:
                    raise ValueError("LLM returned empty code.")

                # Inject Path Variable safety
                path_val = repr(data_path) if data_path else "None"
                injected_code = f"import base64\nfile_path = {path_val}\n" + code

                script_path = os.path.join(self.work_dir, "solve.py")
                with open(script_path, "w") as f: f.write(injected_code)
                
                # Execute
                result = subprocess.run(["python", script_path], capture_output=True, text=True, cwd=self.work_dir, timeout=30)
                
                if result.returncode == 0:
                    ans = result.stdout.strip()
                    if not ans: 
                        last_error = "Script executed successfully but printed NOTHING."
                        continue
                    logger.info(f"Answer: {ans[:100]}...")
                    return ans
                else:
                    last_error = result.stderr
                    logger.warning(f"Script Error: {last_error}")

            except json.JSONDecodeError:
                last_error = "LLM failed to return valid JSON."
            except Exception as e:
                last_error = str(e)
                
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
                task = self.parse_task(page_md)
                
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