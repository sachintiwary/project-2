import os
import logging
import json
import requests
import subprocess
import tempfile
import time
import base64
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
        # Using a model with good reasoning capabilities
        self.model_name = "gemini-2.5-pro" # Faster/Cheaper, usually sufficient. Use Pro if needed.
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
            except Exception as e:
                logger.warning(f"GenAI Attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        raise Exception("Failed to generate content after retries.")

    def scrape_page(self, url: str) -> str:
        logger.info(f"Scraping: {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                # 60s timeout for heavy JS pages
                page.goto(url, wait_until="networkidle", timeout=60000)
                content_html = page.content()
                # Keep tables and basic structure
                content_md = md(content_html, strip=['img', 'a', 'script', 'style']) 
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
        Extract quiz details from this page content.
        
        CONTEXT:
        - Page URL: {current_url}
        - Base Domain: {base_domain}
        
        INSTRUCTIONS:
        1. Identify the Question.
        2. Identify the Data URL (if any file needs downloading).
        3. Identify the Submit URL (where to POST answer).
        4. Identify the JSON key expected for the answer (usually "answer").
        
        CRITICAL DOMAIN FIX:
        - If a URL is relative (starts with /), prepend {base_domain}.
        - If a URL uses "localhost" or "example.com", REPLACE it with {base_domain} UNLESS it is a real external link.
        
        CONTENT:
        {page_content[:15000]} 

        OUTPUT JSON:
        {{
            "question": "...",
            "data_url": "...",
            "submit_url": "...",
            "answer_key": "answer" 
        }}
        """
        try:
            text_resp = self.generate_with_backoff(prompt, is_json=True)
            data = json.loads(text_resp)
            
            # Python-side Sanitization
            for key in ["data_url", "submit_url"]:
                val = data.get(key)
                if val:
                    # Fix relative paths
                    if val.startswith("/"):
                        data[key] = urljoin(base_domain, val)
                    # Fix placeholder domains
                    elif "example.com" in val or "localhost" in val:
                        p_val = urlparse(val)
                        new_val = urljoin(base_domain, p_val.path)
                        if p_val.query: new_val += "?" + p_val.query
                        data[key] = new_val
                        
            return data
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            raise

    def download_file(self, file_url: str) -> Optional[str]:
        if not file_url: return None
        logger.info(f"Downloading {file_url}")
        try:
            r = requests.get(file_url, verify=False, timeout=15)
            r.raise_for_status()
            
            # Simple filename deduction
            filename = os.path.basename(urlparse(file_url).path)
            if not filename or "." not in filename: filename = "data.file"
            
            local_path = os.path.join(self.work_dir, unquote(filename))
            with open(local_path, "wb") as f: f.write(r.content)
            return local_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def execute_python_solution(self, question: str, data_path: str) -> Any:
        max_retries = 3
        last_error = None
        
        # Gather context
        file_info = ""
        if data_path and os.path.exists(data_path):
            file_info = f"File available at: '{data_path}'."
            if data_path.endswith(".csv"):
                try:
                    df = pd.read_csv(data_path, nrows=2)
                    file_info += f"\nColumns: {list(df.columns)}"
                except: pass

        for attempt in range(max_retries):
            prompt = f"""
            Write a Python script to solve this question.
            QUESTION: {question}
            CONTEXT: {file_info}
            
            ENVIRONMENT:
            - Python 3.10+
            - Libraries: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pypdf
            
            REQUIREMENTS:
            1. Calculate the answer.
            2. If the answer is a NUMBER/STRING: Print it directly to stdout.
            3. If the answer is an IMAGE/PLOT: Save it to 'plot.png', convert 'plot.png' to a Base64 Data URI string, and print the string.
            4. Handle exceptions gracefully.
            5. Return JSON: {{"code": "python code here"}}
            """
            
            if last_error:
                prompt += f"\nPREVIOUS ERROR: {last_error}\nFix the code."

            try:
                resp = self.generate_with_backoff(prompt, is_json=True)
                code = json.loads(resp).get("code", "")
                
                # Write and Run
                script_path = os.path.join(self.work_dir, "solve.py")
                with open(script_path, "w") as f: f.write(code)
                
                result = subprocess.run(
                    ["python", script_path], 
                    capture_output=True, text=True, cwd=self.work_dir, timeout=40
                )
                
                output = result.stdout.strip()
                if result.returncode != 0:
                    last_error = result.stderr
                    continue
                    
                if not output:
                    last_error = "Script ran but printed nothing."
                    continue
                    
                return output # Success

            except Exception as e:
                last_error = str(e)
        
        return None

    def solve_recursive(self, start_url: str, email: str, secret: str):
        current_url = start_url
        history = []
        
        # Limit recursion depth to prevent infinite loops
        for _ in range(5):
            if not current_url: break
            
            logger.info(f"Processing Task: {current_url}")
            try:
                # 1. Scrape & Parse
                page_md = self.scrape_page(current_url)
                task = self.parse_task(page_md, current_url)
                
                # 2. Get Data
                data_path = self.download_file(task.get("data_url"))
                
                # 3. Solve
                answer = self.execute_python_solution(task["question"], data_path)
                
                # 4. Format Answer
                # Try to convert to number if it looks like one, otherwise keep string
                final_answer = answer
                try:
                    if answer.replace('.','',1).isdigit():
                        final_answer = float(answer) if '.' in answer else int(answer)
                except: pass

                # 5. Submit
                submit_url = task.get("submit_url")
                payload = {
                    "email": email, 
                    "secret": secret, 
                    "url": current_url, 
                    task.get("answer_key", "answer"): final_answer
                }
                
                logger.info(f"Submitting to {submit_url}")
                resp = requests.post(submit_url, json=payload, verify=False, timeout=30)
                
                if resp.status_code == 200:
                    resp_data = resp.json()
                    history.append(resp_data)
                    logger.info(f"Result: {resp_data}")
                    
                    if resp_data.get("correct", False):
                        next_url = resp_data.get("url")
                        if next_url:
                            current_url = next_url # Recursion step
                        else:
                            return # Done!
                    else:
                        logger.warning("Incorrect answer. Retrying not implemented in this snippet to save recursion depth.")
                        return 
                else:
                    logger.error(f"Submission failed: {resp.status_code} {resp.text}")
                    return

            except Exception as e:
                logger.error(f"Error in loop: {e}")
                return