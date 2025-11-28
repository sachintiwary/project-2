import os
import logging
import json
import requests
import subprocess
import re
import tempfile
from typing import Dict, Any, Optional
from urllib.parse import urljoin

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
                # 20s timeout to fail fast if site is down
                page.goto(url, wait_until="networkidle", timeout=20000)
                
                # Get HTML and convert to Markdown (Preserves Table Structure)
                content_html = page.content()
                content_md = md(content_html, strip=['a', 'img']) 
                return content_md
            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                raise
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
        
        # Handle relative URLs (e.g., "data.csv" -> "https://site.com/data.csv")
        full_url = urljoin(base_url, file_url)
        filename = full_url.split("/")[-1]
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

    def execute_python_solution(self, question: str, data_path: str) -> Any:
        """Generates and runs code in a loop to handle errors."""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            logger.info(f"Code Gen Attempt {attempt+1}")
            
            prompt = (
                f"Question: {question}\n"
                f"Data File Path: '{data_path}' (Verify file exists before reading)\n"
                f"Working Directory: {self.work_dir}\n"
                "Write a Python script to solve this. \n"
                "CRITICAL RULES:\n"
                "1. PRINT ONLY the final answer to stdout using print(). Do NOT print debug info.\n"
                "2. Handle CSV/Excel/PDF parsing using pandas/openpyxl/pypdf.\n"
                "3. If the answer is a string, strip whitespace.\n"
            )
            
            if last_error:
                prompt += f"\nPREVIOUS ERROR: {last_error}\nFIX THE CODE. Did you forget to print the answer?"

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
                    # [FIX FOR EMPTY ANSWER ERROR]
                    if not ans:
                        last_error = "Script executed successfully but printed NOTHING. You MUST print(result)."
                        logger.warning(f"Attempt {attempt+1} failed: Empty Output")
                        continue 
                    
                    logger.info(f"Answer found: {ans}")
                    return ans
                else:
                    last_error = result.stderr
                    logger.warning(f"Script Error: {last_error}")
            except Exception as e:
                last_error = str(e)
                
        raise Exception("Failed to solve question after retries")

    def solve_recursive(self, start_url: str, email: str, secret: str):
        """The main loop handling the chain of quizzes."""
        current_url = start_url
        visited = set()
        history = []

        while current_url:
            if current_url in visited:
                logger.error("Infinite loop detected in quiz URLs")
                break
            
            visited.add(current_url)
            logger.info(f"--- Processing Level: {current_url} ---")
            
            try:
                # 1. Scrape
                page_md = self.scrape_page(current_url)
                
                # 2. Parse Instructions
                task = self.parse_task(page_md)
                
                # [FIX FOR INVALID URL ERROR] Normalize the Submit URL
                submit_url = task.get("submit_url")
                if submit_url:
                    submit_url = urljoin(current_url, submit_url)
                
                # 3. Get Data
                data_path = self.download_file(task.get("data_url"), current_url)
                
                # 4. Solve
                answer = self.execute_python_solution(task["question"], data_path)
                
                # Numeric conversion heuristic
                try:
                    clean_ans = str(answer).replace(',', '')
                    if '.' in clean_ans:
                        json_answer = float(clean_ans)
                    else:
                        json_answer = int(clean_ans)
                except:
                    json_answer = answer 

                # 5. Submit
                payload = {
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    task.get("answer_key", "answer"): json_answer
                }
                
                logger.info(f"Submitting payload to {submit_url}")
                resp = requests.post(submit_url, json=payload, verify=False, timeout=10)
                resp_data = resp.json()
                
                history.append({"url": current_url, "status": resp_data})
                
                # 6. Check Logic for Next Step
                if resp_data.get("correct", False):
                    next_url = resp_data.get("url")
                    if next_url:
                        logger.info(f"Correct! Advancing to {next_url}")
                        current_url = next_url
                    else:
                        logger.info("Quiz Completed Successfully!")
                        return {"status": "completed", "history": history}
                else:
                    logger.error(f"Wrong Answer: {resp_data}")
                    return {"status": "failed", "reason": resp_data, "history": history}

            except Exception as e:
                logger.error(f"Error in loop: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
        
        return {"status": "completed", "history": history}