"""
Tools Module - All tools the agent can use
Each tool is a function the agent can call to accomplish tasks
"""
import os
import re
import json
import base64
import logging
import tempfile
import requests
import zipfile
from typing import Any, Dict, List
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# ============================================================
# TOOL DEFINITIONS FOR FUNCTION CALLING
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and return the text content of a webpage or API endpoint",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "headers": {"type": "object", "description": "Optional HTTP headers"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_and_read_file",
            "description": "Download a file and read its contents. Supports CSV, JSON, TXT, PDF, ZIP files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The file URL to download"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_image_dominant_color",
            "description": "Download an image and get its dominant/most common color as a hex code like #aabbcc",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The image URL"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Download and transcribe an audio file to text. Returns the spoken words.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The audio file URL (mp3, wav, opus, etc.)"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "count_github_files",
            "description": "Count files in a GitHub repo matching a path prefix and extension. Returns the count. Use for GitHub tree questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "sha": {"type": "string", "description": "Commit SHA"},
                    "path_prefix": {"type": "string", "description": "Path prefix to filter files (e.g., 'project-1/')"},
                    "extension": {"type": "string", "description": "File extension to count (e.g., '.md')"}
                },
                "required": ["owner", "repo", "sha", "path_prefix", "extension"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code for data analysis. The code should define a 'result' variable with the answer. Libraries available: pandas, numpy, json, re, math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Must set 'result' variable."},
                    "data": {"type": "object", "description": "Optional data to make available as 'data' variable"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_with_email_offset",
            "description": "Calculate a value with email-based offset. offset = (email_length mod divisor)",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_value": {"type": "number", "description": "The base value before offset"},
                    "divisor": {"type": "integer", "description": "The mod divisor (e.g., 2 or 5)"}
                },
                "required": ["base_value", "divisor"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_csv_to_json",
            "description": "Normalize CSV data: convert keys to snake_case, dates to ISO-8601 (YYYY-MM-DD), values to integers, sort by id ascending. Returns JSON array.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the CSV file"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sum_invoice_total",
            "description": "Calculate sum of Quantity * UnitPrice from a PDF invoice. Returns total rounded to 2 decimals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the PDF invoice"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sum_log_bytes",
            "description": "Download a ZIP of log files and sum 'bytes' field where event=='download'",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the logs.zip file"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_answer",
            "description": "Submit the final answer to the quiz. Call this when you have determined the answer. For arrays or objects, pass as JSON string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer (string, number, or JSON-encoded array/object)"}
                },
                "required": ["answer"]
            }
        }
    }
]


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def fetch_webpage(url: str, headers: Dict = None) -> str:
    """Fetch content from a URL"""
    try:
        logger.info(f"Fetching: {url}")
        resp = requests.get(url, headers=headers or {}, timeout=30)
        content_type = resp.headers.get('content-type', '')
        
        if 'json' in content_type:
            return json.dumps(resp.json(), indent=2)
        return resp.text[:50000]  # Limit size
    except Exception as e:
        return f"Error fetching URL: {e}"


def download_and_read_file(url: str) -> str:
    """Download and read file contents"""
    try:
        logger.info(f"Downloading: {url}")
        resp = requests.get(url, timeout=60)
        
        ext = os.path.splitext(url.split('?')[0])[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(resp.content)
            path = f.name
        
        # Handle different file types
        if ext == '.json':
            with open(path) as f:
                return json.dumps(json.load(f), indent=2)
        
        elif ext == '.csv':
            import pandas as pd
            df = pd.read_csv(path)
            return f"CSV with {len(df)} rows, columns: {list(df.columns)}\n\nData:\n{df.to_string()}"
        
        elif ext == '.pdf':
            import pdfplumber
            text = ""
            tables = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    for t in page.extract_tables():
                        tables.append(t)
            return f"PDF Text:\n{text}\n\nTables:\n{json.dumps(tables)}"
        
        elif ext == '.zip':
            content = {}
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        content[name] = f.read().decode('utf-8', errors='ignore')
            return json.dumps(content, indent=2)
        
        else:
            return resp.text[:50000]
            
    except Exception as e:
        return f"Error reading file: {e}"


def get_image_dominant_color(url: str) -> str:
    """Get dominant color from image"""
    try:
        from PIL import Image
        from collections import Counter
        import io
        
        logger.info(f"Getting dominant color: {url}")
        resp = requests.get(url, timeout=30)
        img = Image.open(io.BytesIO(resp.content))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        pixels = list(img.getdata())
        most_common = Counter(pixels).most_common(1)[0][0]
        
        return '#{:02x}{:02x}{:02x}'.format(*most_common)
    except Exception as e:
        return f"Error: {e}"


def transcribe_audio(url: str) -> str:
    """Transcribe audio file using OpenAI client with gpt-4o-transcribe"""
    try:
        from openai import OpenAI
        from config import AIPIPE_TOKEN, OPENAI_BASE_URL
        
        logger.info(f"Transcribing: {url}")
        
        # Download audio
        resp = requests.get(url, timeout=60)
        ext = os.path.splitext(url.split('?')[0])[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(resp.content)
            audio_path = f.name
        
        # Convert to mp3 if needed (gpt-4o-transcribe may need standard format)
        if ext in ['.opus', '.ogg', '.m4a']:
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                mp3_path = audio_path.replace(ext, '.mp3')
                audio.export(mp3_path, format='mp3')
                audio_path = mp3_path
                logger.info(f"Converted to: {mp3_path}")
            except Exception as e:
                logger.warning(f"Audio conversion failed: {e}")
        
        # Use OpenAI client with AI Pipe
        client = OpenAI(
            api_key=AIPIPE_TOKEN,
            base_url=OPENAI_BASE_URL
        )
        
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
        
        logger.info(f"Transcription: {transcript.text}")
        return transcript.text
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Error: {e}"


def count_github_files(owner: str, repo: str, sha: str, path_prefix: str, extension: str) -> int:
    """Count files in GitHub repo matching path prefix and extension, add email offset"""
    from config import USER_EMAIL
    
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
        logger.info(f"GitHub API: {url}")
        
        resp = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'}, timeout=30)
        
        if resp.status_code != 200:
            return f"Error: {resp.status_code} - {resp.text}"
        
        tree = resp.json().get('tree', [])
        
        # Count files matching criteria
        count = 0
        for item in tree:
            path = item.get('path', '')
            if path.startswith(path_prefix) and path.endswith(extension):
                count += 1
                logger.info(f"Found: {path}")
        
        # Add email offset (mod 2 for .md files)
        offset = len(USER_EMAIL) % 2
        result = count + offset
        
        logger.info(f"Count: {count}, Email offset (mod 2): {offset}, Final: {result}")
        return result
        
    except Exception as e:
        return f"Error: {e}"


def run_python(code: str, data: Any = None) -> str:
    """Execute Python code"""
    import pandas as pd
    import numpy as np
    import math
    
    exec_globals = {
        'pd': pd, 'np': np, 'json': json, 're': re, 'math': math,
        'data': data, '__builtins__': __builtins__
    }
    exec_locals = {}
    
    try:
        exec(code, exec_globals, exec_locals)
        if 'result' in exec_locals:
            result = exec_locals['result']
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return result.to_string()
            return json.dumps(result) if not isinstance(result, str) else result
        return "Code executed but no 'result' variable defined"
    except Exception as e:
        return f"Error: {e}"


def calculate_with_email_offset(base_value: float, divisor: int) -> int:
    """Calculate value with email offset"""
    from config import USER_EMAIL
    offset = len(USER_EMAIL) % divisor
    result = int(base_value) + offset
    logger.info(f"Base: {base_value}, divisor: {divisor}, offset: {offset}, result: {result}")
    return result


def normalize_csv_to_json(url: str) -> List[Dict]:
    """Normalize CSV to JSON with proper formatting"""
    import pandas as pd
    
    try:
        logger.info(f"Normalizing CSV: {url}")
        resp = requests.get(url, timeout=30)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as f:
            f.write(resp.content)
            path = f.name
        
        df = pd.read_csv(path)
        
        # Convert column names to snake_case
        def to_snake(s):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(s))
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower().replace(' ', '_')
        
        df.columns = [to_snake(c) for c in df.columns]
        
        result = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                
                if col == 'joined' or 'date' in col:
                    try:
                        record[col] = pd.to_datetime(val).strftime('%Y-%m-%d')
                    except:
                        record[col] = str(val) if pd.notna(val) else None
                elif col == 'value':
                    record[col] = int(float(val)) if pd.notna(val) else 0
                elif col == 'id':
                    record[col] = int(val) if pd.notna(val) else 0
                else:
                    record[col] = str(val).strip() if pd.notna(val) else ""
            result.append(record)
        
        # Sort by id
        result = sorted(result, key=lambda x: x.get('id', 0))
        return result
        
    except Exception as e:
        return f"Error: {e}"


def sum_invoice_total(url: str) -> float:
    """Calculate invoice total from PDF"""
    import pdfplumber
    
    try:
        logger.info(f"Calculating invoice: {url}")
        resp = requests.get(url, timeout=30)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as f:
            f.write(resp.content)
            path = f.name
        
        total = 0.0
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if len(table) < 2:
                        continue
                    
                    header = [str(c).lower().replace(' ', '') if c else '' for c in table[0]]
                    qty_idx = price_idx = None
                    
                    for i, c in enumerate(header):
                        if 'quantity' in c or 'qty' in c:
                            qty_idx = i
                        if 'unitprice' in c or 'price' in c:
                            price_idx = i
                    
                    if qty_idx is not None and price_idx is not None:
                        for row in table[1:]:
                            try:
                                qty = float(re.sub(r'[^\d.]', '', str(row[qty_idx] or '0')))
                                price = float(re.sub(r'[^\d.]', '', str(row[price_idx] or '0')))
                                total += qty * price
                            except:
                                pass
        
        return round(total, 2)
    except Exception as e:
        return f"Error: {e}"


def sum_log_bytes(url: str) -> int:
    """Sum bytes from log files where event==download"""
    try:
        logger.info(f"Summing log bytes: {url}")
        resp = requests.get(url, timeout=30)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip', mode='wb') as f:
            f.write(resp.content)
            path = f.name
        
        total = 0
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    for line in content.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get('event') == 'download' and 'bytes' in entry:
                                total += int(entry['bytes'])
                        except:
                            pass
        
        return total
    except Exception as e:
        return f"Error: {e}"


# Tool dispatcher
TOOL_FUNCTIONS = {
    "fetch_webpage": fetch_webpage,
    "download_and_read_file": download_and_read_file,
    "get_image_dominant_color": get_image_dominant_color,
    "transcribe_audio": transcribe_audio,
    "count_github_files": count_github_files,
    "run_python": run_python,
    "calculate_with_email_offset": calculate_with_email_offset,
    "normalize_csv_to_json": normalize_csv_to_json,
    "sum_invoice_total": sum_invoice_total,
    "sum_log_bytes": sum_log_bytes,
}


def execute_tool(name: str, arguments: Dict) -> Any:
    """Execute a tool by name with arguments"""
    if name == "submit_final_answer":
        # Handle various ways LLM might pass the answer
        answer = arguments.get("answer", "")
        
        # If answer is empty but there are other keys, use the first key or value
        if not answer and arguments:
            # LLM sometimes passes {"/path": "/path"} instead of {"answer": "/path"}
            for key, val in arguments.items():
                if key != "answer":
                    # Use the key if it looks like an answer, otherwise use value
                    answer = key if key and key != val else val
                    break
        
        if not answer:
            return ""
        
        # Try to parse as JSON (for arrays/objects)
        try:
            return json.loads(answer)
        except:
            # Try as number
            try:
                if '.' in str(answer):
                    return float(answer)
                return int(answer)
            except:
                return answer
    
    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"
    
    try:
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as e:
        return f"Tool error: {e}"

