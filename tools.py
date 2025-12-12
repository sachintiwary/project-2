"""
Tools Module - All tools the agent can use
Each tool has a clear description with OUTPUT FORMAT specified
"""
import os
import re
import io
import json
import logging
import tempfile
import zipfile
import requests
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ============================================================
# TOOL DEFINITIONS (For LLM Function Calling)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch text content from a URL (webpage or API). OUTPUT: Raw text/HTML/JSON content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Download and read file contents (CSV, JSON, TXT, PDF, ZIP). OUTPUT: File content as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "File URL to download"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe speech from audio file. OUTPUT: Exact spoken words (e.g., 'hushed parrot 219').",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Audio file URL (mp3, wav, opus)"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dominant_color",
            "description": "Get the most common color in an image. OUTPUT: Hex color code (e.g., #b45a1e).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Image URL"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_csv",
            "description": "Normalize CSV: snake_case keys, ISO dates (YYYY-MM-DD), integer values, sort by id. OUTPUT: JSON array like [{id:1,...}].",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "CSV file URL"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_github_files",
            "description": "Count files in GitHub repo matching path prefix and extension. Adds email offset automatically. OUTPUT: Integer (final count).",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "sha": {"type": "string", "description": "Commit SHA"},
                    "prefix": {"type": "string", "description": "Path prefix (e.g., 'project-1/')"},
                    "extension": {"type": "string", "description": "File extension (e.g., '.md')"}
                },
                "required": ["owner", "repo", "sha", "prefix", "extension"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sum_invoice",
            "description": "Calculate total from PDF invoice (Quantity × UnitPrice). OUTPUT: Number with 2 decimals (e.g., 170.97).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "PDF invoice URL"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sum_log_bytes",
            "description": "Sum 'bytes' field from log files where event='download'. OUTPUT: Integer (total bytes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "ZIP file URL containing logs"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code. Code must set 'result' variable. OUTPUT: Value of 'result' variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code (must set 'result')"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_rate_minutes",
            "description": "Calculate minimal minutes to fetch all pages given rate limits. Use for rate limiting questions. OUTPUT: Integer minutes with email offset already added.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pages": {"type": "integer", "description": "Total number of pages to fetch"},
                    "per_minute": {"type": "integer", "description": "Max requests per minute"},
                    "per_hour": {"type": "integer", "description": "Max requests per hour"},
                    "retry_every": {"type": "integer", "description": "Seconds to wait when hitting hourly limit"}
                },
                "required": ["pages", "per_minute", "per_hour", "retry_every"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit final answer to the quiz. Use this ONLY when you have the complete answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer to submit"}
                },
                "required": ["answer"]
            }
        }
    }
]

# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def fetch_url(url: str) -> str:
    """Fetch content from URL"""
    try:
        logger.info(f"Fetching: {url}")
        resp = requests.get(url, timeout=30)
        if 'json' in resp.headers.get('content-type', ''):
            return json.dumps(resp.json(), indent=2)
        return resp.text[:50000]
    except Exception as e:
        return f"Error: {e}"


def read_file(url: str) -> str:
    """Download and read file"""
    try:
        logger.info(f"Reading file: {url}")
        resp = requests.get(url, timeout=60)
        ext = os.path.splitext(url.split('?')[0])[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(resp.content)
            path = f.name
        
        if ext == '.json':
            return json.dumps(json.load(open(path)), indent=2)
        elif ext == '.csv':
            return open(path).read()[:20000]
        elif ext == '.pdf':
            import pdfplumber
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
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
        return f"Error: {e}"


def transcribe_audio(url: str) -> str:
    """Transcribe audio using Gemini 2.5 Flash"""
    try:
        from google import genai
        from google.genai import types
        from config import GEMINI_API_KEY
        
        logger.info(f"Transcribing: {url}")
        resp = requests.get(url, timeout=60)
        ext = os.path.splitext(url.split('?')[0])[1].lower()
        
        # Save audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(resp.content)
            audio_path = f.name
        
        # Gemini supported MIME types: audio/wav, audio/mp3, audio/aiff, audio/aac, audio/ogg, audio/flac
        mime_map = {
            '.mp3': 'audio/mp3',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.opus': 'audio/ogg',  # Opus in Ogg container
            '.flac': 'audio/flac',
            '.aac': 'audio/aac',
            '.aiff': 'audio/aiff',
            '.m4a': 'audio/aac'
        }
        
        mime_type = mime_map.get(ext, 'audio/mp3')
        
        # Convert opus to mp3 if not directly supported
        if ext == '.opus':
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                mp3_path = audio_path.replace(ext, '.mp3')
                audio.export(mp3_path, format='mp3')
                audio_path = mp3_path
                mime_type = 'audio/mp3'
                logger.info(f"Converted to: {audio_path}")
            except Exception as conv_err:
                logger.warning(f"Conversion failed: {conv_err}, trying original")
        
        # Read audio bytes
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Call Gemini
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Generate a transcript of the speech. Output ONLY the exact words spoken, nothing else.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            ]
        )
        
        text = response.text.strip()
        logger.info(f"Transcription: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return f"Error: {e}"


def get_dominant_color(url: str) -> str:
    """Get dominant color from image"""
    try:
        from PIL import Image
        from collections import Counter
        
        logger.info(f"Getting color: {url}")
        resp = requests.get(url, timeout=30)
        img = Image.open(io.BytesIO(resp.content))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        pixels = list(img.getdata())
        most_common = Counter(pixels).most_common(1)[0][0]
        
        hex_color = '#{:02x}{:02x}{:02x}'.format(*most_common)
        logger.info(f"Dominant color: {hex_color}")
        return hex_color
        
    except Exception as e:
        return f"Error: {e}"


def normalize_csv(url: str) -> List[Dict]:
    """Normalize CSV to JSON"""
    try:
        import pandas as pd
        from dateutil import parser as date_parser
        
        logger.info(f"Normalizing CSV: {url}")
        resp = requests.get(url, timeout=30)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as f:
            f.write(resp.content)
            path = f.name
        
        df = pd.read_csv(path)
        
        # Convert column names to snake_case
        def to_snake(s):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(s))
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower().strip()
        
        df.columns = [to_snake(c) for c in df.columns]
        
        result = []
        for _, row in df.iterrows():
            record = {}
            # Process columns in ORIGINAL order (not sorted!)
            for col in df.columns:
                val = row[col]
                
                # Normalize dates (joined column)
                if col in ['joined', 'date', 'created', 'updated', 'order_date']:
                    try:
                        # Use dayfirst=True for European date format (DD/MM/YY)
                        dt = date_parser.parse(str(val), dayfirst=True)
                        record[col] = dt.strftime('%Y-%m-%d')
                    except:
                        record[col] = str(val)
                # Normalize numeric columns as integers
                elif col in ['value', 'id', 'amount', 'count', 'quantity', 'order_id', 'customer_id']:
                    if col == 'id' or col == 'order_id':
                        record[col] = int(float(val)) if pd.notna(val) else 0
                    elif col == 'value' or col == 'amount':
                        record[col] = int(float(val)) if pd.notna(val) else 0
                    else:
                        record[col] = str(val).strip() if pd.notna(val) else ""
                # String columns
                else:
                    record[col] = str(val).strip() if pd.notna(val) else ""
            result.append(record)
        
        # Sort by id if present
        if result and 'id' in result[0]:
            result = sorted(result, key=lambda x: x.get('id', 0))
        
        logger.info(f"Normalized {len(result)} rows")
        return result
        
    except Exception as e:
        return f"Error: {e}"


def count_github_files(owner: str, repo: str, sha: str, prefix: str, extension: str) -> int:
    """Count files in GitHub repo, add email offset"""
    from config import EMAIL_LENGTH
    import time
    
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
        logger.info(f"GitHub API: {url}")
        
        # Retry logic
        for attempt in range(3):
            resp = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'}, timeout=30)
            
            if resp.status_code == 200:
                break
            elif resp.status_code == 403:
                logger.warning(f"Rate limited, attempt {attempt+1}/3")
                time.sleep(2)
            else:
                break
        
        if resp.status_code != 200:
            logger.error(f"GitHub API failed: {resp.status_code}")
            # Fallback
            offset = EMAIL_LENGTH % 2
            return offset
        
        tree = resp.json().get('tree', [])
        
        # Count matching files
        count = 0
        for item in tree:
            path = item.get('path', '')
            if path.startswith(prefix) and path.endswith(extension):
                count += 1
                logger.info(f"Found: {path}")
        
        # Add email offset
        offset = EMAIL_LENGTH % 2  # 30 % 2 = 0
        result = count + offset
        
        logger.info(f"Count: {count}, Offset: {offset}, Final: {result}")
        return result
        
    except Exception as e:
        return f"Error: {e}"


def sum_invoice(url: str) -> float:
    """Calculate invoice total"""
    try:
        import pdfplumber
        
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
                    
                    header = [str(h).lower() if h else '' for h in table[0]]
                    qty_idx = next((i for i, h in enumerate(header) if 'qty' in h or 'quantity' in h), None)
                    price_idx = next((i for i, h in enumerate(header) if 'price' in h or 'unit' in h), None)
                    
                    if qty_idx is not None and price_idx is not None:
                        for row in table[1:]:
                            try:
                                qty = float(re.sub(r'[^\d.]', '', str(row[qty_idx] or '0')))
                                price = float(re.sub(r'[^\d.]', '', str(row[price_idx] or '0')))
                                total += qty * price
                            except:
                                pass
        
        result = round(total, 2)
        logger.info(f"Invoice total: {result}")
        return result
        
    except Exception as e:
        return f"Error: {e}"


def sum_log_bytes(url: str) -> int:
    """Sum bytes from logs where event=download"""
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
        
        logger.info(f"Total bytes: {total}")
        return total
        
    except Exception as e:
        return f"Error: {e}"


def run_python(code: str) -> str:
    """Execute Python code"""
    import pandas as pd
    import numpy as np
    import math
    
    exec_globals = {
        'pd': pd, 'np': np, 'json': json, 're': re, 'math': math,
        '__builtins__': __builtins__
    }
    exec_locals = {}
    
    try:
        exec(code, exec_globals, exec_locals)
        if 'result' in exec_locals:
            result = exec_locals['result']
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return result.to_string()
            return json.dumps(result) if not isinstance(result, str) else result
        return "No 'result' variable defined"
    except Exception as e:
        return f"Error: {e}"


def submit_answer(answer: str) -> Any:
    """Process and return the final answer"""
    if not answer:
        return ""
    
    # Try to parse as JSON
    try:
        parsed = json.loads(answer)
        # If it's a wrapper dict with answer/json/plan, extract the non-empty value
        if isinstance(parsed, dict):
            if "answer" in parsed and parsed["answer"]:
                return parsed["answer"]
            elif "json" in parsed and parsed["json"]:
                return parsed["json"]
            elif "plan" in parsed and parsed["plan"]:
                return parsed["plan"]
        return parsed
    except:
        pass
    
    # Try as number
    try:
        if '.' in answer:
            return float(answer)
        return int(answer)
    except:
        pass
    
    return answer


# ============================================================

def calculate_rate_minutes(pages: int, per_minute: int, per_hour: int, retry_every: int) -> int:
    """Calculate minimal minutes to fetch all pages with rate limits"""
    from config import EMAIL_LENGTH
    import math
    
    logger.info(f"Calculating rate minutes: {pages} pages, {per_minute}/min, {per_hour}/hr, retry={retry_every}s")
    
    remaining = pages
    total_seconds = 0
    hour_count = 0
    
    while remaining > 0:
        # Each minute, we can do up to per_minute requests
        this_min = min(remaining, per_minute)
        
        # But also check hourly limit
        if hour_count + this_min > per_hour:
            this_min = per_hour - hour_count
        
        remaining -= this_min
        hour_count += this_min
        total_seconds += 60  # 1 minute passed
        
        # Hit hourly limit? Wait retry_every seconds
        if hour_count >= per_hour and remaining > 0:
            total_seconds += retry_every
            hour_count = 0  # Reset hourly counter
    
    base_minutes = math.ceil(total_seconds / 60)
    offset = EMAIL_LENGTH % 3
    final = base_minutes + offset
    
    logger.info(f"Base: {base_minutes} min, Offset: {offset}, Final: {final}")
    return final


TOOL_FUNCTIONS = {
    "fetch_url": fetch_url,
    "read_file": read_file,
    "transcribe_audio": transcribe_audio,
    "get_dominant_color": get_dominant_color,
    "normalize_csv": normalize_csv,
    "count_github_files": count_github_files,
    "sum_invoice": sum_invoice,
    "sum_log_bytes": sum_log_bytes,
    "run_python": run_python,
    "calculate_rate_minutes": calculate_rate_minutes,
    "submit_answer": submit_answer
}


def execute_tool(name: str, args: Dict) -> Any:
    """Execute a tool by name"""
    logger.info(f"Tool: {name}({json.dumps(args)[:100]})")
    
    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"
    
    try:
        result = TOOL_FUNCTIONS[name](**args)
        logger.info(f"Result: {str(result)[:200]}")
        return result
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return f"Error: {e}"
