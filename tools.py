"""
Universal Tools Module - Provides all capabilities the LLM might need
"""
import os
import re
import json
import base64
import logging
import tempfile
import requests
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)


def fetch_url(url: str, headers: Dict = None) -> Dict:
    """Fetch content from any URL (webpage or API)"""
    try:
        logger.info(f"Fetching URL: {url}")
        response = requests.get(url, headers=headers or {}, timeout=30)
        
        content_type = response.headers.get('content-type', '')
        
        if 'application/json' in content_type:
            return {"type": "json", "data": response.json(), "status": response.status_code}
        else:
            return {"type": "text", "data": response.text, "status": response.status_code}
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return {"type": "error", "error": str(e)}


def download_file(url: str) -> str:
    """Download any file and return the local path"""
    try:
        logger.info(f"Downloading: {url}")
        response = requests.get(url, timeout=60)
        
        # Determine extension from URL or content-type
        ext = os.path.splitext(url.split('?')[0])[1] or '.bin'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(response.content)
            logger.info(f"Downloaded to: {f.name}")
            return f.name
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise


def extract_pdf_content(file_path: str) -> Dict:
    """Extract all content from PDF - text and tables"""
    import pdfplumber
    
    result = {"text": "", "tables": [], "pages": 0}
    
    try:
        with pdfplumber.open(file_path) as pdf:
            result["pages"] = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text() or ""
                result["text"] += f"\n--- Page {i+1} ---\n{text}"
                
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    result["tables"].append({
                        "page": i+1,
                        "data": table
                    })
        
        logger.info(f"Extracted {result['pages']} pages, {len(result['tables'])} tables")
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        result["error"] = str(e)
    
    return result


def extract_csv_content(file_path: str) -> Dict:
    """Extract and parse CSV file"""
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, encoding='latin-1')
    
    return {
        "columns": list(df.columns),
        "shape": df.shape,
        "data": df.to_dict(orient='records'),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def extract_json_content(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_zip_content(file_path: str) -> Dict:
    """Extract and read all files from a zip"""
    import zipfile
    
    result = {"files": {}}
    
    with zipfile.ZipFile(file_path, 'r') as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                content = f.read().decode('utf-8', errors='ignore')
                result["files"][name] = content
    
    return result


def run_python_code(code: str, context: Dict = None) -> Dict:
    """Execute Python code safely and return results"""
    import pandas as pd
    import numpy as np
    
    # Create execution context with common libraries
    exec_globals = {
        'pd': pd,
        'np': np,
        'json': json,
        're': re,
        'requests': requests,
        '__builtins__': __builtins__,
    }
    
    if context:
        exec_globals.update(context)
    
    exec_locals = {}
    
    try:
        exec(code, exec_globals, exec_locals)
        
        # Return result variable if defined
        if 'result' in exec_locals:
            return {"success": True, "result": exec_locals['result']}
        else:
            return {"success": True, "locals": {k: str(v)[:1000] for k, v in exec_locals.items()}}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_chart(chart_type: str, data: Any, **kwargs) -> str:
    """Generate a chart and return as base64"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    try:
        if chart_type == 'bar':
            ax.bar(data['x'], data['y'])
        elif chart_type == 'line':
            ax.plot(data['x'], data['y'])
        elif chart_type == 'pie':
            ax.pie(data['values'], labels=data.get('labels'))
        elif chart_type == 'scatter':
            ax.scatter(data['x'], data['y'])
        elif chart_type == 'histogram':
            ax.hist(data['values'], bins=kwargs.get('bins', 10))
        else:
            # Default to bar
            if isinstance(data, dict) and 'x' in data:
                ax.bar(data['x'], data['y'])
        
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        b64_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{b64_image}"
    except Exception as e:
        plt.close(fig)
        logger.error(f"Chart generation error: {e}")
        raise


def call_api(url: str, method: str = 'GET', headers: Dict = None, params: Dict = None, data: Any = None) -> Dict:
    """Make any API call"""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
        else:
            response = requests.request(method, url, headers=headers, params=params, json=data, timeout=30)
        
        try:
            json_data = response.json()
        except:
            json_data = None
        
        return {
            "status": response.status_code,
            "json": json_data,
            "text": response.text[:10000] if not json_data else None
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_image(image_path: str, question: str) -> str:
    """Analyze image using vision API"""
    from llm import ask_llm_with_image
    
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    return ask_llm_with_image(question, image_base64=image_b64)


def get_dominant_color(image_path: str) -> str:
    """Get dominant color from image as hex"""
    from PIL import Image
    from collections import Counter
    
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    pixels = list(img.getdata())
    most_common = Counter(pixels).most_common(1)[0][0]
    
    return '#{:02x}{:02x}{:02x}'.format(*most_common)


def transcribe_audio_file(audio_path: str) -> str:
    """Transcribe audio file"""
    from llm import transcribe_audio
    return transcribe_audio(audio_path)


def geo_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in km"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def normalize_to_json(data: List[Dict], keys_mapping: Dict = None, sort_by: str = None) -> List[Dict]:
    """Normalize data: snake_case keys, ISO dates, sorted"""
    import re
    from datetime import datetime
    import pandas as pd
    
    def to_snake_case(s):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(s))
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower().replace(' ', '_')
    
    result = []
    for item in data:
        new_item = {}
        for key, value in item.items():
            new_key = to_snake_case(key)
            
            # Handle dates
            if 'date' in new_key or new_key == 'joined':
                try:
                    parsed = pd.to_datetime(value)
                    value = parsed.strftime('%Y-%m-%d')
                except:
                    pass
            
            # Handle numeric values
            elif 'value' in new_key or new_key == 'id':
                try:
                    value = int(float(value))
                except:
                    pass
            
            new_item[new_key] = value
        
        result.append(new_item)
    
    if sort_by:
        result = sorted(result, key=lambda x: x.get(sort_by, 0))
    
    return result
