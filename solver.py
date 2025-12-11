"""
Main solver orchestrator
Coordinates page rendering, task detection, solving, and submission
"""
import re
import json
import logging
import requests
from typing import Any, Optional
from urllib.parse import urljoin

from browser import render_page, download_file
from llm import ask_llm, ask_llm_with_image, transcribe_audio
from submitter import submit_answer

logger = logging.getLogger(__name__)


def solve_quiz(email: str, secret: str, url: str):
    """
    Main entry point - solve a quiz and handle the chain of questions
    """
    current_url = url
    max_questions = 20  # Safety limit
    question_count = 0
    
    while current_url and question_count < max_questions:
        question_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {question_count}: {current_url}")
        logger.info('='*60)
        
        try:
            # Step 1: Render the page
            page_data = render_page(current_url)
            
            # Step 2: Solve the question
            answer = solve_question(page_data)
            
            if answer is None:
                logger.error("Failed to solve question - no answer generated")
                break
            
            # Step 3: Submit the answer
            result = submit_answer(
                submit_url=page_data["submit_url"],
                email=email,
                secret=secret,
                quiz_url=current_url,
                answer=answer
            )
            
            # Step 4: Handle response
            if result.get("correct"):
                logger.info("✓ Answer correct!")
            else:
                logger.warning(f"✗ Answer incorrect: {result.get('reason', 'No reason given')}")
            
            # Check for next URL
            next_url = result.get("url")
            if next_url:
                logger.info(f"Next URL: {next_url}")
                current_url = next_url
            else:
                logger.info("No more URLs - quiz complete!")
                break
                
        except Exception as e:
            logger.error(f"Error solving question: {e}", exc_info=True)
            break
    
    logger.info(f"\nCompleted {question_count} questions")


def solve_question(page_data: dict) -> Any:
    """
    Analyze the question and generate an answer
    """
    content = page_data["content"]
    html = page_data["html"]
    base_url = page_data["base_url"]
    
    logger.info("Analyzing question...")
    
    # Detect task type and solve accordingly
    task_type = detect_task_type(content, html)
    logger.info(f"Detected task type: {task_type}")
    
    if task_type == "audio":
        return solve_audio_task(content, html, base_url)
    elif task_type == "image":
        return solve_image_task(content, html, base_url)
    elif task_type == "file_download":
        return solve_file_task(content, html, base_url)
    elif task_type == "api":
        return solve_api_task(content, html, base_url)
    else:
        # Default: use LLM to solve text-based question
        return solve_text_task(content)


def detect_task_type(content: str, html: str) -> str:
    """Detect the type of task from page content AND html"""
    content_lower = content.lower()
    html_lower = html.lower()
    
    # Check for audio FIRST (priority) - check both content and HTML
    audio_keywords = ['transcribe', 'audio', 'listen', 'spoken', 'passphrase', 'speech']
    if any(kw in content_lower for kw in audio_keywords):
        return "audio"
    if re.search(r'\.(mp3|wav|ogg|m4a|opus)\b', html_lower):
        return "audio"
    
    # Check for image tasks
    image_keywords = ['heatmap', 'dominant color', 'image', 'chart', 'graph', 'picture']
    if any(kw in content_lower for kw in image_keywords):
        return "image"
    if re.search(r'\.(png|jpg|jpeg|gif)\b', content_lower):
        return "image"
    
    # Check for file downloads (log files, zip files too!)
    if re.search(r'\.(pdf|csv|json|xlsx|xls|log|txt|zip)\b', html_lower):
        return "file_download"
    
    # Check for API tasks
    if 'api' in content_lower and ('curl' in content_lower or 'endpoint' in content_lower):
        return "api"
    
    # Check for uv command tasks
    if 'uv' in content_lower and ('command' in content_lower or 'http' in content_lower):
        return "api"
    
    return "text"


def solve_text_task(content: str) -> Any:
    """Solve a text-based task using LLM"""
    logger.info("Solving as text task")
    
    prompt = f"""You are solving a quiz question. Read the following question carefully and provide ONLY the answer.

QUESTION:
{content}

IMPORTANT INSTRUCTIONS:
1. Provide ONLY the answer, no explanations
2. If asked for a command, provide the exact command string
3. If asked for a number, provide just the number
4. If asked for text, provide just the text
5. If asked for JSON, provide valid JSON
6. Do NOT include any markdown formatting like ```
7. Do NOT include phrases like "The answer is..."

YOUR ANSWER:"""

    answer = ask_llm(prompt)
    answer = clean_answer(answer)
    
    logger.info(f"Generated answer: {answer[:200]}...")
    return answer


def solve_file_task(content: str, html: str, base_url: str) -> Any:
    """Solve a task that requires downloading and processing a file"""
    logger.info("Solving as file task")
    
    # Extract file URL - now includes .log files
    file_url = extract_file_url(content, html, base_url)
    if not file_url:
        logger.error("Could not find file URL")
        return solve_text_task(content)
    
    logger.info(f"Found file URL: {file_url}")
    
    # Download the file
    file_path = download_file(file_url)
    
    # Process based on file type
    if file_url.endswith('.json'):
        return process_json_file(file_path, content)
    elif file_url.endswith('.csv'):
        return process_csv_file(file_path, content)
    elif file_url.endswith('.pdf'):
        return process_pdf_file(file_path, content)
    elif file_url.endswith('.log') or file_url.endswith('.txt'):
        return process_log_file(file_path, content)
    elif file_url.endswith('.zip'):
        return process_zip_file(file_path, content)
    else:
        # Read file and let LLM process
        with open(file_path, 'r', errors='ignore') as f:
            file_content = f.read()
        return solve_with_context(content, file_content)


def solve_audio_task(content: str, html: str, base_url: str) -> Any:
    """Solve a task that requires audio transcription"""
    logger.info("Solving as audio task")
    
    # Extract audio URL from HTML - include .opus format
    audio_patterns = [
        r'href=["\']([^"\']*\.(?:mp3|wav|ogg|m4a|opus))["\']',
        r'src=["\']([^"\']*\.(?:mp3|wav|ogg|m4a|opus))["\']',
        r'(https?://[^\s<>"\']+\.(?:mp3|wav|ogg|m4a|opus))',
        r'/([^"\'\s<>]+\.(?:mp3|wav|ogg|m4a|opus))',
    ]
    
    audio_url = None
    for pattern in audio_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            audio_url = match.group(1)
            if not audio_url.startswith('http'):
                audio_url = urljoin(base_url + '/', audio_url.lstrip('/'))
            break
    
    if not audio_url:
        logger.error("Could not find audio URL in HTML")
        # Try to find it in content
        for pattern in audio_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                audio_url = match.group(1)
                if not audio_url.startswith('http'):
                    audio_url = urljoin(base_url + '/', audio_url.lstrip('/'))
                break
    
    if not audio_url:
        logger.error("Could not find audio URL anywhere")
        return solve_text_task(content)
    
    logger.info(f"Found audio URL: {audio_url}")
    
    try:
        # Download and transcribe
        audio_path = download_file(audio_url)
        transcription = transcribe_audio(audio_path)
        
        logger.info(f"Transcription: {transcription}")
        
        # For passphrase questions, just return the transcription
        if 'passphrase' in content.lower() or 'phrase' in content.lower():
            return transcription.strip()
        
        # Use transcription to answer the question
        prompt = f"""Based on this audio transcription, answer the question.

QUESTION:
{content}

AUDIO TRANSCRIPTION:
{transcription}

Provide ONLY the answer (the spoken text/phrase), no explanations."""

        answer = ask_llm(prompt)
        return clean_answer(answer)
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return solve_text_task(content)


def solve_image_task(content: str, html: str, base_url: str) -> Any:
    """Solve a task that requires image analysis"""
    logger.info("Solving as image task")
    
    # Extract image URL
    image_patterns = [
        r'href=["\']([^"\']+\.(?:png|jpg|jpeg|gif))["\']',
        r'src=["\']([^"\']+\.(?:png|jpg|jpeg|gif))["\']',
        r'(https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif))',
    ]
    
    image_url = None
    for pattern in image_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            image_url = match.group(1)
            if not image_url.startswith('http'):
                image_url = urljoin(base_url + '/', image_url.lstrip('/'))
            break
    
    if not image_url:
        logger.error("Could not find image URL")
        return solve_text_task(content)
    
    logger.info(f"Found image URL: {image_url}")
    
    # Check if it's a color-related question (heatmap, dominant color)
    if 'color' in content.lower() or 'heatmap' in content.lower():
        return solve_color_task(image_url, content)
    
    # Use vision API for general image questions
    prompt = f"""Analyze this image carefully and answer the question.

QUESTION:
{content}

IMPORTANT:
- Provide ONLY the answer
- If asked for a color, provide the hex code (e.g., #ff5733)
- If asked for a number, provide just the number
- Be precise and specific

YOUR ANSWER:"""

    answer = ask_llm_with_image(prompt, image_url=image_url)
    return clean_answer(answer)


def solve_color_task(image_url: str, content: str) -> str:
    """Solve a task that requires finding dominant color in an image"""
    logger.info("Solving color/heatmap task")
    
    try:
        from PIL import Image
        from collections import Counter
        import io
        
        # Download image
        response = requests.get(image_url, timeout=30)
        img = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get all pixels
        pixels = list(img.getdata())
        
        # Count colors
        color_counts = Counter(pixels)
        
        # Get most common color
        most_common = color_counts.most_common(1)[0][0]
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(most_common[0], most_common[1], most_common[2])
        
        logger.info(f"Dominant color found: {hex_color}")
        return hex_color
        
    except Exception as e:
        logger.error(f"Color extraction failed: {e}")
        # Fallback to vision API
        prompt = f"""Look at this image and find the dominant/most common color.
        
QUESTION: {content}

Return ONLY the hex color code (e.g., #b45a1e). No explanations."""
        
        answer = ask_llm_with_image(prompt, image_url=image_url)
        return clean_answer(answer)


def solve_api_task(content: str, html: str, base_url: str) -> Any:
    """Solve a task that requires API interaction or command generation"""
    logger.info("Solving as API task")
    
    prompt = f"""Analyze this task and provide the exact answer requested.

TASK:
{content}

INSTRUCTIONS:
- If asked to "craft a command", provide ONLY the exact command string
- If asked for a curl/uv command, provide the complete command
- Replace <your email> with the actual email if mentioned in the task
- Do NOT include explanations

YOUR ANSWER:"""

    response = ask_llm(prompt)
    return clean_answer(response)


def extract_file_url(content: str, html: str, base_url: str) -> Optional[str]:
    """Extract file URL from page content - now includes log and zip files"""
    file_patterns = [
        r'href=["\']([^"\']+\.(?:pdf|csv|json|xlsx|xls|log|txt|zip))["\']',
        r'(https?://[^\s<>"\']+\.(?:pdf|csv|json|xlsx|xls|log|txt|zip))',
    ]
    
    for pattern in file_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            url = match.group(1)
            if not url.startswith('http'):
                url = urljoin(base_url + '/', url.lstrip('/'))
            return url
    
    return None


def process_json_file(file_path: str, question: str) -> Any:
    """Process a JSON file and answer the question"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to string for LLM
    data_str = json.dumps(data, indent=2)
    if len(data_str) > 50000:
        data_str = data_str[:50000] + "...[truncated]"
    
    return solve_with_context(question, data_str)


def process_csv_file(file_path: str, question: str) -> Any:
    """Process a CSV file and answer the question"""
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path)
    except:
        # Try with different encoding
        df = pd.read_csv(file_path, encoding='latin-1')
    
    # Provide full data for better analysis
    full_data = df.to_string()
    
    # Also provide stats
    summary = f"""CSV Data Analysis:
Shape: {df.shape}
Columns: {list(df.columns)}
Data Types: {df.dtypes.to_dict()}

FULL DATA:
{full_data}

Statistics:
{df.describe().to_string()}
"""
    
    if len(summary) > 60000:
        summary = summary[:60000] + "...[truncated]"
    
    return solve_with_context(question, summary)


def process_pdf_file(file_path: str, question: str) -> Any:
    """Process a PDF file and answer the question"""
    import pdfplumber
    
    text_content = ""
    tables_content = ""
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text
            text_content += f"--- Page {i+1} ---\n"
            text_content += page.extract_text() or ""
            text_content += "\n\n"
            
            # Extract tables
            tables = page.extract_tables()
            if tables:
                for j, table in enumerate(tables):
                    tables_content += f"--- Table {j+1} on Page {i+1} ---\n"
                    for row in table:
                        tables_content += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                    tables_content += "\n"
    
    # Combine text and tables
    full_content = f"""PDF TEXT CONTENT:
{text_content}

PDF TABLES:
{tables_content}
"""
    
    if len(full_content) > 60000:
        full_content = full_content[:60000] + "...[truncated]"
    
    # Special handling for invoice/total questions
    if 'total' in question.lower() or 'sum' in question.lower() or 'invoice' in question.lower():
        prompt = f"""Analyze this PDF content and calculate the requested total.

QUESTION:
{question}

PDF CONTENT:
{full_content}

IMPORTANT:
- Calculate the exact total from the line items/values shown
- Return ONLY the number (with 2 decimal places if applicable)
- Do not include currency symbols

YOUR ANSWER:"""
        answer = ask_llm(prompt)
        return clean_answer(answer)
    
    return solve_with_context(question, full_content)


def process_log_file(file_path: str, question: str) -> Any:
    """Process a log file and answer the question"""
    logger.info("Processing log file")
    
    with open(file_path, 'r', errors='ignore') as f:
        log_content = f.read()
    
    # Parse log entries
    log_summary = f"""LOG FILE CONTENT:
{log_content}

LOG ANALYSIS:
- Total lines: {len(log_content.splitlines())}
"""
    
    # Special handling for download bytes / sum questions
    if 'download' in question.lower() or 'bytes' in question.lower() or 'sum' in question.lower():
        prompt = f"""Analyze this log file and calculate what's requested.

QUESTION:
{question}

LOG CONTENT:
{log_content}

IMPORTANT:
- Parse the log entries carefully
- Calculate the exact sum/total requested
- Return ONLY the number
- If the question mentions adding an offset (like email length mod 5), calculate and include it

YOUR ANSWER:"""
        answer = ask_llm(prompt)
        return clean_answer(answer)
    
    return solve_with_context(question, log_summary)


def process_zip_file(file_path: str, question: str) -> Any:
    """Process a zip file containing logs and answer the question"""
    import zipfile
    import json as json_module
    
    logger.info("Processing zip file for logs")
    
    all_content = ""
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                logger.info(f"Found file in zip: {file_name}")
                with zip_ref.open(file_name) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    all_content += f"--- {file_name} ---\n{content}\n\n"
        
        # Special handling for download bytes / sum questions
        if 'download' in question.lower() or 'bytes' in question.lower():
            # Try to parse as JSON lines and calculate sum
            total_bytes = 0
            lines = all_content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('---'):
                    try:
                        entry = json_module.loads(line)
                        if entry.get('event') == 'download' and 'bytes' in entry:
                            total_bytes += int(entry['bytes'])
                    except:
                        pass
            
            # Check if we need to add email offset
            if 'offset' in question.lower() or 'email' in question.lower():
                # Email length mod 5
                email = "23f3003663@ds.study.iitm.ac.in"
                offset = len(email) % 5
                logger.info(f"Email: {email}, length: {len(email)}, offset: {offset}")
                total_bytes += offset
            
            logger.info(f"Calculated total bytes: {total_bytes}")
            return total_bytes
        
        # Fallback to LLM
        return solve_with_context(question, all_content)
        
    except Exception as e:
        logger.error(f"Zip processing error: {e}")
        return solve_with_context(question, f"Error processing zip: {e}")


def solve_with_context(question: str, context: str) -> Any:
    """Solve a question with additional context"""
    prompt = f"""Answer this question based on the provided data.

QUESTION:
{question}

DATA:
{context}

IMPORTANT:
1. Provide ONLY the answer, no explanations
2. For numerical answers, provide just the number (with decimals if needed)
3. For commands, provide the exact command string
4. For text answers, provide just the text
5. For JSON, provide valid JSON
6. Be precise and accurate

YOUR ANSWER:"""

    answer = ask_llm(prompt)
    return clean_answer(answer)


def clean_answer(answer: str) -> Any:
    """Clean and parse the answer into appropriate type"""
    if not answer:
        return ""
    
    # Remove markdown code blocks
    answer = re.sub(r'```\w*\n?', '', answer)
    answer = answer.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "The answer is:",
        "The answer is",
        "Answer:",
        "Answer",
        "Result:",
        "Result",
    ]
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # Try to parse as JSON if it looks like JSON
    if answer.startswith('{') or answer.startswith('['):
        try:
            return json.loads(answer)
        except:
            pass
    
    # Try to parse as number
    try:
        # Remove any trailing text after number
        num_match = re.match(r'^-?[\d,]+\.?\d*', answer.replace(',', ''))
        if num_match:
            num_str = num_match.group()
            if '.' in num_str:
                return float(num_str)
            return int(num_str)
    except:
        pass
    
    # Try to parse as boolean
    if answer.lower() in ['true', 'yes']:
        return True
    if answer.lower() in ['false', 'no']:
        return False
    
    return answer
