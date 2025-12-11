"""
Universal Solver - LLM-driven approach to solve ANY quiz task
"""
import re
import json
import logging
import traceback
from typing import Any, Optional, Dict, List
from urllib.parse import urljoin

from browser import render_page, download_file as browser_download
from llm import ask_llm, ask_llm_with_image, transcribe_audio
from submitter import submit_answer
import tools

logger = logging.getLogger(__name__)

# Email for offset calculations
USER_EMAIL = "23f3003663@ds.study.iitm.ac.in"


def solve_quiz(email: str, secret: str, url: str):
    """Main entry point - solve a quiz and handle the chain of questions"""
    current_url = url
    max_questions = 20
    question_count = 0
    
    while current_url and question_count < max_questions:
        question_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {question_count}: {current_url}")
        logger.info('='*60)
        
        try:
            # Step 1: Render the page
            page_data = render_page(current_url)
            
            # Step 2: Solve the question using universal solver
            answer = solve_question_universal(page_data, email)
            
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


def solve_question_universal(page_data: dict, email: str) -> Any:
    """
    Universal question solver - uses LLM to understand and solve ANY task
    """
    content = page_data["content"]
    html = page_data["html"]
    base_url = page_data["base_url"]
    
    logger.info("Analyzing question with universal solver...")
    
    # Step 1: Extract all resources (URLs, files) from the page
    resources = extract_resources(html, content, base_url)
    logger.info(f"Found resources: {list(resources.keys())}")
    
    # Step 2: Determine task type and solve accordingly
    task_analysis = analyze_task(content, resources)
    logger.info(f"Task analysis: {task_analysis['type']}")
    
    # Step 3: Execute the appropriate solution strategy
    try:
        if task_analysis['type'] == 'audio':
            answer = solve_audio(resources, content)
        elif task_analysis['type'] == 'image':
            answer = solve_image(resources, content, base_url)
        elif task_analysis['type'] == 'api_call':
            answer = solve_api_call(resources, content, email)
        elif task_analysis['type'] == 'data_processing':
            answer = solve_data_processing(resources, content, email)
        elif task_analysis['type'] == 'visualization':
            answer = solve_visualization(resources, content)
        else:
            # Default: Let LLM solve with context
            answer = solve_with_llm(resources, content, email)
        
        # Step 4: Clean and validate answer
        answer = clean_and_validate_answer(answer, content)
        logger.info(f"Final answer: {str(answer)[:200]}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Solution error: {e}")
        traceback.print_exc()
        # Fallback to pure LLM
        return solve_with_pure_llm(content, email)


def extract_resources(html: str, content: str, base_url: str) -> Dict:
    """Extract all resources (files, links, APIs) from the page"""
    resources = {}
    
    # File patterns
    file_patterns = {
        'pdf': r'href=["\']([^"\']+\.pdf)["\']',
        'csv': r'href=["\']([^"\']+\.csv)["\']',
        'json': r'href=["\']([^"\']+\.json)["\']',
        'zip': r'href=["\']([^"\']+\.zip)["\']',
        'audio': r'href=["\']([^"\']+\.(?:mp3|wav|opus|ogg|m4a))["\']',
        'image': r'(?:href|src)=["\']([^"\']+\.(?:png|jpg|jpeg|gif))["\']',
        'excel': r'href=["\']([^"\']+\.(?:xlsx|xls))["\']',
    }
    
    for file_type, pattern in file_patterns.items():
        matches = re.findall(pattern, html, re.IGNORECASE)
        if matches:
            resources[file_type] = []
            for match in matches:
                url = match if match.startswith('http') else urljoin(base_url + '/', match.lstrip('/'))
                resources[file_type].append(url)
    
    # Extract API mentions
    api_patterns = [
        r'GET\s+(/[^\s]+)',
        r'POST\s+(/[^\s]+)',
        r'api\.github\.com',
        r'https?://[^\s<>"\']+/api/[^\s<>"\']+',
    ]
    
    for pattern in api_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            resources['has_api'] = True
            break
    
    return resources


def analyze_task(content: str, resources: Dict) -> Dict:
    """Analyze what type of task this is"""
    content_lower = content.lower()
    
    # Keywords for task detection
    if 'audio' in resources or any(kw in content_lower for kw in ['transcribe', 'listen', 'spoken', 'audio']):
        return {'type': 'audio'}
    
    if 'image' in resources and any(kw in content_lower for kw in ['color', 'heatmap', 'chart', 'image', 'picture']):
        return {'type': 'image'}
    
    if resources.get('has_api') or any(kw in content_lower for kw in ['github api', 'api:', 'get /', 'post /']):
        return {'type': 'api_call'}
    
    if 'json' in resources or 'csv' in resources or 'zip' in resources or 'pdf' in resources:
        return {'type': 'data_processing'}
    
    if any(kw in content_lower for kw in ['chart', 'plot', 'graph', 'visualize', 'visualization']):
        return {'type': 'visualization'}
    
    return {'type': 'text'}


def solve_audio(resources: Dict, content: str) -> Any:
    """Solve audio transcription tasks"""
    logger.info("Solving audio task")
    
    audio_urls = resources.get('audio', [])
    if not audio_urls:
        logger.error("No audio URL found")
        return None
    
    audio_url = audio_urls[0]
    logger.info(f"Downloading audio: {audio_url}")
    
    audio_path = browser_download(audio_url)
    transcription = transcribe_audio(audio_path)
    
    logger.info(f"Transcription: {transcription}")
    
    # For passphrase questions, just return the transcription
    if 'passphrase' in content.lower() or 'phrase' in content.lower():
        return transcription.strip().lower()
    
    return transcription


def solve_image(resources: Dict, content: str, base_url: str) -> Any:
    """Solve image-related tasks"""
    logger.info("Solving image task")
    
    image_urls = resources.get('image', [])
    if not image_urls:
        return None
    
    image_url = image_urls[0]
    
    # Color detection
    if 'color' in content.lower() or 'heatmap' in content.lower():
        logger.info("Detecting dominant color")
        image_path = browser_download(image_url)
        return tools.get_dominant_color(image_path)
    
    # Generic image analysis
    return ask_llm_with_image(f"Analyze this image and answer: {content}", image_url=image_url)


def solve_api_call(resources: Dict, content: str, email: str) -> Any:
    """Solve tasks requiring API calls"""
    logger.info("Solving API task")
    
    # Check for GitHub API task
    json_urls = resources.get('json', [])
    if json_urls and 'github' in content.lower():
        # Download the params file
        params_path = browser_download(json_urls[0])
        params = tools.extract_json_content(params_path)
        
        logger.info(f"API params: {params}")
        
        # Build GitHub API call
        owner = params.get('owner', '')
        repo = params.get('repo', '')
        sha = params.get('sha', '')
        path_prefix = params.get('pathPrefix', '')
        
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
        response = tools.call_api(api_url, headers={'Accept': 'application/vnd.github.v3+json'})
        
        if response.get('status') == 200:
            tree = response.get('json', {}).get('tree', [])
            
            # Count files matching pattern
            if '.md' in content.lower():
                count = sum(1 for item in tree if item.get('path', '').startswith(path_prefix) and item.get('path', '').endswith('.md'))
                
                # Add email offset
                if 'mod 2' in content.lower():
                    offset = len(email) % 2
                elif 'mod 5' in content.lower():
                    offset = len(email) % 5
                else:
                    offset = 0
                
                logger.info(f"Count: {count}, Offset: {offset}")
                return count + offset
    
    # Generic API task - let LLM generate the command/answer
    return solve_with_llm(resources, content, email)


def solve_data_processing(resources: Dict, content: str, email: str) -> Any:
    """Solve data processing tasks"""
    logger.info("Solving data processing task")
    
    # PDF Processing
    if 'pdf' in resources:
        pdf_url = resources['pdf'][0]
        pdf_path = browser_download(pdf_url)
        pdf_data = tools.extract_pdf_content(pdf_path)
        
        # Invoice calculation
        if 'invoice' in content.lower() or 'quantity' in content.lower():
            return calculate_invoice_total(pdf_data)
        
        # General PDF question
        return solve_with_context_llm(content, f"PDF Content:\n{pdf_data['text']}\n\nTables: {pdf_data['tables']}", email)
    
    # ZIP Processing (logs)
    if 'zip' in resources:
        zip_url = resources['zip'][0]
        zip_path = browser_download(zip_url)
        zip_data = tools.extract_zip_content(zip_path)
        
        # Sum bytes where event=="download"
        if 'bytes' in content.lower() and 'download' in content.lower():
            return calculate_log_bytes(zip_data, email, content)
        
        return solve_with_context_llm(content, f"ZIP Content:\n{json.dumps(zip_data, indent=2)}", email)
    
    # CSV Processing
    if 'csv' in resources:
        csv_url = resources['csv'][0]
        csv_path = browser_download(csv_url)
        csv_data = tools.extract_csv_content(csv_path)
        
        # Normalization task
        if 'normalize' in content.lower() or 'snake_case' in content.lower():
            return normalize_csv_data(csv_data['data'])
        
        return solve_with_context_llm(content, f"CSV Data:\n{json.dumps(csv_data, indent=2)}", email)
    
    # JSON Processing
    if 'json' in resources:
        json_url = resources['json'][0]
        json_path = browser_download(json_url)
        json_data = tools.extract_json_content(json_path)
        
        return solve_with_context_llm(content, f"JSON Data:\n{json.dumps(json_data, indent=2)}", email)
    
    return solve_with_llm(resources, content, email)


def solve_visualization(resources: Dict, content: str) -> Any:
    """Solve visualization tasks - generate charts as base64"""
    logger.info("Solving visualization task")
    
    # TODO: Implement chart generation based on data
    # For now, fall back to LLM
    return solve_with_pure_llm(content, USER_EMAIL)


def solve_with_llm(resources: Dict, content: str, email: str) -> Any:
    """Let LLM solve with knowledge of available resources"""
    context = f"Available resources: {json.dumps(resources, indent=2)}"
    return solve_with_context_llm(content, context, email)


def solve_with_context_llm(question: str, context: str, email: str) -> Any:
    """Solve using LLM with provided context"""
    prompt = f"""You are solving a data analysis quiz. Analyze the data and answer the question.

QUESTION:
{question}

DATA:
{context}

EMAIL (for offset calculations): {email}
EMAIL LENGTH: {len(email)}

IMPORTANT INSTRUCTIONS:
1. Provide ONLY the answer, no explanations
2. If asked for a number, provide just the number
3. If asked for a command, provide the exact command string
4. If asked for JSON, provide valid JSON
5. If there's an offset calculation (e.g., email length mod X), include it in your answer
6. For dates, use ISO-8601 format (YYYY-MM-DD)
7. Do NOT include markdown formatting

YOUR ANSWER:"""

    answer = ask_llm(prompt)
    return clean_and_validate_answer(answer, question)


def solve_with_pure_llm(content: str, email: str) -> Any:
    """Pure LLM solve when no special processing needed"""
    prompt = f"""You are solving a quiz question. Read carefully and provide ONLY the answer.

QUESTION:
{content}

EMAIL (if needed): {email}
EMAIL LENGTH: {len(email)}

IMPORTANT:
1. Provide ONLY the answer
2. No explanations or formatting
3. For numbers, just the number
4. For commands, the exact command

YOUR ANSWER:"""

    answer = ask_llm(prompt)
    return clean_and_validate_answer(answer, content)


def calculate_invoice_total(pdf_data: Dict) -> float:
    """Calculate sum(Quantity * UnitPrice) from PDF tables"""
    import re
    
    total = 0.0
    
    for table_info in pdf_data.get('tables', []):
        table = table_info.get('data', [])
        if len(table) < 2:
            continue
        
        header = [str(cell).lower().replace(' ', '') if cell else '' for cell in table[0]]
        
        qty_idx = price_idx = None
        for i, col in enumerate(header):
            if 'quantity' in col or 'qty' in col:
                qty_idx = i
            if 'unitprice' in col or 'price' in col:
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


def calculate_log_bytes(zip_data: Dict, email: str, content: str) -> int:
    """Calculate sum of bytes where event==download from log files"""
    import json as json_module
    
    total = 0
    
    for filename, file_content in zip_data.get('files', {}).items():
        for line in file_content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json_module.loads(line)
                if entry.get('event') == 'download' and 'bytes' in entry:
                    total += int(entry['bytes'])
            except:
                pass
    
    # Add offset
    if 'mod 5' in content.lower():
        offset = len(email) % 5
    elif 'mod 2' in content.lower():
        offset = len(email) % 2
    else:
        offset = 0
    
    logger.info(f"Log bytes: {total}, offset: {offset}")
    return total + offset


def normalize_csv_data(data: List[Dict]) -> List[Dict]:
    """Normalize CSV data: snake_case, ISO dates, integers, sorted"""
    return tools.normalize_to_json(data, sort_by='id')


def clean_and_validate_answer(answer: Any, question: str) -> Any:
    """Clean and validate the answer format"""
    if answer is None:
        return ""
    
    # If already non-string, return as-is
    if not isinstance(answer, str):
        return answer
    
    # Remove markdown
    answer = re.sub(r'```\w*\n?', '', answer).strip()
    
    # Remove common prefixes
    for prefix in ['The answer is:', 'Answer:', 'Result:']:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # Try parsing as JSON
    if answer.startswith('{') or answer.startswith('['):
        try:
            return json.loads(answer)
        except:
            pass
    
    # Try parsing as number
    try:
        clean_num = re.sub(r'[^\d.-]', '', answer.split()[0] if answer else '')
        if clean_num:
            if '.' in clean_num:
                return float(clean_num)
            return int(clean_num)
    except:
        pass
    
    # Boolean
    if answer.lower() in ['true', 'yes']:
        return True
    if answer.lower() in ['false', 'no']:
        return False
    
    return answer
