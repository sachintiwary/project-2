"""
Agent Module - Core ReAct agent with tool calling
Uses gpt-4o-mini with function calling for cost efficiency
"""
import json
import logging
import requests
from typing import Any, Dict, Optional

from config import AIPIPE_TOKEN, OPENAI_BASE_URL, LLM_MODEL, MAX_AGENT_STEPS, USER_EMAIL
from tools import TOOLS, execute_tool
from browser import render_page
from submitter import submit_answer

logger = logging.getLogger(__name__)


def solve_quiz(email: str, secret: str, url: str):
    """Main entry - solve quiz using agent"""
    current_url = url
    max_questions = 20
    question_count = 0
    
    while current_url and question_count < max_questions:
        question_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {question_count}: {current_url}")
        logger.info('='*60)
        
        try:
            # Render page
            page_data = render_page(current_url)
            
            # Use agent to solve
            answer = agent_solve(page_data)
            
            if answer is None:
                logger.error("Agent failed to produce answer")
                break
            
            # Submit
            result = submit_answer(
                submit_url=page_data["submit_url"],
                email=email,
                secret=secret,
                quiz_url=current_url,
                answer=answer
            )
            
            if result.get("correct"):
                logger.info("✓ Answer correct!")
            else:
                logger.warning(f"✗ Answer incorrect: {result.get('reason', 'No reason')}")
            
            next_url = result.get("url")
            if next_url:
                current_url = next_url
            else:
                logger.info("Quiz complete!")
                break
                
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            break
    
    logger.info(f"\nCompleted {question_count} questions")


def agent_solve(page_data: dict) -> Any:
    """
    Agent solves the question using ReAct loop with tool calling
    """
    content = page_data["content"]
    html = page_data["html"]
    base_url = page_data["base_url"]
    
    # Extract URLs from the page for context
    urls = extract_urls_from_page(html, base_url)
    
    # System prompt
    system = f"""You are an AI agent that solves quiz questions by using tools.

AVAILABLE TOOLS:
- fetch_webpage: Get content from any URL
- download_and_read_file: Download and read CSV/JSON/PDF/ZIP files
- get_image_dominant_color: Get hex color from image
- transcribe_audio: Transcribe audio files (mp3, wav, opus)
- call_github_api: Call GitHub API for repository data
- run_python: Execute Python code for calculations (set 'result' variable)
- calculate_with_email_offset: Add email-based offset to a number
- normalize_csv_to_json: Convert CSV to normalized JSON array
- sum_invoice_total: Calculate sum(Quantity * UnitPrice) from PDF
- sum_log_bytes: Sum bytes from log ZIP where event=='download'
- submit_final_answer: Submit your final answer

IMPORTANT:
- User email: {USER_EMAIL} (length: {len(USER_EMAIL)})
- For offset calculations: use calculate_with_email_offset
- Always call submit_final_answer when you have the answer
- Be concise - call the right tool immediately
- For audio transcription: just return what was said, no explanation
- For colors: return the hex code like #aabbcc
- For JSON normalization: snake_case keys, ISO dates (YYYY-MM-DD), integers for values, sort by id
"""

    # Initial user message
    url_context = "\n".join([f"- {t}: {u}" for t, u in urls.items()]) if urls else "No file URLs found"
    
    user_msg = f"""QUIZ QUESTION:
{content}

DETECTED URLS ON PAGE:
{url_context}

Analyze the question and use tools to solve it. When you have the answer, call submit_final_answer."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ]
    
    # Agent loop
    for step in range(MAX_AGENT_STEPS):
        logger.info(f"Agent step {step + 1}")
        
        response = call_llm_with_tools(messages)
        
        if not response:
            logger.error("LLM call failed")
            break
        
        message = response.get("choices", [{}])[0].get("message", {})
        
        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            # No tools called - might have direct answer
            content = message.get("content", "")
            logger.info(f"Agent response (no tools): {content[:200]}")
            return content
        
        # Add assistant message with tool calls
        messages.append(message)
        
        # Execute each tool
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except:
                args = {}
            
            logger.info(f"Tool: {tool_name}({json.dumps(args)[:100]})")
            
            # Execute tool
            result = execute_tool(tool_name, args)
            
            # Check if this is the final answer
            if tool_name == "submit_final_answer":
                logger.info(f"Final answer: {result}")
                return result
            
            # Add tool result to messages
            result_str = json.dumps(result) if not isinstance(result, str) else result
            if len(result_str) > 10000:
                result_str = result_str[:10000] + "...[truncated]"
            
            logger.info(f"Tool result: {result_str[:200]}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result_str
            })
    
    logger.warning("Agent reached max steps without answer")
    return None


def call_llm_with_tools(messages: list) -> Optional[dict]:
    """Call LLM with function calling support"""
    try:
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0
        }
        
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"LLM error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def extract_urls_from_page(html: str, base_url: str) -> Dict[str, str]:
    """Extract file URLs from page HTML"""
    import re
    from urllib.parse import urljoin
    
    urls = {}
    
    patterns = {
        'csv': r'href=["\']([^"\']+\.csv)["\']',
        'json': r'href=["\']([^"\']+\.json)["\']',
        'pdf': r'href=["\']([^"\']+\.pdf)["\']',
        'zip': r'href=["\']([^"\']+\.zip)["\']',
        'audio': r'href=["\']([^"\']+\.(?:mp3|wav|opus|ogg|m4a))["\']',
        'image': r'(?:href|src)=["\']([^"\']+\.(?:png|jpg|jpeg|gif))["\']',
    }
    
    for file_type, pattern in patterns.items():
        matches = re.findall(pattern, html, re.IGNORECASE)
        for match in matches:
            url = match if match.startswith('http') else urljoin(base_url + '/', match.lstrip('/'))
            urls[file_type] = url
    
    return urls
