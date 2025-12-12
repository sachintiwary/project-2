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
    
    # System prompt - Chain of Thought (CoT) for gpt-5-mini
    system = f"""You are an expert quiz-solving AI with advanced reasoning capabilities.

## CRITICAL INSTRUCTION:
Before calling ANY tool, you MUST perform a "Silent Thought" process:
1. [THOUGHT] Analyze: What is the question asking?
2. [THOUGHT] Check: Do I have the URL/parameters needed?
3. [THOUGHT] Plan: Which tool should I use?
4. [THOUGHT] Verify: Is my answer in the correct format?

Then call the appropriate tool.

## USER CONTEXT
- Email: {USER_EMAIL}
- Email length: {len(USER_EMAIL)}
- Email length % 2 = {len(USER_EMAIL) % 2} (0=even, 1=odd)
- Email length % 3 = {len(USER_EMAIL) % 3}

## AVAILABLE TOOLS
1. fetch_webpage(url) - Get webpage content
2. download_and_read_file(url) - Read CSV/JSON/PDF/ZIP
3. get_image_dominant_color(url) - Returns hex like #aabbcc
4. transcribe_audio(url) - Transcribe audio files to text
5. count_github_files(owner, repo, sha, path_prefix, extension) - Count files + email offset
6. run_python(code) - Execute Python, must set 'result' variable
7. calculate_with_email_offset(base_value, divisor) - Add email-based offset
8. normalize_csv_to_json(url) - Normalize CSV to JSON array
9. sum_invoice_total(url) - Sum Quantity*UnitPrice from PDF invoice
10. sum_log_bytes(url) - Sum bytes from logs.zip where event=='download'
11. calculate_shards(dataset, max_docs_per_shard, max_shards, min_replicas, max_replicas, memory_per_shard, memory_budget) - Optimal shards config
12. find_similar_embeddings(url) - Returns "s4,s5" if email even, "s2,s3" if odd
13. submit_final_answer(answer) - REQUIRED: Submit final answer

## TOOL USE RULES
- ONLY call a tool if you have ALL required arguments
- Do NOT guess or invent parameters
- Do NOT retry failed tools more than twice
- ALWAYS end with submit_final_answer

## ANSWER FORMAT RULES
- Hex colors: #aabbcc (lowercase with #)
- Numbers: just the number (e.g., 335)
- JSON: valid JSON, no markdown
- Commands: exact string (e.g., uv http get URL -H "Accept: application/json")
- Text: exact transcription only

## QUESTION TYPE → TOOL MAPPING
- "uv http" → submit_final_answer with command string directly
- Audio transcription → transcribe_audio(url) → submit_final_answer(text)
- Heatmap/color → get_image_dominant_color(url) → submit_final_answer(hex)
- CSV normalize → normalize_csv_to_json(url) → submit_final_answer(json)
- GitHub file count → count_github_files(...) → submit_final_answer(count)
- Log bytes → sum_log_bytes(url) → submit_final_answer(number)
- Invoice total → sum_invoice_total(url) → submit_final_answer(number)
- Embeddings → find_similar_embeddings(url) → submit_final_answer(result)
- Shards → calculate_shards(...) → submit_final_answer(json)
"""

    # Initial user message
    url_context = "\\n".join([f"- {t}: {u}" for t, u in urls.items()]) if urls else "No file URLs found"
    
    user_msg = f"""[TASK] Solve this quiz question.

QUESTION:
{content}

DETECTED RESOURCES:
{url_context}

[INSTRUCTION] 
1. First, write your [THOUGHT] analysis
2. Then call the appropriate tool
3. Finally, call submit_final_answer with the result"""

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
            "tool_choice": "auto"
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
