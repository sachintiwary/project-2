"""
Agent Module - The Brain
ReAct architecture with strict format enforcement
"""
import json
import logging
import requests
from typing import Any, Dict, Optional

from config import AIPIPE_TOKEN, OPENAI_BASE_URL, LLM_MODEL, MAX_STEPS, USER_EMAIL
from tools import TOOLS, execute_tool
from browser import render_page
from submitter import submit_answer

logger = logging.getLogger(__name__)

# ============================================================
# SYSTEM PROMPT - THE KEY TO SUCCESS
# ============================================================

SYSTEM_PROMPT = """You are a quiz-solving AI agent. Your job is to answer questions correctly.

## WORKFLOW
1. Read the question carefully
2. Call ONE appropriate tool to get data
3. Call submit_answer with the result

## ANSWER FORMAT RULES (CRITICAL!)

### Single Letter Questions (chart, visualization, best option):
- Submit ONLY the letter: A, B, C, or D
- CORRECT: B
- WRONG: {{"answer": "B"}}

### Number Questions:
- Submit ONLY the number
- CORRECT: 335
- CORRECT: 170.97
- WRONG: "The answer is 335"

### Hex Color Questions:
- Submit ONLY the hex code
- CORRECT: #b45a1e
- WRONG: "The dominant color is #b45a1e"

### Text/Transcription Questions:
- Submit ONLY the exact text
- CORRECT: hushed parrot 219
- WRONG: "The passphrase is hushed parrot 219"

### JSON Array Questions (CSV normalization, tool calls):
- Submit ONLY the array, no wrapper
- CORRECT: [{{"id": 1}}, {{"id": 2}}]
- WRONG: {{"data": [...]}}

### Command Questions (uv http):
- Submit the exact command string
- CORRECT: uv http get https://example.com -H "Accept: application/json"
- WRONG: uv http get "https://example.com"  (no quotes around URL)

## TOOL MAPPING
- Audio passphrase → transcribe_audio → submit exact text
- Heatmap/color → get_dominant_color → submit hex code
- CSV normalize → normalize_csv → submit array directly
- GitHub files → count_github_files → submit the number returned
- Invoice total → sum_invoice → submit the number
- Log bytes → sum_log_bytes → submit the number

## CONTEXT
- User email: {email}
- Email length: {email_length}

IMPORTANT: Always submit CLEAN answers, not wrapped in objects!
"""

# ============================================================
# MAIN SOLVER
# ============================================================

def solve_quiz(email: str, secret: str, start_url: str):
    """Main quiz solving loop"""
    current_url = start_url
    question_num = 0
    max_questions = 20
    
    while current_url and question_num < max_questions:
        question_num += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {question_num}: {current_url}")
        logger.info('='*60)
        
        try:
            # Render page
            page_data = render_page(current_url)
            
            # Solve with agent
            answer = solve_question(page_data)
            
            if answer is None:
                logger.error("Agent failed to produce answer")
                break
            
            logger.info(f"Final answer: {answer}")
            
            # Submit
            result = submit_answer(
                submit_url=page_data["submit_url"],
                email=email,
                secret=secret,
                quiz_url=current_url,
                answer=answer
            )
            
            if result.get("correct"):
                logger.info("✓ CORRECT!")
            else:
                logger.warning(f"✗ WRONG: {result.get('reason', 'No reason')}")
            
            # Move to next
            current_url = result.get("url")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            break
    
    logger.info(f"\nCompleted {question_num} questions")


def solve_question(page_data: dict) -> Optional[Any]:
    """Use LLM to solve a single question"""
    
    # Build system prompt
    system = SYSTEM_PROMPT.format(
        email=USER_EMAIL,
        email_length=len(USER_EMAIL)
    )
    
    # Build user message
    user_msg = f"""
## QUESTION
{page_data['content']}

## FILE URLS
{json.dumps(page_data['files'], indent=2)}

## INSTRUCTIONS
1. Analyze what type of question this is
2. Call the appropriate tool with the correct URL
3. Submit the answer in the EXACT format required

Remember: Submit ONLY the answer value, not wrapped in any object!
"""
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ]
    
    # Agent loop
    for step in range(MAX_STEPS):
        logger.info(f"Step {step + 1}")
        
        response = call_llm(messages)
        if not response:
            logger.error("LLM call failed")
            break
        
        message = response.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            # No tools called - check for direct answer in content
            content = message.get("content", "")
            logger.info(f"No tools, content: {content[:200]}")
            return None
        
        # Add assistant message
        messages.append(message)
        
        # Execute tools
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name", "")
            
            if not tool_name:
                logger.warning(f"Tool call has no name: {tool_call}")
                continue
            
            try:
                args = json.loads(func.get("arguments", "{}"))
            except:
                args = {}
            
            logger.info(f"Tool: {tool_name}({json.dumps(args)[:80]})")
            
            # Execute
            result = execute_tool(tool_name, args)
            
            # Check if this is submit_answer
            if tool_name == "submit_answer":
                return result
            
            # Add tool result
            result_str = json.dumps(result) if not isinstance(result, str) else result
            if len(result_str) > 10000:
                result_str = result_str[:10000] + "...[truncated]"
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": result_str
            })
    
    logger.warning("Max steps reached without answer")
    return None


def call_llm(messages: list) -> Optional[dict]:
    """Call LLM with function calling"""
    try:
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {AIPIPE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"LLM error: {response.status_code} - {response.text[:200]}")
            return None
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None
