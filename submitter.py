"""
Submitter Module - Answer submission logic
CRITICAL: Submits to URL extracted from page, NOT hallucinated!
"""
import logging
import requests

logger = logging.getLogger(__name__)


def submit_answer(submit_url: str, email: str, secret: str, quiz_url: str, answer) -> dict:
    """
    Submit an answer to the quiz
    
    Args:
        submit_url: URL extracted from page content
        email: Student email
        secret: Student secret
        quiz_url: Current question URL
        answer: The answer (any type)
    
    Returns:
        {correct: bool, url: str|None, reason: str|None}
    """
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    logger.info(f"Submitting to: {submit_url}")
    logger.info(f"Quiz URL: {quiz_url}")
    logger.info(f"Answer type: {type(answer).__name__}")
    
    try:
        response = requests.post(
            submit_url,
            json=payload,
            timeout=30
        )
        
        logger.info(f"Status: {response.status_code}")
        
        if response.status_code == 405:
            return {
                "correct": False,
                "reason": f"405 error - wrong URL: {submit_url}",
                "url": None
            }
        
        result = response.json()
        logger.info(f"Response: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Submit error: {e}")
        return {
            "correct": False,
            "reason": str(e),
            "url": None
        }
