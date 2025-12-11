"""
Answer submission logic
CRITICAL: Submits to the URL extracted from the page, NOT a hallucinated URL
"""
import logging
import requests

logger = logging.getLogger(__name__)


def submit_answer(submit_url: str, email: str, secret: str, 
                  quiz_url: str, answer) -> dict:
    """
    Submit an answer to the quiz
    
    Args:
        submit_url: The submission URL (extracted from page, NOT hallucinated!)
        email: Student email
        secret: Student secret
        quiz_url: The original quiz URL
        answer: The answer (can be bool, number, string, object, or base64)
    
    Returns:
        Response from the server containing:
        - correct: bool
        - url: Optional next URL
        - reason: Optional error reason
    """
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    logger.info(f"Submitting answer to: {submit_url}")
    logger.info(f"Quiz URL: {quiz_url}")
    logger.info(f"Answer type: {type(answer).__name__}")
    
    try:
        response = requests.post(
            submit_url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 405:
            logger.error(f"405 Method Not Allowed - WRONG URL! Got: {submit_url}")
            logger.error("This usually means we're POSTing to a question page instead of /submit")
            return {
                "correct": False,
                "reason": f"405 error - wrong submission URL: {submit_url}"
            }
        
        result = response.json()
        logger.info(f"Response: {result}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return {
            "correct": False,
            "reason": str(e)
        }
    except Exception as e:
        logger.error(f"Submission error: {e}")
        return {
            "correct": False,
            "reason": str(e)
        }
