"""
LLM Quiz Solver - Main Flask Application
Receives quiz tasks and solves them using LLM + Playwright
"""
import os
import json
import logging
import threading
from flask import Flask, request, jsonify

from config import MY_SECRET
from solver import solve_quiz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "service": "LLM Quiz Solver",
        "status": "running",
        "endpoints": {
            "POST /solve": "Submit a quiz task",
            "GET /health": "Health check"
        }
    }), 200


@app.route('/solve', methods=['POST'])
def solve():
    """
    Main endpoint to receive quiz tasks
    
    Expected payload:
    {
        "email": "student@example.com",
        "secret": "your-secret",
        "url": "https://example.com/quiz-123"
    }
    """
    # Validate JSON
    try:
        data = request.get_json()
        if not data:
            logger.warning("Invalid JSON received")
            return jsonify({"error": "Invalid JSON"}), 400
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Extract fields
    email = data.get('email')
    secret = data.get('secret')
    url = data.get('url')
    
    # Validate required fields
    if not all([email, secret, url]):
        logger.warning(f"Missing required fields. Got: email={bool(email)}, secret={bool(secret)}, url={bool(url)}")
        return jsonify({"error": "Missing required fields"}), 400
    
    # Verify secret
    if secret != MY_SECRET:
        logger.warning(f"Invalid secret from {email}")
        return jsonify({"error": "Invalid secret"}), 403
    
    logger.info(f"Valid request from {email} for URL: {url}")
    
    # Start solving in background thread (non-blocking response)
    thread = threading.Thread(
        target=solve_quiz_background,
        args=(email, secret, url)
    )
    thread.daemon = True
    thread.start()
    
    # Return 200 immediately as required
    return jsonify({
        "status": "processing",
        "message": "Quiz solving started",
        "url": url
    }), 200


def solve_quiz_background(email: str, secret: str, url: str):
    """Background task to solve the quiz"""
    try:
        logger.info(f"Starting to solve quiz: {url}")
        solve_quiz(email, secret, url)
        logger.info(f"Completed solving quiz: {url}")
    except Exception as e:
        logger.error(f"Error solving quiz {url}: {e}", exc_info=True)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting LLM Quiz Solver on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
