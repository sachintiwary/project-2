"""
Flask API for God-Level Quiz Solver Agent
"""
import logging
import threading
from flask import Flask, request, jsonify

from config import MY_SECRET
from agent import solve_quiz

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        "name": "God-Level Quiz Solver Agent",
        "version": "3.0",
        "model": "gpt-5-mini",
        "architecture": "ReAct with Tool Calling",
        "endpoints": {
            "/": "This page",
            "/health": "Health check",
            "/solve": "POST - Solve quiz"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy", "agent": "ready"})


@app.route('/solve', methods=['POST'])
def solve():
    """
    Solve a quiz
    
    POST JSON: {email, secret, url}
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Expected JSON body"}), 400
        
        email = data.get('email')
        secret = data.get('secret')
        url = data.get('url')
        
        if not all([email, secret, url]):
            return jsonify({"error": "Missing: email, secret, or url"}), 400
        
        # Verify secret
        if secret != MY_SECRET:
            logger.warning(f"Invalid secret from {email}")
            return jsonify({"error": "Invalid secret"}), 403
        
        logger.info(f"Valid request from {email}")
        
        # Solve in background thread
        thread = threading.Thread(
            target=solve_quiz,
            args=(email, secret, url)
        )
        thread.start()
        
        return jsonify({
            "status": "processing",
            "message": "Quiz solving started",
            "url": url
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
