import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.solver import QuizAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="IITM Data Agent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
MY_SECRET = os.getenv("MY_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

def run_agent_background(url: str, email: str, secret: str):
    """Wrapper to run the agent in the background"""
    logger.info(f"Background task started for {url}")
    try:
        agent = QuizAgent(api_key=GEMINI_API_KEY)
        agent.solve_recursive(url, email, secret)
    except Exception as e:
        logger.error(f"Background task failed: {e}")

@app.get("/")
def read_root():
    return {"status": "alive", "service": "data-agent-gemini"}

@app.post("/run-quiz")
def run_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    # 1. Verify Secret
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server config error")

    logger.info(f"Received job for {request.email}. Delegating to background.")
    
    # 2. Add to Background Tasks (Non-blocking)
    background_tasks.add_task(run_agent_background, request.url, request.email, request.secret)
    
    # 3. Return 200 OK immediately
    return {
        "message": "Job accepted. Processing in background.",
        "status": "processing"
    }