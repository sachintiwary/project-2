import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.solver import QuizAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(title="IITM Data Agent")

# [CHANGE] Use Google Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
MY_SECRET = os.getenv("MY_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
def read_root():
    return {"status": "alive", "service": "data-agent-gemini"}

@app.post("/run-quiz")
def run_quiz(request: QuizRequest):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is missing!")
        raise HTTPException(status_code=500, detail="Server config error: No API Key")

    logger.info(f"Starting job for {request.email}")
    
    # Initialize Agent with Gemini Key
    agent = QuizAgent(api_key=GEMINI_API_KEY)
    
    result = agent.solve_recursive(request.url, request.email, request.secret)
    return result