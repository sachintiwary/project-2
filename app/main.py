import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.solver import QuizAgent

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="IITM Data Agent",
    description="Autonomous Quiz Solver",
    version="2.0.0"
)

# Secrets
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN") or os.getenv("OPENAI_API_KEY")
MY_SECRET = os.getenv("MY_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
def read_root():
    return {"status": "alive", "service": "data-agent-v2"}

@app.post("/run-quiz")
def run_quiz(request: QuizRequest):
    """
    Standard entry point.
    FastAPI runs standard 'def' functions in a threadpool automatically.
    This ensures Playwright (sync) doesn't block the Async event loop.
    """
    if request.secret != MY_SECRET:
        logger.warning(f"Unauthorized access attempt: {request.secret}")
        raise HTTPException(status_code=403, detail="Invalid Secret")

    if not AIPIPE_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfiguration: No API Token")

    logger.info(f"Starting job for {request.email} at {request.url}")
    
    agent = QuizAgent(api_key=AIPIPE_TOKEN)
    
    # Run the recursive solver
    result = agent.solve_recursive(request.url, request.email, request.secret)
    
    if result.get("status") == "error":
         # We return 200 with error details because the calling server expects JSON response
         # even on failure, usually.
         return result
         
    return result