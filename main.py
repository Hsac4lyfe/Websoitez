import logging
import os
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from celery.result import AsyncResult
from kombu.exceptions import KombuError

from celery_app import celery_app, transcribe_task

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logic to run on startup
    logger.info("Application starting up...")
    yield
    # Logic to run on shutdown
    logger.info("Application shutting down...")


app = FastAPI(title="Shorts2Text Transcriber", lifespan=lifespan)

# ==============================================================================
# ✅ THIS IS THE MODIFIED SECTION FOR RENDER DEPLOYMENT
# It reads the frontend's URL from an environment variable set by Render.
# ==============================================================================
CLIENT_ORIGIN = os.getenv("CLIENT_ORIGIN", "*")
if CLIENT_ORIGIN == "*":
    logger.warning("CLIENT_ORIGIN is not set. Allowing all origins (OK for local dev).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_ORIGIN] if CLIENT_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==============================================================================


class TranscribeRequest(BaseModel):
    url: HttpUrl
    format: str = "plain"


class TranscribeResponse(BaseModel):
    task_id: str


class ResultResponse(BaseModel):
    status: str
    progress: int | None = None
    step: str | None = None
    transcript: str | None = None
    error: str | None = None


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> dict[str, Any]:
    """
    Accepts a URL and format, and dispatches a transcription task.
    """
    try:
        task = transcribe_task.delay(str(req.url), req.format)
        return {"task_id": task.id}
    except KombuError as exc:
        logger.exception("Celery task submission failed.")
        raise HTTPException(
            status_code=503,
            detail="Transcription service is currently unavailable. Please try again later."
        ) from exc


@app.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str) -> dict[str, Any]:
    """
    Retrieves the result of a transcription job, including progress.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if result.state == "PENDING":
            return {"status": "pending", "progress": 0, "step": "queued"}

        elif result.state == "PROGRESS":
            meta = result.info or {}
            return {
                "status": "processing",
                "progress": meta.get("progress", 0),
                "step": meta.get("step", "working"),
            }

        elif result.state == "SUCCESS":
            return {
                "status": "completed",
                "progress": 100,
                "step": "done",
                "transcript": result.get(),
            }

        elif result.state == "FAILURE":
            error_info = str(result.info) if result.info else "An unknown error occurred."
            return {
                "status": "error",
                "progress": 100,
                "step": "failed",
                "error": error_info,
            }

        return {"status": result.state.lower()}

    except Exception:
        logger.exception(f"An error occurred while fetching result for task_id: {task_id}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@app.get("/")
def root():
    return {"status": "ok"}