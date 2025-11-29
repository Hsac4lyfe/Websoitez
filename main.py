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

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="Shorts2Text Transcriber", lifespan=lifespan)

# ---------- CORS ----------
CLIENT_ORIGIN = os.getenv("CLIENT_ORIGIN", "https://websoitez-frontend.vercel.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- models ----------
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

# ---------- routes ----------
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> dict[str, Any]:
    try:
        task = transcribe_task.delay(str(req.url), req.format)
        return {"task_id": task.id}
    except KombuError as exc:
        logger.exception("Celery task submission failed.")
        raise HTTPException(503, detail="Transcription service unavailable") from exc

@app.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str) -> dict[str, Any]:
    try:
        result = AsyncResult(task_id, app=celery_app)
        if result.state == "PENDING":
            return {"status": "pending", "progress": 0, "step": "queued"}
        if result.state == "PROGRESS":
            meta = result.info or {}
            return {"status": "processing", "progress": meta.get("progress", 0), "step": meta.get("step", "working")}
        if result.state == "SUCCESS":
            return {"status": "completed", "progress": 100, "step": "done", "transcript": result.get()}
        if result.state == "FAILURE":
            error_info = str(result.info) if result.info else "Unknown error"
            return {"status": "error", "progress": 100, "step": "failed", "error": error_info}
        return {"status": result.state.lower()}
    except Exception:
        logger.exception(f"Result fetch failed for {task_id}")
        raise HTTPException(500, detail="Internal error")

@app.get("/")
def root():
    return {"status": "ok"}
