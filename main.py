import logging
import os
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from celery.result import AsyncResult
from kombu.exceptions import KombuError
import smtplib
from email.message import EmailMessage

from celery_app import celery_app, transcribe_task

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="Shorts2Text Transcriber", lifespan=lifespan)

# ==============================================================================
#  CORS – updated to include both frontend domains
# ==============================================================================
CLIENT_ORIGIN = os.getenv("CLIENT_ORIGIN", "*")
ADVERTISE_ORIGIN = os.getenv("ADVERTISE_ORIGIN", "https://websoitez-frontend.vercel.app")  # Update this to your Vercel domain for advertise.html

if CLIENT_ORIGIN == "*":
    logger.warning("CLIENT_ORIGIN is not set. Allowing all origins (OK for local dev).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_ORIGIN, ADVERTISE_ORIGIN] if CLIENT_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
#  Existing Pydantic models
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

# ==============================================================================
#  Existing endpoints
# ==============================================================================
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> dict[str, Any]:
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
        elif result.state == "UCCESS":
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

# ==============================================================================
#  NEW: Advertise inquiry endpoint
# ==============================================================================
class AdInquiry(BaseModel):
    message: str = Field(..., min_length=10, max_length=500)

@app.post("/advertise")
async def advertise_inquiry(payload: AdInquiry) -> dict[str, str]:
    """
    Receives ad-space inquiry and forwards it to your inbox
    plus an auto-reply chatbot address.
    """
    our_email = os.getenv("AD_EMAIL", "your-ads-email@example.com")
    bot_email = os.getenv("BOT_EMAIL", "chatbot@example.com")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    if not (smtp_user and smtp_pass):
        raise HTTPException(503, "Email service not configured")

    def send(to_addr: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        msg.set_content(body)
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)

    # 1. Forward to you
    send(
        our_email,
        "New Ad-Space Inquiry",
        f"Someone wants to buy ad space:\n\n{payload.message}\n\nReply to close the deal."
    )

    # 2. Auto-reply via chatbot
    send(
        bot_email,
        "Ad Inquiry Auto-Reply",
        f"Auto-reply triggered by:\n\n{payload.message}"
    )

    return {"status": "sent"}
