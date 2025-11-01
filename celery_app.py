import os
import shutil
import tempfile
import yt_dlp
import subprocess
import time
import shlex
import redis
from celery import Celery
from celery.schedules import crontab
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

# -------------------------
# Celery setup - Simple version, reads full URLs from environment
# -------------------------
celery_app = Celery(
    "celery_app",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

celery_app.conf.worker_prefetch_multiplier = 1

# This is now handled by the worker's start command in render.yaml
COOKIE_REFRESH_HOURS = int(os.getenv("COOKIE_REFRESH_HOURS", "12"))
celery_app.conf.beat_schedule = {
    "refresh-cookies-every-12h": {
        "task": "refresh_cookies_task",
        "schedule": crontab(minute=0, hour=f"*/{COOKIE_REFRESH_HOURS}"),
    },
}

# -------------------------
# Redis and Locking
# -------------------------
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
COOKIE_LOCK = "cookie_refresh_lock"

# -------------------------
# Whisper / ffmpeg config
# -------------------------
WHISPER_CONFIG = {
    "model_path": os.getenv("WHISPER_MODEL_PATH", "/app/whisper.cpp/models/ggml-tiny.en-q8_0.bin"),
    "threads": int(os.getenv("WHISPER_THREADS", str(os.cpu_count() or 1))),
    "cli_path": os.getenv("WHISPER_CLI_PATH", "whisper-cli"),
    "language": os.getenv("WHISPER_LANGUAGE", ""),
    "translate": os.getenv("WHISPER_TRANSLATE", "false").lower() == "true",
    "split_on_word": os.getenv("WHISPER_SPLIT_ON_WORD", "false").lower() == "true",
    "max_len": os.getenv("WHISPER_MAX_LEN", "0"),
    "max_context": os.getenv("WHISPER_MAX_CONTEXT", "0"),
    "best_of": os.getenv("WHISPER_BEST_OF", ""),
    "beam_size": os.getenv("WHISPER_BEAM_SIZE", ""),
}

FFMPEG_PATH = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

# -------------------------
# Cookie Management
# -------------------------
COOKIES_FILE = "cookies.txt"
USE_BROWSER_COOKIES = os.getenv("USE_BROWSER_COOKIES", "true").lower() == "true"
BROWSER_NAME = os.getenv("BROWSER_NAME", "edge")

def refresh_cookies() -> None:
    if not USE_BROWSER_COOKIES:
        print("Cookie refresh skipped: USE_BROWSER_COOKIES is false.")
        return
    print("🔄 Attempting to refresh cookies.txt...")
    # ... rest of the function is the same ...
    try:
        with redis_client.lock(COOKIE_LOCK, timeout=60):
            print("Acquired lock, refreshing cookies...")
            cmd = [
                "yt-dlp",
                f"--cookies-from-browser={BROWSER_NAME}",
                "--max-downloads", "0",
                "--cookies", COOKIES_FILE,
                "https://www.tiktok.com",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ cookies.txt refreshed successfully.")
    except redis.exceptions.LockError:
        print("Could not acquire lock, another process is refreshing cookies.")
    except Exception as e:
        print(f"⚠️ Failed to refresh cookies automatically: {e}")


def ensure_cookies() -> str | None:
    if not USE_BROWSER_COOKIES:
        return None
    if not os.path.exists(COOKIES_FILE) or (time.time() - os.path.getmtime(COOKIES_FILE)) > (COOKIE_REFRESH_HOURS * 3600):
        refresh_cookies()
    return COOKIES_FILE if os.path.exists(COOKIES_FILE) else None

# ... all other helper functions (download_audio, etc.) and the transcribe_task remain exactly the same ...

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_audio(url: str) -> str:
    """Download audio using yt-dlp and return path to WAV file (with retries)."""
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "worstaudio/best",
        "outtmpl": tmp_path,
        "quiet": True,
        "noplaylist": True,
        "ffmpeg_location": FFMPEG_PATH,
        "retries": 5,
        "fragment_retries": 5,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "64",
            }
        ],
        "cookiefile": ensure_cookies()
    }
    if not ydl_opts["cookiefile"]:
        del ydl_opts["cookiefile"]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        wav_filename = f"{os.path.splitext(filename)[0]}.wav"
        if os.path.exists(wav_filename):
            return wav_filename
    
    raise FileNotFoundError("yt-dlp did not produce a WAV file")


def safe_download_audio(url: str) -> str:
    """Convert RetryError into a normal Exception."""
    try:
        return download_audio(url)
    except RetryError as e:
        last = e.last_attempt.exception()
        raise Exception(f"Download failed after retries: {last}") from last


def cleanup_temp_dir_by_audio(audio_path: str) -> None:
    """Remove temp dir containing the audio file."""
    try:
        tmpdir = os.path.dirname(audio_path)
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"Failed to cleanup temp dir: {e}")

def build_whisper_cmd(audio_path: str, out_base_path: str, fmt: str) -> list[str]:
    """Compose whisper-cli command list securely."""
    opts = {
        "--model": WHISPER_CONFIG["model_path"],
        "--threads": str(WHISPER_CONFIG["threads"]),
        "--output-file": out_base_path,
        "--language": WHISPER_CONFIG["language"] or None,
        "--max-context": WHISPER_CONFIG["max_context"] or None,
        "--max-len": WHISPER_CONFIG["max_len"] or None,
        "--best-of": WHISPER_CONFIG["best_of"] or None,
        "--beam-size": WHISPER_CONFIG["beam_size"] or None,
    }
    cmd = [shlex.quote(WHISPER_CONFIG["cli_path"])]
    for k, v in opts.items():
        if v is not None and str(v) != "0" and v != "":
            cmd.extend([k, shlex.quote(str(v))])

    if fmt == "plain":
        cmd.extend(["--output-txt", "--no-timestamps"])
    elif fmt == "timestamps":
        cmd.append("--output-srt")
    if WHISPER_CONFIG["translate"]:
        cmd.append("--translate")
    if WHISPER_CONFIG["split_on_word"]:
        cmd.append("--split-on-word")
        
    cmd.append(shlex.quote(audio_path))
    return cmd

@celery_app.task
def refresh_cookies_task():
    """Scheduled task to refresh cookies periodically."""
    refresh_cookies()
    return "Cookies refreshed"


@celery_app.task(
    bind=True,
    name="transcribe_task",
    acks_late=True,
    reject_on_worker_lost=True,
)
def transcribe_task(self, url: str, format: str = "plain") -> str:
    """
    Download audio and transcribe with whisper-cli.
    """
    audio_path = None
    try:
        self.update_state(state="PROGRESS", meta={"step": "downloading", "progress": 10})
        audio_path = safe_download_audio(url)

        self.update_state(state="PROGRESS", meta={"step": "transcribing", "progress": 50})

        output_base_path = os.path.splitext(audio_path)[0]
        cmd = build_whisper_cmd(audio_path, output_base_path, format)

        print("Running whisper-cli:", " ".join(cmd))
        proc = subprocess.run(" ".join(cmd), capture_output=True, text=True, check=True, shell=True)
        
        self.update_state(state="PROGRESS", meta={"step": "finalizing", "progress": 90})
        
        file_extension = "txt" if format == "plain" else "srt"
        transcript_file_path = f"{output_base_path}.{file_extension}"

        if not os.path.exists(transcript_file_path):
            raise FileNotFoundError(f"Transcript file not found at {transcript_file_path}. Stderr: {proc.stderr}")

        with open(transcript_file_path, "r", encoding="utf-8") as fh:
            transcript = fh.read().strip()

        return transcript or "No speech detected."
    except Exception as e:
        raise e
    finally:
        if audio_path:
            cleanup_temp_dir_by_audio(audio_path)
