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
# Celery setup - Reads full URLs directly from environment variables.
# This works perfectly for both local Docker and Railway.
# -------------------------
celery_app = Celery(
    "celery_app",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

celery_app.conf.worker_prefetch_multiplier = 1

# -------------------------
# NEW: Conditional Beat Schedule for Cookie Refresh
# This allows disabling the cookie refresh task entirely for cloud deployments
# where browser-based cookie extraction is not feasible or desired.
# Set USE_BROWSER_COOKIES=false in your Railway env vars to disable.
# -------------------------
USE_BROWSER_COOKIES_ENABLED = os.getenv("USE_BROWSER_COOKIES", "true").lower() == "true"
COOKIE_REFRESH_HOURS = int(os.getenv("COOKIE_REFRESH_HOURS", "12"))

if USE_BROWSER_COOKIES_ENABLED:
    print(f"Cookie refresh beat schedule is ENABLED. Will refresh every {COOKIE_REFRESH_HOURS} hours.")
    celery_app.conf.beat_schedule = {
        "refresh-cookies-every-12h": {
            "task": "celery_app.refresh_cookies_task", # Explicitly name task with module path
            "schedule": crontab(minute=0, hour=f"*/{COOKIE_REFRESH_HOURS}"),
        },
    }
else:
    print("Cookie refresh beat schedule is DISABLED as USE_BROWSER_COOKIES is false.")
    celery_app.conf.beat_schedule = {} # Empty schedule if not using browser cookies

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
# Use the global flag to control if browser cookies are expected/used
USE_BROWSER_COOKIES = USE_BROWSER_COOKIES_ENABLED
BROWSER_NAME = os.getenv("BROWSER_NAME", "edge")

def refresh_cookies() -> None:
    """
    Attempts to refresh cookies.txt using yt-dlp's browser cookie extraction.
    This function should only be called if USE_BROWSER_COOKIES is true.
    """
    if not USE_BROWSER_COOKIES:
        print("Cookie refresh skipped: USE_BROWSER_COOKIES is false. Function was called unexpectedly.")
        return

    print(f"🔄 Attempting to refresh cookies.txt using {BROWSER_NAME} browser...")
    try:
        with redis_client.lock(COOKIE_LOCK, timeout=60):
            print("Acquired lock, refreshing cookies...")
            cmd = [
                "yt-dlp",
                f"--cookies-from-browser={BROWSER_NAME}",
                "--max-downloads", "0",
                "--cookies", COOKIES_FILE,
                "https://www.youtube.com", # Changed target to YouTube for better relevance
            ]
            # Use shell=False for security, shlex.quote already handles this if cmd is string.
            # But since cmd is a list, subprocess handles quoting.
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ cookies.txt refreshed successfully.")
    except redis.exceptions.LockError:
        print("Could not acquire lock, another process is refreshing cookies.")
    except Exception as e:
        print(f"⚠️ Failed to refresh cookies automatically: {e}. "
              f"Ensure a browser ({BROWSER_NAME}) is available and logged in if USE_BROWSER_COOKIES is true.")


def ensure_cookies() -> str | None:
    """
    Checks if a cookie file is needed and attempts to refresh it if expired.
    Returns the cookie file path if browser cookies are enabled and available, else None.
    """
    if not USE_BROWSER_COOKIES:
        return None # If browser cookies are disabled, no cookie file is ever used for downloads.
    
    # If browser cookies are enabled, proceed with refresh logic.
    # The refresh_cookies_task will handle periodic updates.
    # This check ensures we have a recent cookie file if needed for a download.
    if not os.path.exists(COOKIES_FILE) or (time.time() - os.path.getmtime(COOKIES_FILE)) > (COOKIE_REFRESH_HOURS * 3600):
        refresh_cookies() # Attempt immediate refresh if file is missing or old

    return COOKIES_FILE if os.path.exists(COOKIES_FILE) else None

# --- ALL OTHER HELPER FUNCTIONS AND TASKS ARE UNCHANGED ---

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_audio(url: str) -> str:
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
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "64"}],
        "cookiefile": ensure_cookies() # This will be None if USE_BROWSER_COOKIES is false
    }
    if not ydl_opts["cookiefile"]:
        del ydl_opts["cookiefile"] # Remove the key entirely if no cookie file is provided
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        wav_filename = f"{os.path.splitext(filename)[0]}.wav"
        if os.path.exists(wav_filename):
            return wav_filename
    raise FileNotFoundError("yt-dlp did not produce a WAV file")

def safe_download_audio(url: str) -> str:
    try:
        return download_audio(url)
    except RetryError as e:
        last = e.last_attempt.exception()
        raise Exception(f"Download failed after retries: {last}") from last

def cleanup_temp_dir_by_audio(audio_path: str) -> None:
    try:
        tmpdir = os.path.dirname(audio_path)
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"Failed to cleanup temp dir: {e}")

def build_whisper_cmd(audio_path: str, out_base_path: str, fmt: str) -> list[str]:
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
    # It's better to pass command as a list for subprocess.run(..., shell=False)
    # The current code uses shell=True, so it expects a string.
    # Let's keep it as a string for now to avoid breaking existing shell=True usage,
    # but be aware that shell=True can be a security risk with untrusted input.
    cmd_parts = [WHISPER_CONFIG["cli_path"]]
    for k, v in opts.items():
        if v is not None and str(v) != "0" and v != "":
            cmd_parts.extend([k, str(v)]) # str() will handle quoting if shell=True, but safer as list
    if fmt == "plain":
        cmd_parts.extend(["--output-txt", "--no-timestamps"])
    elif fmt == "timestamps":
        cmd_parts.append("--output-srt")
    if WHISPER_CONFIG["translate"]:
        cmd_parts.append("--translate")
    if WHISPER_CONFIG["split_on_word"]:
        cmd_parts.append("--split-on-word")
    cmd_parts.append(audio_path)
    
    # If subprocess.run is called with shell=True, it expects a single string.
    # Otherwise, it expects a list of strings.
    # Your current call uses " ".join(cmd), which implies shell=True.
    # For robust list-based command, you'd do: subprocess.run(cmd_parts, shell=False)
    # For now, sticking to your existing subprocess.run(" ".join(cmd), shell=True) pattern.
    return [shlex.quote(part) for part in cmd_parts] # Re-quote for safety if " ".join and shell=True

# IMPORTANT: Explicitly name the task here.
@celery_app.task(name="celery_app.refresh_cookies_task")
def refresh_cookies_task():
    """
    Celery task to periodically refresh cookies. Only scheduled if
    USE_BROWSER_COOKIES_ENABLED is true.
    """
    refresh_cookies()
    return "Cookies refreshed"

@celery_app.task(bind=True, name="transcribe_task", acks_late=True, reject_on_worker_lost=True)
def transcribe_task(self, url: str, format: str = "plain") -> str:
    audio_path = None
    try:
        self.update_state(state="PROGRESS", meta={"step": "downloading", "progress": 10})
        audio_path = safe_download_audio(url)
        self.update_state(state="PROGRESS", meta={"step": "transcribing", "progress": 50})
        output_base_path = os.path.splitext(audio_path)[0]
        
        # build_whisper_cmd now returns a list of quoted strings
        cmd_list = build_whisper_cmd(audio_path, output_base_path, format)
        cmd_str = " ".join(cmd_list) # Join for shell=True
        
        print("Running whisper-cli:", cmd_str) # Print the joined command string
        proc = subprocess.run(cmd_str, capture_output=True, text=True, check=True, shell=True)
        
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
