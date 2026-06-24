import subprocess
import os
import asyncio
from app.core.database import init_db

PORT = os.environ.get("PORT", "10000")

gunicorn_cmd = [
    "gunicorn",
    "main:app",
    "-k", "uvicorn.workers.UvicornWorker",
    "--bind", f"0.0.0.0:{PORT}",
    "--timeout", "90",
    "--workers", "1"]

# Run database initialization sequentially BEFORE starting workers
# This prevents race conditions where multiple workers try to create tables simultaneously
asyncio.run(init_db())

gunicorn_proc = subprocess.Popen(gunicorn_cmd)

# Start Celery worker
celery_cmd = [
    "celery",
    "-A", "app.core.worker",
    "worker",
    "--pool=solo",
    "--loglevel=info"
]

celery_proc = subprocess.Popen(celery_cmd)

# Wait for both
gunicorn_proc.wait()
celery_proc.wait()