import subprocess
import os

# Use Render dynamic port
PORT = os.environ.get("PORT", "10000")

# Start FastAPI (Gunicorn + Uvicorn workers)
gunicorn_cmd = [
    "gunicorn",
    "main:app",
    "-k", "uvicorn.workers.UvicornWorker",
    "--bind", f"0.0.0.0:{PORT}",
    "--timeout", "90",
    "--workers", "2"
]

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