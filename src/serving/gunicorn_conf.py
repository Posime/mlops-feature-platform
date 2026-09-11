import multiprocessing
import os

# Network Binding
bind = os.getenv("BIND", "0.0.0.0:8000")

# Worker Concurrency
# Formula: (2 * CPU Cores) + 1, capped to prevent CPU thrashing during ONNX execution
cores = multiprocessing.cpu_count()
default_workers = max(2, min(4, (2 * cores) + 1))
workers = int(os.getenv("WEB_CONCURRENCY", default_workers))
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts & Keep-Alive
timeout = int(os.getenv("TIMEOUT", 30))
keepalive = int(os.getenv("KEEP_ALIVE", 5))

# Memory Leak Prevention: recycle worker after handling N requests
max_requests = int(os.getenv("MAX_REQUESTS", 2000))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", 200))

# Logging Configuration
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"