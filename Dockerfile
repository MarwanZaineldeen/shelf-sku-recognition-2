# ==============================================================================
# RETAIL AI PLATFORM — BACKEND SERVICE DOCKERFILE
# ==============================================================================
# Base image: Python 3.11 slim
FROM python:3.11-slim AS base

# Prevent python from writing pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RETAIL_AI_ROOT=/app \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    PORT=5000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies required by OpenCV, PyTorch, and Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source code & configurations
COPY server/ ./server/
COPY configs/ ./configs/
COPY ml/ ./ml/
COPY scripts/ ./scripts/

# Expose internal service port
EXPOSE 5000

# Healthcheck endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

# Launch FastAPI application server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "5000"]
