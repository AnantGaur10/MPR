# ============================================================
# Sign Language Backend - Docker Image
# ============================================================
# Multi-stage build for a lean production image
# Uses Python 3.11 slim + system deps for OpenCV / MediaPipe
# ============================================================

# ---------- Stage 1: Builder ----------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim-bookworm

LABEL maintainer="Sign Language App"
LABEL description="Sign Language to Audio Backend Service"

# Runtime system dependencies for OpenCV, MediaPipe, and pyttsx3/espeak
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    espeak \
    libespeak1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY app.py .
COPY auth.py .
COPY auth_routes.py .
COPY database.py .
COPY models.py .
COPY sign_routes.py .
COPY utils.py .

# Copy the ML model file(s)
# Note: app.py handles checking CWD or parent dir for these
COPY model.p .
COPY model1.p .
COPY model2.p .

# Copy templates
COPY templates/ templates/

# Create runtime directories
RUN mkdir -p user_data downloads

# Expose Flask port
EXPOSE 8080

# Environment defaults (can be overridden via docker-compose or docker run -e)
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0
ENV PYTHONUNBUFFERED=1

# Health check (model-status endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/api/model-status || exit 1

# Run the application
CMD ["python", "app.py"]
