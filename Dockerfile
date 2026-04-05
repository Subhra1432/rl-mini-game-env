# Multi-stage build for Email Triage Environment
# Supports both Hugging Face Spaces (port 7860) and local (port 8000) deployment

FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy application code first so pip has the source correctly for installation
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[baseline]"

# ---- Runtime stage ----
FROM python:3.10-slim

WORKDIR /app

# Create non-root user (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy environment code
COPY --chown=user . $HOME/app

# Set PYTHONPATH
ENV PYTHONPATH="$HOME/app:$PYTHONPATH"

# Default port (7860 for HF Spaces, override with PORT env var)
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://localhost:{__import__(\"os\").environ.get(\"PORT\", 7860)}/health')" || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd $HOME/app && uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
