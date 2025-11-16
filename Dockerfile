# Multi-stage Dockerfile optimized for Raspberry Pi 5 (ARM64)
# Target: Minimal image size with production-ready inference engine

# ============================================================================
# Stage 1: Builder - Compile dependencies
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.0

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry (no virtualenv in Docker)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim-bookworm

# Install runtime dependencies only
# - libgomp1: Required for OpenMP in ONNX Runtime
# - libglib2.0-0, libsm6, libxext6, libxrender-dev: Required for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash bsort

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=bsort:bsort . .

# Copy configuration
COPY --chown=bsort:bsort settings.yaml ./

# Create directories
RUN mkdir -p data/raw data/processed data/models runs && \
    chown -R bsort:bsort /app

# Switch to non-root user
USER bsort

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORT_NUM_THREADS=4

# Expose port if needed (for future web interface)
EXPOSE 8000

# Health check (basic Python import test)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import bsort; print('OK')" || exit 1

# Default command
CMD ["bsort", "--help"]
