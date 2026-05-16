# SparkleImage - AI Photo Repair & Enhancement
# Optimized for Linux ARM64 (Ampere) servers

FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel

# --- Runtime stage ---
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="SparkleImage Team"
LABEL description="AI Photo Repair, Recovery & Enhancement Tool"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user for security
RUN groupadd -r sparkle && useradd -r -g sparkle -d /app -s /sbin/nologin sparkle

# Copy built wheel and install
COPY --from=builder /build/dist/*.whl ./
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy application code
COPY app/ ./app/

# Create directories and set permissions
RUN mkdir -p data/uploads data/output data/config && \
    chown -R sparkle:sparkle /app

USER sparkle

# Expose Gradio port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

ENTRYPOINT ["python", "-m", "app.main"]
