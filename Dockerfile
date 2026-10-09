# SparkleImage — AI photo repair & enhancement
# Optimised for Linux ARM64 (Ampere) servers.

FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app/ ./app/
RUN pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel

# --- Face models ---
# YuNet (detector, ~230KB) and insightface's ArcFace w600k_r50 (identity,
# 166MB, shipped only inside the 275MB buffalo_l zip — the rest is discarded).
# Both are required: a build that can't fetch them fails rather than shipping
# an app that silently can't compare faces. ArcFace weights are licensed for
# non-commercial use only.
FROM debian:bookworm-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models && cd /tmp \
    && curl -fsSL -o /models/face_detection_yunet_2023mar.onnx \
       https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    && curl -fsSL -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    && unzip -j buffalo_l.zip w600k_r50.onnx -d /models \
    && rm buffalo_l.zip

# --- Runtime ---
FROM python:3.11-slim-bookworm AS runtime

LABEL description="AI photo repair, recovery and enhancement"

# opencv-python-headless needs no GUI libraries — only libgomp for its
# threading. The previous image pulled in libgl1/libsm6/libxext6 to satisfy the
# non-headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The wheel is the only copy of the application code — the previous image
# installed the wheel *and* copied app/ over the top of it.
COPY --from=builder /build/dist/*.whl ./
RUN pip install --no-cache-dir *.whl && rm *.whl

# Face models live in /app/models, not /app/data/models: compose bind-mounts
# ./data over /app/data, which would hide anything baked in underneath it.
COPY --from=models /models /app/models

# No usage reporting to Hugging Face from a private photo app.
ENV GRADIO_ANALYTICS_ENABLED=False

# Run as a normal user. Bind-mounted host directories must be writable by this
# uid; docker-compose sets it from the host so the container need not run as root.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "${APP_GID}" sparkle 2>/dev/null || true && \
    useradd -u "${APP_UID}" -g "${APP_GID}" -d /app -s /sbin/nologin sparkle 2>/dev/null || true && \
    mkdir -p /app/data/jobs /app/data/output && \
    chown -R "${APP_UID}:${APP_GID}" /app
USER ${APP_UID}:${APP_GID}

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

ENTRYPOINT ["python", "-m", "app.main"]
